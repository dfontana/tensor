use std::convert::TryFrom;

use mlua::{Error, Lua, MetaMethod, Result, Table, UserData, UserDataMethods, Value};

/// Fixed-size, Lua-GC-owned storage for native tensor values.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Storage(Vec<f32>);

impl Storage {
    pub(crate) fn from_table(table: &Table) -> Result<Self> {
        let length = table.raw_len();
        let length_as_integer =
            i64::try_from(length).map_err(|_| Error::runtime("storage table is too large"))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(length)
            .map_err(|_| Error::runtime("could not allocate storage"))?;

        for index in 1..=length {
            let key =
                i64::try_from(index).map_err(|_| Error::runtime("storage table is too large"))?;
            let value: Value = table.raw_get(key)?;
            if value == Value::Nil {
                return Err(Error::runtime("storage table must be a dense sequence"));
            }
            values.push(value_to_f32(value)?);
        }

        // `raw_len` is only a sequence boundary, so inspect every key to reject
        // holes and unrelated hash entries deterministically.
        for pair in table.pairs::<Value, Value>() {
            let (key, _) = pair?;
            match key {
                Value::Integer(index) if (1..=length_as_integer).contains(&index) => {}
                _ => {
                    return Err(Error::runtime(
                        "storage table must contain only 1-based sequence keys",
                    ));
                }
            }
        }

        Ok(Self(values))
    }

    pub(crate) fn as_slice(&self) -> &[f32] {
        &self.0
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.0
    }

    fn zeros(length: usize) -> Result<Self> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(length)
            .map_err(|_| Error::runtime("could not allocate storage"))?;
        values.resize(length, 0.0);
        Ok(Self(values))
    }
}

/// Convert one Lua number using the storage f32 narrowing policy.
pub(crate) fn value_to_f32(value: Value) -> Result<f32> {
    let number = match value {
        Value::Integer(integer) => integer as f64,
        Value::Number(number) => number,
        other => {
            return Err(Error::runtime(format!(
                "storage value must be a Lua number, got {}",
                other.type_name()
            )));
        }
    };

    let f32_max = f32::MAX as f64;
    if number.is_finite() && number.abs() > f32_max {
        return Err(Error::runtime("finite storage value overflows f32"));
    }

    Ok(number as f32)
}

fn storage_index(key: Value, length: usize) -> Result<Option<usize>> {
    let index = match key {
        Value::Integer(index) => index,
        Value::Number(number) if number.is_finite() && number.fract() == 0.0 => {
            if number < i64::MIN as f64 || number > i64::MAX as f64 {
                return Ok(None);
            }
            number as i64
        }
        Value::Number(_) => return Err(Error::runtime("storage index must be an integer")),
        _ => return Ok(None),
    };
    if index < 1 {
        return Ok(None);
    }

    let Some(zero_based) = usize::try_from(index - 1).ok() else {
        return Ok(None);
    };
    if zero_based >= length {
        Ok(None)
    } else {
        Ok(Some(zero_based))
    }
}

fn writable_storage_index(key: Value, length: usize) -> Result<usize> {
    match storage_index(key, length)? {
        Some(index) => Ok(index),
        None => Err(Error::runtime("storage index is out of bounds")),
    }
}

fn storage_length(length: usize) -> Result<i64> {
    i64::try_from(length).map_err(|_| Error::runtime("storage is too large"))
}

fn storage(lua: &Lua, value: Value) -> Result<Value> {
    match value {
        Value::UserData(userdata) if userdata.is::<Storage>() => Ok(Value::UserData(userdata)),
        Value::Table(table) => Ok(Value::UserData(
            lua.create_userdata(Storage::from_table(&table)?)?,
        )),
        other => Err(Error::runtime(format!(
            "storage expects a table or Storage, got {}",
            other.type_name()
        ))),
    }
}

fn storage_zeros(lua: &Lua, length: Value) -> Result<Value> {
    let Value::Integer(length) = length else {
        return Err(Error::runtime(
            "storage_zeros expects a nonnegative integer length",
        ));
    };
    let length = usize::try_from(length)
        .map_err(|_| Error::runtime("storage_zeros expects a nonnegative integer length"))?;
    Ok(Value::UserData(
        lua.create_userdata(Storage::zeros(length)?)?,
    ))
}

fn is_storage(value: Value) -> bool {
    matches!(value, Value::UserData(userdata) if userdata.is::<Storage>())
}

pub(crate) fn register(lua: &Lua, exports: &Table) -> Result<()> {
    exports.set("storage", lua.create_function(storage)?)?;
    exports.set("storage_zeros", lua.create_function(storage_zeros)?)?;
    exports.set(
        "is_storage",
        lua.create_function(|_, value: Value| Ok(is_storage(value)))?,
    )?;
    Ok(())
}

impl UserData for Storage {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("len", |_, this, ()| storage_length(this.as_slice().len()));
        methods.add_method("clone", |lua, this, ()| lua.create_userdata(this.clone()));
        methods.add_method("to_table", |lua, this, ()| {
            lua.create_sequence_from(this.as_slice().iter().copied())
        });
        methods.add_method_mut("fill", |_, this, value: Value| {
            let value = value_to_f32(value)?;
            this.as_mut_slice().fill(value);
            Ok(())
        });

        methods.add_meta_method(MetaMethod::Index, |_, this, key: Value| {
            Ok(match storage_index(key, this.as_slice().len())? {
                Some(index) => Value::Number(this.as_slice()[index] as f64),
                None => Value::Nil,
            })
        });
        methods.add_meta_method_mut(
            MetaMethod::NewIndex,
            |_, this, (key, value): (Value, Value)| {
                let index = writable_storage_index(key, this.as_slice().len())?;
                this.as_mut_slice()[index] = value_to_f32(value)?;
                Ok(())
            },
        );
        methods.add_meta_method(MetaMethod::Len, |_, this, ()| {
            storage_length(this.as_slice().len())
        });
        methods.add_meta_method(MetaMethod::Pairs, |lua, this, ()| {
            let values = this.as_slice().to_vec();
            let iterator = lua.create_function(move |_, (_state, previous): (Value, Value)| {
                let Value::Integer(previous) = previous else {
                    return Err(Error::runtime("storage pairs key must be an integer"));
                };
                if previous < 0 {
                    return Err(Error::runtime("storage pairs key must be nonnegative"));
                }
                let Some(next) = previous.checked_add(1) else {
                    return Ok((Value::Nil, Value::Nil));
                };
                let Some(index) = usize::try_from(next - 1).ok() else {
                    return Ok((Value::Nil, Value::Nil));
                };
                if let Some(value) = values.get(index) {
                    Ok((Value::Integer(next), Value::Number(*value as f64)))
                } else {
                    Ok((Value::Nil, Value::Nil))
                }
            })?;
            Ok((iterator, Value::Nil, Value::Integer(0)))
        });
        methods.add_meta_method(MetaMethod::ToString, |_, this, ()| {
            Ok(format!("Storage(len={})", this.as_slice().len()))
        });
    }
}

#[cfg(test)]
mod tests {
    use super::Storage;

    #[test]
    fn exposes_fixed_size_slices() {
        let mut storage = Storage(vec![1.0, 2.0]);
        assert_eq!(storage.as_slice(), &[1.0, 2.0]);
        storage.as_mut_slice()[1] = 4.5;
        assert_eq!(storage.as_slice(), &[1.0, 4.5]);
    }
}
