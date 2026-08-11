use mlua::prelude::*;

/// A minimal native function exposed to Lua as `tensor_native.hello()`.
pub fn hello() -> &'static str {
    "Hello, world!"
}

#[mlua::lua_module(name = "tensor_native")]
fn tensor_native(lua: &Lua) -> LuaResult<LuaTable> {
    let exports = lua.create_table()?;
    exports.set("hello", lua.create_function(|_, ()| Ok(hello()))?)?;
    Ok(exports)
}

#[cfg(test)]
mod tests {
    use super::hello;

    #[test]
    fn returns_a_hello_world_greeting() {
        assert_eq!(hello(), "Hello, world!");
    }
}
