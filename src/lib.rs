#![cfg_attr(test, allow(dead_code, unused_imports))]

use mlua::prelude::*;

pub(crate) mod kernels;
pub(crate) mod storage;

/// A minimal native function exposed to Lua as `tensor_native.hello()`.
pub fn hello() -> &'static str {
    "Hello, world!"
}

#[cfg(not(test))]
#[mlua::lua_module(name = "tensor_native")]
fn tensor_native(lua: &Lua) -> LuaResult<LuaTable> {
    let exports = lua.create_table()?;
    exports.set("hello", lua.create_function(|_, ()| Ok(hello()))?)?;
    storage::register(lua, &exports)?;
    let level = fearless_simd::Level::new();
    kernels::register(lua, &exports, level)?;
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
