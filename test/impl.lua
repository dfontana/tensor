-- The port is complete: the Fennel implementation is the only implementation.
-- Specs call the core's free functions directly (kebab / `!` / `?` names reached
-- by bracket access from Lua), so there is no method shim any more.
return {
  tensor = require("fnl.tensor"),
  optimizer = require("fnl.optimizer"),
  vocab = require("fnl.vocab"),
}
