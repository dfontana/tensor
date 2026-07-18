local assert = require("luassert")
local tensor = require("lua.tensor")

assert:add_formatter(function(value)
  if getmetatable(value) == tensor then
    return tostring(value)
  end
end)
