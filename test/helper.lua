-- Let `require` resolve Fennel modules (e.g. require("fnl.tensor") -> fnl/tensor.fnl)
-- transparently, compiling them on demand.
require("fennel").install()

local assert = require("luassert")
local tensor = require("test.impl").tensor

-- The core keeps its node metatable private (no `mt` export), so read it off a
-- sample node to teach luassert how to pretty-print tensors in failure messages.
local tensor_mt = getmetatable(tensor.scalar(0))

assert:add_formatter(function(value)
  if getmetatable(value) == tensor_mt then
    return tostring(value)
  end
end)
