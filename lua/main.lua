local tensor = require("lua.tensor")

local a = tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 })
local b = tensor.new({ 3, 2 }, { 7, 8, 9, 10, 11, 12 })
print(a:matmul(b))
