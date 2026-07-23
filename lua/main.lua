local tensor = require("lua.tensor")
local optimizer = require('lua.optimizer')


---@param params ParamMap
---@param x Tensor input data
---@param y Tensor actual output
---@return Tensor (specifically a loss scalar)
local function forward(params, x, y)
  -- We're going to fit m,b
  -- given known correct data x -> y
  -- Where relationship is y = m*x+b

  -- Our loss function will be mean((y_p - y)^2)
  local y_p = params['m']:mul(x):add(params['b'])
  return y_p:sub(y):pow(tensor.scalar(2)):mean()
end

local params = {
  m = tensor.uniform({}, -1, 1),
  b = tensor.uniform({}, -1, 1),
}
local x = tensor.new({ 3 }, { 1, 2, 3 })
local y = tensor.new({ 3 }, { 3, 5, 7 })

local sgd = optimizer.new(params, 0.01)
for i = 1, 100 do
  sgd:zero()
  local loss = forward(params, x, y)
  loss:backwards()
  sgd:step()
  print(i, loss.data[1], params.m.data[1], params.b.data[1])
end
