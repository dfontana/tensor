local tensor = require("lua.tensor")
---@alias ParamMap {[string]: Tensor}

---@class Optimizer
---@field params ParamMap
---@field learningRate number
local Optimizer = {}
Optimizer.__index = Optimizer

---@constructor
---@param params ParamMap Map of name to parameter tensor (the items being
---@param learningRate number Rate of learning optimized)
function Optimizer.new(params, learningRate)
  assert(type(params) == "table")
  assert(type(learningRate) == "number")
  local self = {
    params = params,
    learningRate = learningRate,
  }
  return setmetatable(self, Optimizer)
end

---Zero the parameter gradients
function Optimizer:zero()
  for _, p in pairs(self.params) do
    p:zero_grad()
  end
end

---Apply learnings to parameters
function Optimizer:step()
  for _, p in pairs(self.params) do
    if p.gradient then
      for i = 1, #p.data do
        p.data[i] = p.data[i] - self.learningRate * p.gradient.data[i]
      end
    end
  end
end

---@param pred Tensor Prediction value
---@param actual Tensor Actual value
---@return Tensor (scaler)
local function mean_squared(pred, actual)
  return pred
      :sub(actual)
      :pow(tensor.scalar(2))
      :mean()
end

return {
  Optimizer = Optimizer,
  mean_squared = mean_squared,
}
