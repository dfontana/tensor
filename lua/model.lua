local tensor = require("lua.tensor")
---A model is anything that can compute a forward
---  pass and return a tensor; and tell me what it's
---  parameters are. Anything else is fair game.

---@class Model
---@field forward fun(self: Model, x: Tensor): Tensor
---@field parameters fun(self: Model): Tensor[]


---Note: assumes 2D, N-D comes later
---@param in_feats number
---@param out_feats number
---@return Model
local function linear(in_feats, out_feats)
  local weight = tensor.uniform(
    { in_feats, out_feats },
    -0.1,
    0.1
  )
  local bias = tensor.zeroes({ out_feats })
  return {
    ---@param x Tensor input data
    ---@return Tensor
    forward = function(self, x)
      return x:matmul(weight):add(bias)
    end,
    ---@return Tensor[]
    parameters = function(self)
      return { weight, bias }
    end
  }
end

---@param ... Model[]
---@return Model
local function sequential(...)
  local models = { ... }
  return {
    ---@param x Tensor input data
    ---@return Tensor
    forward = function(self, x)
      local next = x
      for _, m in ipairs(models) do
        next = m:forward(next)
      end
      return next
    end,
    ---@return Tensor[]
    parameters = function(self)
      local params = {}
      for _, m in ipairs(models) do
        for _, p in ipairs(m:parameters()) do
          table.insert(params, p)
        end
      end
      return params
    end
  }
end

---@return Model
local function relu()
  return {
    forward = function(self, x)
      return x:relu()
    end,

    parameters = function(self)
      return {}
    end
  }
end

return {
  Linear = linear,
  Sequential = sequential,
  ReLU = relu
}
