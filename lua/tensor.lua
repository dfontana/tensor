---@alias Shape number[]
---@alias Data number[]

---@class Tensor
---@field shape Shape Shape of the data
---@field data Data flat array of shape
---@field parents Tensor[] Parents that created this tensor, if any
---@field gradient Tensor?, only nil if this is a gradient tensor
---@field _backward fun():nil backwards prop operation
local Tensor = {}
-- TODO: Operator overloading
Tensor.__index = Tensor

---@param shape Shape of tensor
---@return Tensor
function Tensor.zeroes(shape)
  assert(type(shape) == "table")
  local data = {}
  if #shape == 0 then
    data[1] = 0
  else
    for i = 1, shape[1] * shape[2] do
      data[i] = 0
    end
  end
  local self = {
    shape = shape,
    data = data,
    parents = {},
    gradient = nil,
    _backward = function() end
  }
  return setmetatable(self, Tensor)
end

---@constructor
---@param shape Shape of tensor
---@param data Data flat array of data for tensor
function Tensor.new(shape, data)
  assert(type(shape) == "table")
  assert(type(data) == "table")
  local self = {
    shape = shape,
    data = data,
    parents = {},
    gradient = Tensor.zeroes(shape),
    --TODO: this consumes memory, long term an opcode might be better
    _backward = function() end
  }
  return setmetatable(self, Tensor)
end

---@param row number Row
---@param col number Column
---@param shape Shape (to locate offset inside)
---@return number The offset of row/col in the given shape
local function index_of(row, col, shape)
  return ((row - 1) * shape[2]) + col
end

---@param row number Row
---@param col number Column
---@return number
-- TODO: Return index as well as value, remove index_of
function Tensor:index(row, col)
  local offset = index_of(row, col, self.shape)
  --TODO: Sparse data would need manual counts
  assert(#self.data >= offset, "index out of data range: " .. offset .. " (data len: " .. #self.data .. ")")
  return self.data[offset]
end

---@param t Tensor
---@return Tensor
function Tensor:matmul(t)
  assert(self.shape[2] == t.shape[1],
    self:_sshape() .. " cannot be matmul'd with " .. t:_sshape())

  local newT = {}
  for sRow = 1, self.shape[1] do
    for tCol = 1, t.shape[2] do
      local dot = 0
      for tRow = 1, t.shape[1] do
        local v = self:index(sRow, tRow)
        local v2 = t:index(tRow, tCol)
        dot = dot + v * v2
      end
      table.insert(newT, dot)
    end
  end

  local ret = Tensor.new({ self.shape[1], t.shape[2] }, newT)
  ret.parents = { self, t }
  ret._backward = function() end
  return ret
end

---@return Tensor
function Tensor:transpose()
  local data = {}
  local shape = { self.shape[2], self.shape[1] }
  for r = 1, self.shape[1] do
    for c = 1, self.shape[2] do
      local tOffset = index_of(c, r, shape)
      data[tOffset] = self:index(r, c)
    end
  end

  local ret = Tensor.new(shape, data)
  ret.parents = { self }
  ret._backward = function() end
  return ret
end

---Element-wise multiplication
---@param t Tensor
---@return Tensor
function Tensor:mul(t)
  assert(self:eq_shape(t))
  local data = {}
  for i = 1, #self.data do
    data[i] = self.data[i] * t.data[i]
  end
  local ret = Tensor.new(self.shape, data)
  ret.parents = { self, t }
  ret._backward = function() end
  return ret
end

-- Every back function is just the chain rule for that operation, accumulating on each parent. So in the operation of c = a + b, the question is how much does each parent - relative to any other parent - influence how c changes? Not actual delta -- the gradient in c already describes how 'sensitive' it is to changes. The back-prop question is "how much of that sensitive can we attribute to each parent?". For addition, both terms equally contribute, so we add c's gradient to both a & b's. It doesn't _matter_ how large each is -- they have _equal opportunity_ to influence.
-- So for:
--  Addition both terms contribute equally, we add back c.grad
--  Subtraction, a adds positively, but b is subtracted -- sub c.grad there
--  Multiplication, it'd added still but we *weigh* by the opposite term's data; because a is added b times, and b is added a times (in a * b)
--
--  Sum is the same thing as Add
--  Mean is like Add, but divided by total elements (as their influence)
--  Transpose doesn't scale, it just moves
--
--  MatMul is the dot product of rows and columns yielding an output "coordinate", so when we apply c.grad backwards the addition part says all terms in that dot product are equally influential, but then for each pair-wise index we have to apply the multiplication rule. since the same index gets involved in multiple dot products, it mostly means we're going to update grads multiple times by "chance"
--
-- Chain rule: Take upstream gradient and multiply by local derivative, acc to each parent gradient


---@param t Tensor
---@return Tensor
function Tensor:add(t)
  assert(self:eq_shape(t))
  local data = {}
  for i = 1, #self.data do
    data[i] = self.data[i] + t.data[i]
  end
  local ret = Tensor.new(self.shape, data)
  ret.parents = { self, t }
  ret._backward = function() end
  return ret
end

---@param t Tensor
---@return Tensor
function Tensor:sub(t)
  assert(self:eq_shape(t))
  local data = {}
  for i = 1, #self.data do
    data[i] = self.data[i] - t.data[i]
  end
  local ret = Tensor.new(self.shape, data)
  ret.parents = { self, t }
  ret._backward = function() end
  return ret
end

---@param t Tensor (scalar) to scale tensor by
---@return Tensor
function Tensor:scale(t)
  assert(#t.data == 1 and #t.shape == 0)
  local data = {}
  for i = 1, #self.data do
    data[i] = self.data[i] * t.data[1]
  end
  local ret = Tensor.new(self.shape, data)
  ret.parents = { self, t }
  ret._backward = function() end
  return ret
end

---@return Tensor (scalar)
function Tensor:mean()
  local value = 0
  for i = 1, #self.data do
    value = value + self.data[i]
  end
  local ret = Tensor.new({}, { value / #self.data })
  ret.parents = { self }
  ret._backward = function() end
  return ret
end

---@return Tensor (scalar)
function Tensor:sum()
  local value = 0
  for i = 1, #self.data do
    value = value + self.data[i]
  end
  local ret = Tensor.new({}, { value })
  ret.parents = { self }
  ret._backward = function() end
  return ret
end

---@return string
function Tensor:__tostring()
  return self:_sshape() .. " " .. "[" .. table.concat(self.data, ',') .. "]"
end

---@param t any
---@return boolean
function Tensor:__eq(t)
  if getmetatable(t) ~= Tensor then
    return false
  end
  if not self:eq_shape(t) then
    return false
  end
  if #self.data ~= #t.data then
    return false
  end
  for i = 1, #self.data do
    if self.data[i] ~= t.data[i] then
      return false
    end
  end
  return true
end

function Tensor:eq_shape(t)
  if #self.shape ~= #t.shape then
    return false
  end
  for i = 1, #self.shape do
    if self.shape[i] ~= t.shape[i] then
      return false
    end
  end
  return true
end

---@return string
function Tensor:_sshape()
  return "(" .. table.concat(self.shape, ',') .. ")"
end

return Tensor
