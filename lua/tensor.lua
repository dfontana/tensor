---@alias Shape number[]
---@alias Data number[]

---@class Tensor
---@field shape Shape Shape of the data
---@field data Data flat array of shape
local Tensor = {}
Tensor.__index = Tensor

---@constructor
---@param shape Shape of tensor
---@param data Data flat array of data for tensor
function Tensor.new(shape, data)
  assert(type(shape) == "table")
  assert(type(data) == "table")
  local self = {
    shape = shape,
    data = data,
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

  return Tensor.new(
    { self.shape[1], t.shape[2] },
    newT
  )
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

  return Tensor.new(shape, data)
end

---Element-wise multiplication
---@param t Tensor
---@return Tensor
function Tensor:mul(t)
  return self
end

---@param t Tensor
---@return Tensor
function Tensor:add(t)
  return self
end

---@param t Tensor
---@return Tensor
function Tensor:sub(t)
  return self
end

---@param t Tensor (scalar) to scale tensor by
---@return Tensor
function Tensor:scale(t)
  return self
end

---@return Tensor (scalar)
function Tensor:mean()
  return self
end

---@return Tensor (scalar)
function Tensor:sum()
  return self
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
  if #self.shape ~= #t.shape then
    return false
  end
  if #self.data ~= #t.data then
    return false
  end
  for i = 1, #self.shape do
    if self.shape[i] ~= t.shape[i] then
      return false
    end
  end
  for i = 1, #self.data do
    if self.data[i] ~= t.data[i] then
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
