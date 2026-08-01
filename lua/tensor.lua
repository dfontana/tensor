---@alias Shape number[]
---@alias Data number[]

---@class Tensor
---@field shape Shape Shape of the data
---@field data Data flat array of shape
---@field parents Tensor[] Parents that created this tensor, if any
---@field gradient Tensor?, only nil if this is a gradient tensor
---@field _backward fun():nil backwards prop operation
local Tensor = {}
Tensor.__index = Tensor

---@param shape Shape of tensor
---@param options ?table Tensor constructor options
---@return Tensor
function Tensor.zeroes(shape, options)
  return Tensor.fill(shape, 0, options)
end

---@param shape Shape of tensor
---@param v number value to fill tensor
---@param options ?table Tensor constructor options
---@return Tensor
function Tensor.fill(shape, v, options)
  assert(type(shape) == "table")
  assert(type(v) == "number")
  local data = {}
  if #shape == 0 then
    data[1] = v
  else
    for i = 1, Tensor.numel(shape) do
      data[i] = v
    end
  end
  options = options or {}
  local self = {
    shape = shape,
    data = data,
    parents = {},
    gradient = options.require_grad and Tensor.zeroes(shape, { require_grad = false }) or nil,
    _backward = function() end
  }
  return setmetatable(self, Tensor)
end

---@param v number Scalar value
---@param options ?table Tensor constructor options
---@return Tensor
function Tensor.scalar(v, options)
  assert(type(v) == "number")
  return Tensor.new({}, { v }, options)
end

---Uniformly sample each value of a new tensor
---@param shape Shape
---@param low number Lower bound (incl)
---@param high number Upper bound (excl)
---@param options ?table Tensor contructor options
---@return Tensor
function Tensor.uniform(shape, low, high, options)
  local data = {}
  for i = 1, Tensor.numel(shape) do
    data[i] = math.random() * (high - low) + low
  end
  return Tensor.new(shape, data, options)
end

---Sample a normal distribution for each value of a new tensor
---@param shape Shape
---@param mean number
---@param stddev number
---@param options ?table Tensor contructor options
---@return Tensor
function Tensor.normal(shape, mean, stddev, options)
  local data = {}
  for i = 1, Tensor.numel(shape) do
    local u1 = math.random()
    local u2 = math.random()
    local z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    data[i] = mean + stddev * z
  end
  return Tensor.new(shape, data, options)
end

---@constructor
---@param shape Shape of tensor
---@param data Data flat array of data for tensor
---@param options ?table Tensor constructor options
function Tensor.new(shape, data, options)
  assert(type(shape) == "table")
  assert(type(data) == "table")
  options = options or { require_grad = true }
  local self = {
    shape = shape,
    data = data,
    parents = {},
    gradient = options.require_grad and Tensor.zeroes(shape, { require_grad = false }) or nil,
    _backward = function() end
  }
  return setmetatable(self, Tensor)
end

---Zero the gradients of this tensor
function Tensor:zero_grad()
  self.gradient = Tensor.zeroes(self.shape, { require_grad = false })
end

---Number of elements in tensor
---@param shape Shape
---@return integer
function Tensor.numel(shape)
  local n = 1
  for i = 1, #shape do
    n = n * shape[i]
  end
  return n
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
function Tensor:_index(row, col)
  local offset = index_of(row, col, self.shape)
  assert(#self.data >= offset, "index out of data range: " .. offset .. " (data len: " .. #self.data .. ")")
  return self.data[offset]
end

function Tensor:_is_scalar()
  return #self.shape == 0
end

---Perform a backwards pass starting at this scalar tensor. This updates
---all gradients of the graph in-place, so nothing is returned
---@return nil
function Tensor:backward()
  assert(self:_is_scalar(), "Can only run backwards on a scalar")
  self.gradient.data[1] = 1
  local order = {}
  local seen = { [self] = true }
  local frontier = { { node = self, expanded = false } }
  while #frontier ~= 0 do
    local t = table.remove(frontier)
    local next, expanded = t.node, t.expanded
    if not expanded then
      table.insert(frontier, { node = next, expanded = true })
      for _, p in ipairs(next.parents) do
        if not seen[p] then
          seen[p] = true
          table.insert(frontier, { node = p, expanded = false })
        end
      end
    else
      table.insert(order, next)
    end
  end
  for i = #order, 1, -1 do
    order[i]:_backward()
  end
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
        local v = self:_index(sRow, tRow)
        local v2 = t:_index(tRow, tCol)
        dot = dot + v * v2
      end
      table.insert(newT, dot)
    end
  end

  local ret = Tensor.new({ self.shape[1], t.shape[2] }, newT)
  ret.parents = { self, t }
  ret._backward = function()
    t:_accumulate_grad(self:transpose():matmul(ret.gradient))
    self:_accumulate_grad(ret.gradient:matmul(t:transpose()))
  end
  return ret
end

---@return Tensor
function Tensor:transpose()
  local data = {}
  local shape = {}
  if self:_is_scalar() then
    data[1] = self.data[1]
  else
    shape = { self.shape[2], self.shape[1] }
    for r = 1, self.shape[1] do
      for c = 1, self.shape[2] do
        local tOffset = index_of(c, r, shape)
        data[tOffset] = self:_index(r, c)
      end
    end
  end
  local ret = Tensor.new(shape, data)
  ret.parents = { self }
  ret._backward = function()
    self:_accumulate_grad(ret.gradient:transpose())
  end
  return ret
end

---@param t Tensor
---@return Tensor
function Tensor:mul(t)
  local ret = self:_binary_elementwise(t, function(a, b)
    return a * b
  end)
  ret.parents = { self, t }
  ret._backward = function()
    self:_accumulate_grad(ret.gradient:mul(t))
    t:_accumulate_grad(ret.gradient:mul(self))
  end
  return ret
end

---@param t Tensor
---@return Tensor
function Tensor:add(t)
  local ret = self:_binary_elementwise(t, function(a, b)
    return a + b
  end)
  ret.parents = { self, t }
  ret._backward = function()
    self:_accumulate_grad(ret.gradient)
    t:_accumulate_grad(ret.gradient)
  end
  return ret
end

---@param t Tensor
---@return Tensor
function Tensor:sub(t)
  local ret = self:_binary_elementwise(t, function(a, b)
    return a - b
  end)
  ret.parents = { self, t }
  ret._backward = function()
    self:_accumulate_grad(ret.gradient)
    t:_accumulate_grad(ret.gradient:scale(Tensor.scalar(-1)))
  end
  return ret
end

---It's an element-wise max(0, x)
---@return Tensor
function Tensor:relu()
  local ret = self:_unary_elementwise(function(a)
    return math.max(0, a)
  end)
  ret.parents = { self }
  ret._backward = function()
    local data = {}
    for i = 1, Tensor.numel(self.shape) do
      if self.data[i] > 0 then
        data[i] = ret.gradient.data[i]
      else
        data[i] = 0
      end
    end
    self:_accumulate_grad(Tensor.new(self.shape, data))
  end
  return ret
end

---Compute stable softmax probabilities and log-probabilities without
---creating tensors or graph nodes.
---@param tensor Tensor
---@param axis number
---@param temp number
---@return Data probabilities
---@return Data log_probabilities
local function softmax_values(tensor, axis, temp)
  local probabilities = {}
  local log_probabilities = {}

  tensor:_each_axis_slice(axis, function(start, width, stride)
    local max_value = -math.huge
    for i = 0, width - 1 do
      max_value = math.max(max_value, tensor.data[start + i * stride])
    end

    local total = 0
    for i = 0, width - 1 do
      local offset = start + i * stride
      probabilities[offset] = math.exp((tensor.data[offset] - max_value) / temp)
      total = total + probabilities[offset]
    end

    local log_total = math.log(total)
    for i = 0, width - 1 do
      local offset = start + i * stride
      probabilities[offset] = probabilities[offset] / total
      log_probabilities[offset] = (tensor.data[offset] - max_value) / temp - log_total
    end
  end)

  return probabilities, log_probabilities
end

function Tensor:softmax(axis, t)
  axis = self:_norm_axis(axis)

  local temp = t or 1.0
  assert(temp > 0, "temperature must be positive")

  local data, _ = softmax_values(self, axis, temp)
  local ret = Tensor.new(self.shape, data)
  ret.parents = { self }
  ret._backward = function()
    local grad_data = {}

    self:_each_axis_slice(axis, function(start, width, stride)
      local dot = 0
      for i = 0, width - 1 do
        local offset = start + i * stride
        dot = dot + ret.gradient.data[offset] * ret.data[offset]
      end

      for i = 0, width - 1 do
        local offset = start + i * stride
        grad_data[offset] = (ret.data[offset] / temp)
            * (ret.gradient.data[offset] - dot)
      end
    end)

    self:_accumulate_grad(Tensor.new(self.shape, grad_data, { require_grad = false }))
  end

  return ret
end

function Tensor:cross_entropy(actual, axis)
  assert(self:_eq_shape(actual), "shape mismatch")
  axis = self:_norm_axis(axis)

  local probabilities, log_probabilities = softmax_values(self, axis, 1.0)

  local total_loss = 0
  for i = 1, #self.data do
    total_loss = total_loss - actual.data[i] * log_probabilities[i]
  end

  local slice_count = #self.data / self.shape[axis]
  local ret = Tensor.scalar(total_loss / slice_count)
  ret.parents = { self }
  ret._backward = function()
    local grad_data = {}
    local upstream = ret.gradient.data[1]
    for i = 1, #self.data do
      grad_data[i] = upstream
          * (probabilities[i] - actual.data[i])
          / slice_count
    end
    self:_accumulate_grad(Tensor.new(self.shape, grad_data, { require_grad = false }))
  end

  return ret
end

---Return the index of the largest value along an axis.
---The output shape is the input shape with that axis removed.
---@param axis number 1-based axis; negative counts from the end
---@return Tensor integer-ID tensor; does not participate in autograd
function Tensor:argmax(axis)
  axis = self:_norm_axis(axis)

  local output_shape = {}
  for dim = 1, #self.shape do
    if dim ~= axis then
      table.insert(output_shape, self.shape[dim])
    end
  end

  local data = {}
  self:_each_axis_slice(axis, function(start, width, stride)
    local best_index = 1
    local best_value = self.data[start]
    for i = 1, width - 1 do
      local value = self.data[start + i * stride]
      -- Ties choose the first occurrence.
      if value > best_value then
        best_value = value
        best_index = i + 1
      end
    end
    table.insert(data, best_index)
  end)

  return Tensor.new(output_shape, data, { require_grad = false })
end

---@param t Tensor (scalar)
---@return Tensor
function Tensor:pow(t)
  assert(t:_is_scalar())
  local data = {}
  for i = 1, #self.data do
    data[i] = self.data[i] ^ t.data[1]
  end
  local ret = Tensor.new(self.shape, data)
  ret.parents = { self, t }
  ret._backward = function()
    -- upstream * t * x^(t−1)
    local one = Tensor.scalar(1, { require_grad = false })
    self:_accumulate_grad(
      ret.gradient
      :mul(t)
      :mul(self:pow(t:sub(one)))
    )
    -- d(x^t)/dt = x^t * log(x)
    local t_grad = {}
    for i = 1, #self.data do
      local x = self.data[i]
      local upstream = ret.gradient.data[i]
      -- log(x) is undefined for x <= 0; the exponent gradient is meaningless
      -- there, so contribute 0 rather than NaN-poisoning the graph.
      t_grad[i] = x > 0 and (upstream * ret.data[i] * math.log(x)) or 0
    end
    t:_accumulate_grad(Tensor.new(self.shape, t_grad))
  end
  return ret
end

---@param t Tensor (scalar)
---@return Tensor
function Tensor:scale(t)
  assert(t:_is_scalar())
  local data = {}
  for i = 1, #self.data do
    data[i] = self.data[i] * t.data[1]
  end
  local ret = Tensor.new(self.shape, data)
  ret.parents = { self, t }
  ret._backward = function()
    self:_accumulate_grad(ret.gradient:scale(t))
    t:_accumulate_grad(self:mul(ret.gradient))
  end
  return ret
end

---@return Tensor (scalar)
function Tensor:mean()
  local value = 0
  for i = 1, #self.data do
    value = value + self.data[i]
  end
  local ret = Tensor.scalar(value / #self.data)
  ret.parents = { self }
  ret._backward = function()
    self:_accumulate_grad(ret.gradient:scale(Tensor.scalar(1 / #self.data)))
  end
  return ret
end

---@return Tensor (scalar)
function Tensor:sum()
  local value = 0
  for i = 1, #self.data do
    value = value + self.data[i]
  end
  local ret = Tensor.scalar(value)
  ret.parents = { self }
  ret._backward = function()
    self:_accumulate_grad(ret.gradient)
  end
  return ret
end

---Call fn once for each slice along axis.
---start is the first 1-based flat offset; stride moves within the slice.
---@param axis number
---@param fn fun(start: number, width: number, stride: number)
function Tensor:_each_axis_slice(axis, fn)
  local width = self.shape[axis]
  local outer = 1
  local inner = 1
  for dim = 1, axis - 1 do
    outer = outer * self.shape[dim]
  end
  for dim = axis + 1, #self.shape do
    inner = inner * self.shape[dim]
  end
  for outer_index = 0, outer - 1 do
    for inner_index = 0, inner - 1 do
      local start = outer_index * width * inner + inner_index + 1
      fn(start, width, inner)
    end
  end
end

---Compute the broadcast output shape against the given Tensors. Only two forms are
---supported: scalar {} against anything, and a vector {N} against a tensor whose
---last dimension is N. Anything else (e.g. {4,3}+{4}, {4,3}+{4,1}) is an error.
---@param at Tensor
---@param bt Tensor
---@return Shape
local function broadcast_output_shape(at, bt)
  local a = at.shape
  local b = bt.shape
  if #a == 0 then
    return b
  end
  if #b == 0 then
    return a
  end
  if at:_eq_shape(bt) then
    return a
  end
  if #a == 1 then
    assert(b[#b] == a[1],
      "cannot broadcast (" .. table.concat(a, ',') .. ") with (" .. table.concat(b, ',') .. ")")
    return b
  end
  if #b == 1 then
    assert(a[#a] == b[1],
      "cannot broadcast (" .. table.concat(a, ',') .. ") with (" .. table.concat(b, ',') .. ")")
    return a
  end
  error("cannot broadcast (" .. table.concat(a, ',') .. ") with (" .. table.concat(b, ',') .. ")")
end

---Read an operand's value for the i-th element of a broadcast result of `size`
---elements. Scalars repeat their single value; a vector {N} repeats along the
---(contiguous) last dimension; a full-shaped operand indexes directly.
---@param operand Tensor
---@param i number flat (1-based) index into the broadcast result
---@param size number number of elements in the broadcast result
---@return number
local function broadcast_get(operand, i, size)
  if #operand.shape == 0 then
    return operand.data[1]
  end
  if #operand.data == size then
    return operand.data[i]
  end
  local n = operand.shape[#operand.shape]
  return operand.data[((i - 1) % n) + 1]
end

-- Apply a binary elementwise operation with broadcasting. Broadcasting is
-- limited to two forms: a scalar {} against anything, and a vector {N} against
-- a tensor whose last dimension is N (the vector is repeated along the leading
-- dimensions). The result takes the fuller shape.
---@param t Tensor
---@param op fun(number, number): number
---@return Tensor
function Tensor:_binary_elementwise(t, op)
  local shape = broadcast_output_shape(self, t)
  local size = Tensor.numel(shape)
  local data = {}
  for i = 1, size do
    data[i] = op(broadcast_get(self, i, size), broadcast_get(t, i, size))
  end
  return Tensor.new(shape, data)
end

-- Apply a unary elementwise operation
---@param op fun(number): number
---@return Tensor
function Tensor:_unary_elementwise(op)
  local data = {}
  for i = 1, Tensor.numel(self.shape) do
    data[i] = op(self.data[i])
  end
  return Tensor.new(self.shape, data)
end

--- Accumulate an incoming gradient into self, undoing any broadcasting that
--- happened in the forward pass. A gradient arrives shaped like the op output;
--- to land on self it must be summed back down to self's shape (summing every
--- dimension self was broadcast across).
---@param grad Tensor
---@return nil
function Tensor:_accumulate_grad(grad)
  local g = self.gradient
  if g == nil then
    return
  end
  -- Same shape: straight elementwise accumulation.
  if self:_eq_shape(grad) then
    for i = 1, #g.data do
      g.data[i] = g.data[i] + grad.data[i]
    end
    return
  end
  -- Incoming scalar gradient broadcast up across self.
  if grad:_is_scalar() then
    for i = 1, #g.data do
      g.data[i] = g.data[i] + grad.data[1]
    end
    return
  end
  -- Self is scalar: reduce the full incoming gradient down to it.
  if self:_is_scalar() then
    local total = 0
    for i = 1, #grad.data do
      total = total + grad.data[i]
    end
    g.data[1] = g.data[1] + total
    return
  end
  -- Self is a vector {N} broadcast along a tensor's last dimension: sum every
  -- leading (broadcast) dimension back onto the N vector elements.
  local n = self.shape[#self.shape]
  for i = 1, #grad.data do
    local j = ((i - 1) % n) + 1
    g.data[j] = g.data[j] + grad.data[i]
  end
end

---@param axis number
---@return number normalized positive axis
function Tensor:_norm_axis(axis)
  if axis < 0 then
    axis = #self.shape + axis + 1
  end
  assert(axis >= 1 and axis <= #self.shape, "axis out of range")
  return axis
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
  if not self:_eq_shape(t) then
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

function Tensor:_eq_shape(t)
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
