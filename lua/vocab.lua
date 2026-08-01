local tensor = require("lua.tensor")

---@class Vocab
---@field v string[] The possible token values
---@field to_id table token-to-id map

local Vocab = {}
Vocab.__index = Vocab

---@constructor
function Vocab:new(v)
  local to_id = {}
  for index, value in ipairs(v) do
    to_id[value] = index
  end
  return setmetatable({ v = v, to_id = to_id }, Vocab)
end

---@return number Size of this vocabulary
function Vocab:size()
  return #self.v
end

---@param token string
---@return number ID of token for one-hot
function Vocab:encode(token)
  local ret = self.to_id[token]
  if ret == nil then
    error("Token not in vocab: " .. token)
  end
  return ret
end

---@param tokens string[]
---@return Tensor 1D tensor with no gradient
function Vocab:encode_many(tokens)
  local data = {}
  for i, token in ipairs(tokens) do
    data[i] = self:encode(token)
  end
  return tensor.new({ #tokens }, data, { require_grad = false })
end

---@param id number
---@return string
function Vocab:decode(id)
  return self.v[id]
end

---@param ids Tensor integer-ID tensor
---@return string[]
function Vocab:decode_many(ids)
  local tokens = {}
  for i, id in ipairs(ids.data) do
    tokens[i] = self:decode(id)
  end
  return tokens
end

---@param token string
---@return Tensor 1-D one-hot encoding of the given token
function Vocab:one_hot(token)
  local id = self:encode(token)
  local data = {}
  for i = 1, self:size() do
    if i == id then
      data[i] = 1
    else
      data[i] = 0
    end
  end
  return tensor.new({ self:size() }, data, { require_grad = false })
end

---@param tokens string[]
---@return Tensor A matrix shaped {token_count, vocabulary_size}
function Vocab:one_hot_many(tokens)
  local vocab_size = self:size()
  local data = {}

  -- Initialize the flat matrix buffer to zero.
  for i = 1, #tokens * vocab_size do
    data[i] = 0
  end

  -- Set the corresponding vocabulary column in each row.
  for row, token in ipairs(tokens) do
    local id = self:encode(token)
    local offset = (row - 1) * vocab_size + id
    data[offset] = 1
  end

  return tensor.new(
    { #tokens, vocab_size },
    data,
    { require_grad = false }
  )
end

return Vocab
