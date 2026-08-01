local tensor = require("lua.tensor")
local m = require("lua.model")
local o = require('lua.optimizer')
local Vocab = require('lua.vocab')

local function train(model, input, loss_fn)
  local sgd = o.Optimizer.new(model:parameters(), 0.05)
  for i = 1, 5000 do
    sgd:zero()
    local prediction = model:forward(input)
    local loss = loss_fn(prediction)
    loss:backward()
    sgd:step()
    if i % 100 == 0 then
      print(i, loss.data[1])
    end
  end
end

-- Relu Example: learning xor operation
local function learn_xor()
  local x = tensor.new({ 4, 2 }, { 0, 0, 0, 1, 1, 0, 1, 1, })
  local y = tensor.new({ 4, 1 }, { 0, 1, 1, 0, })
  local xor = m.Sequential(
    m.Linear(2, 4),
    m.ReLU(),
    m.Linear(4, 1)
  )
  train(xor, x, function(pred) return o.mean_squared(pred, y) end)
end

-- learn_xor()

--- Next Token Prediction: Learning to predict the next token
local function learn_bigram()
  local vocab = Vocab:new({ "red", "green", "blue" })
  local inputs = vocab:one_hot_many({
    "red",
    "green",
    "blue",
  })
  local targets = vocab:one_hot_many({
    "green",
    "blue",
    "red",
  })
  local net = m.Linear(vocab:size(), vocab:size())
  train(net, inputs, function(pred) return pred:cross_entropy(targets, -1) end)
  local tokens = vocab:decode_many(net
    :forward(vocab:one_hot_many({ "red" }))
    -- Technically softmax not needed since this is bigram and
    -- not sampling the next token
    -- :softmax(-1)
    :argmax(-1)
  )
  print("'red' predicts: " .. tokens[1])
end

-- learn_bigram()

--- Next Token Prediction: Embedding
local function learn_embeds()
  local vocab = Vocab:new({ "red", "green", "blue" })
  local inputIds = vocab:encode_many({
    "red",
    "green",
    "blue",
  })
  local inf = vocab:encode_many({ "red" })
  local targets = vocab:one_hot_many({
    "green",
    "blue",
    "red",
  })

  local net = m.Sequential(
    m.Embedding(vocab, 32),
    m.Linear(32, vocab:size())
  )
  train(net, inputIds, function(pred) return pred:cross_entropy(targets, -1) end)
  local tokens = vocab:decode_many(net
    :forward(inf)
    -- Technically softmax not needed since this is bigram and
    -- not sampling the next token
    -- :softmax(-1)
    :argmax(-1)
  )
  print("'red' predicts: " .. tokens[1])
end

learn_embeds()
