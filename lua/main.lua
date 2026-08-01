local tensor = require("lua.tensor")
local m = require("lua.model")
local o = require('lua.optimizer')
local e = require('lua.embed')

-- Relu Example: learning xor operation
local function learn_xor()
  local x = tensor.new({ 4, 2 }, { 0, 0, 0, 1, 1, 0, 1, 1, })
  local y = tensor.new({ 4, 1 }, { 0, 1, 1, 0, })
  local xor = m.Sequential(
    m.Linear(2, 4),
    m.ReLU(),
    m.Linear(4, 1)
  )
  local sgd = o.Optimizer.new(xor:parameters(), 0.05)
  for i = 1, 5000 do
    sgd:zero()
    local prediction = xor:forward(x)
    local loss = o.mean_squared(prediction, y)
    loss:backward()
    sgd:step()
    if i % 100 == 0 then
      print(i, loss.data[1])
    end
  end
end

--- Next Token Prediction: Learning to predict the next token
local function learn_bigram()
  local vocab = e.Vocab:new({ "red", "green", "blue" })
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
  local sgd = o.Optimizer.new(net:parameters(), 0.1)
  for step = 1, 5000 do
    sgd:zero()
    local pred = net:forward(inputs)
    local loss = pred:cross_entropy(targets, -1)
    loss:backward()
    sgd:step()
    if step % 100 == 0 then
      print(step, loss.data[1])
    end
  end

  local tokens = vocab:decode_many(net
    :forward(vocab:one_hot_many({ "red" }))
    -- Technically softmax not needed since this is bigram and
    -- not sampling the next token
    -- :softmax(-1)
    :argmax(-1)
  )
  print("'red' predicts: " .. tokens[1])
end

learn_bigram()
