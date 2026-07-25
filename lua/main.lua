local tensor = require("lua.tensor")
local m = require("lua.model")
local optimizer = require('lua.optimizer')


local x = tensor.new(
  { 4, 2 },
  {
    0, 0,
    0, 1,
    1, 0,
    1, 1,
  }
)
local y = tensor.new(
  { 4, 1 },
  {
    0,
    1,
    1,
    0,
  }
)
local model = m.Sequential(
  m.Linear(2, 4),
  m.ReLU(),
  m.Linear(4, 1)
)
local sgd = optimizer.new(model:parameters(), 0.05)
for i = 1, 5000 do
  sgd:zero()
  local prediction = model:forward(x)
  local loss = prediction
      :sub(y)
      :pow(tensor.scalar(2))
      :mean()
  loss:backward()
  sgd:step()
  if i % 100 == 0 then
    print(i, loss.data[1])
  end
end
