local impl = require("test.impl")
local tensor = impl.tensor
local optimizer = impl.optimizer
local assert = require("luassert")

-- Free-function API. The kebab / `!` names are reached by bracket access from Lua.
local zero = optimizer["zero!"]
local step = optimizer["step!"]
local mean_squared = optimizer["mean-squared"]
local backward = tensor["backward!"]

describe("optimizer.zero!", function()
  it("resets the gradients of every parameter in the (string-keyed) param map", function()
    local params = {
      m = tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 }),
      b = tensor.scalar(0.2),
    }
    params.m.gradient = tensor.new(params.m.shape, { 9, 9, 9, 9, 9, 9 })
    params.b.gradient = tensor.scalar(9)
    zero(params)
    for i = 1, #params.m.gradient.data do
      assert.equal(0, params.m.gradient.data[i])
    end
    assert.equal(0, params.b.gradient.data[1])
  end)
end)

describe("optimizer.step!", function()
  it("moves each parameter one learning-rate step down its gradient", function()
    local params = {
      m = tensor.new({ 1, 3 }, { 1, 2, 3 }),
      b = tensor.scalar(10),
    }
    params.m.gradient = tensor.new(params.m.shape, { 1, 1, 1 })
    params.b.gradient = tensor.scalar(2)
    step(params, 0.1)
    -- p := p - lr * grad
    assert.same({ 0.9, 1.9, 2.9 }, params.m.data)
    assert.near(9.8, params.b.data[1], 1e-9)
  end)
end)

describe("optimizer.mean-squared", function()
  it("reduces to a scalar tensor holding the mean of the squared errors", function()
    local pred = tensor.new({ 1, 3 }, { 1, 2, 3 })
    local actual = tensor.new({ 1, 3 }, { 0, 0, 0 })
    local loss = mean_squared(pred, actual)
    assert.same({}, loss.shape)
    -- mean(1^2, 2^2, 3^2) = 14 / 3
    assert.near(14 / 3, loss.data[1], 1e-9)
  end)

  it("is zero exactly when the prediction equals the target", function()
    local pred = tensor.new({ 2, 2 }, { 1, 2, 3, 4 })
    local actual = tensor.new({ 2, 2 }, { 1, 2, 3, 4 })
    assert.equal(0, mean_squared(pred, actual).data[1])
  end)

  it("is symmetric in prediction and target (the error is squared)", function()
    local a = tensor.new({ 1, 3 }, { 1, 2, 3 })
    local b = tensor.new({ 1, 3 }, { 4, 0, 5 })
    assert.near(
      mean_squared(a, b).data[1],
      mean_squared(b, a).data[1],
      1e-9)
  end)

  it("backpropagates the gradient 2*(pred - actual)/n onto the prediction", function()
    -- Composed from sub/pow/mean, each already gradchecked in tensor_spec;
    -- this pins the composition's end-to-end gradient at the leaf.
    local pred = tensor.new({ 1, 3 }, { 1, 2, 3 })
    local actual = tensor.new({ 1, 3 }, { 0, 0, 0 })
    local loss = mean_squared(pred, actual)
    backward(loss)
    -- 2 * {1, 2, 3} / 3
    assert.near(2 / 3, pred.gradient.data[1], 1e-9)
    assert.near(4 / 3, pred.gradient.data[2], 1e-9)
    assert.near(6 / 3, pred.gradient.data[3], 1e-9)
  end)
end)

describe("Optimizer end-to-end (linear regression, the milestone goal)", function()
  -- Fit y = m*x + b to the single point (x=3, y=1) starting from m=b=0.
  -- With gradients reset each step, the loss must fall monotonically.
  local function forward(params)
    local pred = tensor.add(tensor.mul(params.m, tensor.scalar(3)), params.b)
    return tensor.mean(tensor.pow(tensor.sub(pred, tensor.scalar(1)), tensor.scalar(2)))
  end

  it("decreases the loss on every step and drives it toward zero", function()
    local params = {
      m = tensor.new({}, { 0 }),
      b = tensor.new({}, { 0 }),
    }
    local previous = math.huge
    local last
    for _ = 1, 8 do
      zero(params)
      local loss = forward(params)
      last = loss.data[1]
      assert.is_true(last < previous)
      previous = last
      backward(loss)
      step(params, 0.02)
    end
    assert.is_true(last < 1e-2)
  end)

  it("would stall without zero!(): stale gradients accumulate across steps", function()
    -- Same loop but never zeroing. Gradients from earlier steps pile onto
    -- later ones, so the trajectory is no longer plain gradient descent and
    -- the clean monotonic descent above is lost. This pins down *why* zero!()
    -- is part of the step/zero/forward/backward cycle.
    local params = {
      m = tensor.new({}, { 0 }),
      b = tensor.new({}, { 0 }),
    }
    local monotonic = true
    local previous = math.huge
    for _ = 1, 8 do
      local loss = forward(params)
      if loss.data[1] >= previous then monotonic = false end
      previous = loss.data[1]
      backward(loss)
      step(params, 0.02)
    end
    assert.is_false(monotonic)
  end)
end)
