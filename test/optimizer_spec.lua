local tensor = require("lua.tensor")
local optimizer = require("lua.optimizer")
local assert = require("luassert")

describe("Optimizer:zero", function()
  it("resets the gradients of every parameter in the (string-keyed) param map", function()
    local params = {
      m = tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 }),
      b = tensor.scalar(0.2),
    }
    params.m.gradient = tensor.new(params.m.shape, { 9, 9, 9, 9, 9, 9 })
    params.b.gradient = tensor.scalar(9)
    optimizer.new(params, 0.01):zero()
    for i = 1, #params.m.gradient.data do
      assert.equal(0, params.m.gradient.data[i])
    end
    assert.equal(0, params.b.gradient.data[1])
  end)
end)

describe("Optimizer:step", function()
  it("moves each parameter one learning-rate step down its gradient", function()
    local params = {
      m = tensor.new({ 1, 3 }, { 1, 2, 3 }),
      b = tensor.scalar(10),
    }
    params.m.gradient = tensor.new(params.m.shape, { 1, 1, 1 })
    params.b.gradient = tensor.scalar(2)
    optimizer.new(params, 0.1):step()
    -- p := p - lr * grad
    assert.same({ 0.9, 1.9, 2.9 }, params.m.data)
    assert.near(9.8, params.b.data[1], 1e-9)
  end)
end)

describe("Optimizer end-to-end (linear regression, the milestone goal)", function()
  -- Fit y = m*x + b to the single point (x=3, y=1) starting from m=b=0.
  -- With gradients reset each step, the loss must fall monotonically.
  local function forward(params)
    local pred = params.m:mul(tensor.scalar(3)):add(params.b)
    return pred:sub(tensor.scalar(1)):pow(tensor.scalar(2)):mean()
  end

  it("decreases the loss on every step and drives it toward zero", function()
    local params = {
      m = tensor.new({}, { 0 }),
      b = tensor.new({}, { 0 }),
    }
    local sgd = optimizer.new(params, 0.02)
    local previous = math.huge
    local last
    for _ = 1, 8 do
      sgd:zero()
      local loss = forward(params)
      last = loss.data[1]
      assert.is_true(last < previous)
      previous = last
      loss:backwards()
      sgd:step()
    end
    assert.is_true(last < 1e-2)
  end)

  it("would stall without zero(): stale gradients accumulate across steps", function()
    -- Same loop but never zeroing. Gradients from earlier steps pile onto
    -- later ones, so the trajectory is no longer plain gradient descent and
    -- the clean monotonic descent above is lost. This pins down *why* zero()
    -- is part of the step/zero/forward/backward cycle.
    local params = {
      m = tensor.new({}, { 0 }),
      b = tensor.new({}, { 0 }),
    }
    local sgd = optimizer.new(params, 0.02)
    local monotonic = true
    local previous = math.huge
    for _ = 1, 8 do
      local loss = forward(params)
      if loss.data[1] >= previous then monotonic = false end
      previous = loss.data[1]
      loss:backwards()
      sgd:step()
    end
    assert.is_false(monotonic)
  end)
end)
