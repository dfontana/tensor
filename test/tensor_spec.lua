local tensor = require("lua.tensor")
local assert = require("luassert")

local a = tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 })

describe("tensor", function()
  it("indexes elements", function()
    for _, case in ipairs({ { 1, 1, 1 }, { 1, 3, 3 }, { 2, 2, 5 } }) do
      assert.equal(case[3], a:_index(case[1], case[2]))
    end
  end)

  it("multiplies matrices", function()
    local b = tensor.new({ 3, 4 }, { 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 })
    assert.equal(tensor.new({ 2, 4 }, { 74, 80, 86, 92, 173, 188, 203, 218 }), a:matmul(b))
  end)

  it("transposes matrices", function()
    assert.equal(tensor.new({ 3, 2 }, { 1, 4, 2, 5, 3, 6 }), a:transpose())
  end)

  it("satisfies matrix identities", function()
    local b = tensor.new({ 3, 4 }, { 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 })
    local i = tensor.new({ 3, 3 }, { 1, 0, 0, 0, 1, 0, 0, 0, 1 })
    for _, case in ipairs({
      { a:transpose():transpose(), a },
      { a:matmul(i),               a },
      { a:matmul(b):transpose(),   b:transpose():matmul(a:transpose()) },
    }) do
      assert.equal(case[2], case[1])
    end
  end)

  it("satisfies element-wise addition and subtraction identities", function()
    local b = tensor.new({ 2, 3 }, { 6, 5, 4, 3, 2, 1 })
    local zero = tensor.new({ 2, 3 }, { 0, 0, 0, 0, 0, 0 })
    for _, case in ipairs({
      { a:add(b),        b:add(a) },
      { a:add(b):sub(b), a },
      { a:sub(a),        zero },
    }) do
      assert.equal(case[2], case[1])
    end
  end)

  it("satisfies element-wise multiplication identities", function()
    local b = tensor.new({ 2, 3 }, { 6, 5, 4, 3, 2, 1 })
    local zero = tensor.new({ 2, 3 }, { 0, 0, 0, 0, 0, 0 })
    local one = tensor.new({ 2, 3 }, { 1, 1, 1, 1, 1, 1 })
    for _, case in ipairs({
      { a:mul(b),    b:mul(a) },
      { a:mul(one),  a },
      { a:mul(zero), zero },
    }) do
      assert.equal(case[2], case[1])
    end
  end)

  it("satisfies scalar multiplication identities", function()
    local zero = tensor.new({ 2, 3 }, { 0, 0, 0, 0, 0, 0 })
    for _, case in ipairs({
      { a:scale(tensor.scalar(1)), a },
      { a:scale(tensor.scalar(0)), zero },
      { a:scale(tensor.scalar(2)), a:add(a) },
    }) do
      assert.equal(case[2], case[1])
    end
  end)

  it("raises elements to a scalar power", function()
    assert.same({ 1, 4, 9, 16, 25, 36 }, a:pow(tensor.scalar(2)).data)
  end)

  it("satisfies power identities", function()
    local ones = tensor.new({ 2, 3 }, { 1, 1, 1, 1, 1, 1 })
    for _, case in ipairs({
      { a:pow(tensor.scalar(1)), a },
      { a:pow(tensor.scalar(0)), ones },
    }) do
      assert.equal(case[2], case[1])
    end
  end)

  it("reduces tensors to scalar tensors", function()
    local scalar = tensor.scalar(21)
    assert.equal(scalar, a:sum())
    assert.equal(a:sum(), a:transpose():sum())
  end)

  it("computes means as scalar tensors", function()
    local scalar = tensor.scalar(3.5)
    assert.equal(scalar, a:mean())
    assert.equal(a:mean(), a:transpose():mean())
  end)

  it("compares tensors", function()
    for _, case in ipairs({ { a, true }, { tensor.new({ 3, 2 }, { 7, 8, 9, 10, 11, 12 }), false }, { 1, false } }) do
      assert.equal(case[2], a == case[1])
    end
  end)
end)

describe("tensor parent tracking", function()
  local a = tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 })
  local b = tensor.new({ 3, 4 }, { 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 })
  local c = tensor.new({ 2, 3 }, { 6, 5, 4, 3, 2, 1 })
  local s = tensor.scalar(2)

  -- A dedicated tensor identical in value to `a` but a distinct object, to
  -- distinguish identity tracking (what backprop needs) from value equality
  -- (what __eq checks).
  local a_lookalike = tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 })

  it("gives leaf tensors no parents", function()
    assert.equal(0, #a.parents)
    assert.equal(0, #s.parents)
  end)

  it("records exactly the inputs, by identity, as parents for unary ops", function()
    local cases = {
      { a:transpose(), a },
      { a:sum(),       a },
      { a:mean(),      a },
    }
    for _, case in ipairs(cases) do
      local result, input = case[1], case[2]
      assert.equal(1, #result.parents)
      assert.is_true(rawequal(result.parents[1], input))
    end
  end)

  it("records exactly the inputs, by identity and in order, as parents for binary ops", function()
    local cases = {
      { a:matmul(b), a, b },
      { a:add(c),    a, c },
      { a:sub(c),    a, c },
      { a:mul(c),    a, c },
      { a:scale(s),  a, s },
      { a:pow(s),    a, s },
    }
    for _, case in ipairs(cases) do
      local result, lhs, rhs = case[1], case[2], case[3]
      assert.equal(2, #result.parents)
      assert.is_true(rawequal(result.parents[1], lhs))
      assert.is_true(rawequal(result.parents[2], rhs))
    end
  end)

  it("distinguishes parent identity from value equality", function()
    local result = a:add(c)
    -- Value-equal but distinct object: must NOT be accepted as the parent.
    assert.is_true(a_lookalike == a)
    assert.is_false(rawequal(result.parents[1], a_lookalike))
    assert.is_true(rawequal(result.parents[1], a))
  end)

  it("only tracks immediate parents, not the full ancestry, for chained ops", function()
    local sum_ab = a:add(c)
    local chained = sum_ab:sub(c)

    assert.equal(2, #chained.parents)
    assert.is_true(rawequal(chained.parents[1], sum_ab))
    assert.is_true(rawequal(chained.parents[2], c))

    -- The original leaves are reachable through the parent graph, but are
    -- not direct parents of the chained result.
    assert.is_false(rawequal(chained.parents[1], a))
    assert.is_true(rawequal(chained.parents[1].parents[1], a))
    assert.is_true(rawequal(chained.parents[1].parents[2], c))
  end)

  it("gives every non-leaf tensor a nonzero-length parents table", function()
    local results = {
      a:matmul(b), a:transpose(), a:mul(c), a:add(c), a:sub(c),
      a:scale(s), a:pow(s), a:mean(), a:sum(),
    }
    for _, result in ipairs(results) do
      assert.is_true(#result.parents > 0)
    end
  end)
end)

describe("tensor operations with scalar operands", function()
  local m = tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 })
  local three = tensor.scalar(3)
  local four = tensor.scalar(4)

  it("broadcasts a scalar against a matrix, and the result matches the shape of the non-scalar side", function()
    local cases = {
      { m:add(three), { 4, 5, 6, 7, 8, 9 } },
      { three:add(m), { 4, 5, 6, 7, 8, 9 } },
      { m:mul(three), { 3, 6, 9, 12, 15, 18 } },
      { three:mul(m), { 3, 6, 9, 12, 15, 18 } },
      { m:sub(three), { -2, -1, 0, 1, 2, 3 } },
    }
    for _, case in ipairs(cases) do
      local result, expected = case[1], case[2]
      assert.same(m.shape, result.shape)
      assert.same(expected, result.data)
    end
  end)

  it("keeps scalar minus tensor and tensor minus scalar as distinct, non-commutative results", function()
    -- three - m (scalar on the left) must differ from m - three (scalar on
    -- the right): subtraction is not commutative, broadcasting shouldn't
    -- change that.
    assert.same({ 2, 1, 0, -1, -2, -3 }, three:sub(m).data)
    assert.same({ -2, -1, 0, 1, 2, 3 }, m:sub(three).data)
    -- The two are exact negations of each other.
    for i = 1, #m.data do
      assert.equal(-1 * three:sub(m).data[i], m:sub(three).data[i])
    end
  end)

  it("keeps add and mul commutative under broadcasting (identical to the non-scalar case)", function()
    assert.equal(m:add(three), three:add(m))
    assert.equal(m:mul(three), three:mul(m))
  end)

  it("combines two scalars using ordinary scalar arithmetic", function()
    assert.equal(tensor.scalar(7), three:add(four))
    assert.equal(tensor.scalar(12), three:mul(four))
    -- four - three = 1, not three - four = -1: direction must be preserved
    -- even when both operands happen to be scalars.
    assert.equal(tensor.scalar(1), four:sub(three))
    assert.equal(tensor.scalar(-1), three:sub(four))
  end)

  it("treats scalar add/sub/mul by a scalar identity the same way non-scalar identities work", function()
    local zero = tensor.scalar(0)
    local one = tensor.scalar(1)
    for _, case in ipairs({
      { m:add(zero), m },
      { m:mul(one),  m },
      { m:sub(zero), m },
      { m:mul(zero), tensor.new({ 2, 3 }, { 0, 0, 0, 0, 0, 0 }) },
    }) do
      assert.equal(case[2], case[1])
    end
  end)

  it("leaves a scalar unchanged under transpose", function()
    assert.equal(three, three:transpose())
    assert.same({}, three:transpose().shape)
  end)

  it("reduces a scalar tensor to itself under sum and mean", function()
    assert.equal(three, three:sum())
    assert.equal(three, three:mean())
  end)

  it("records both operands as parents, by identity, for scalar-broadcast ops", function()
    local cases = {
      { m:add(three), m,     three },
      { three:add(m), three, m },
      { m:sub(three), m,     three },
      { m:mul(three), m,     three },
    }
    for _, case in ipairs(cases) do
      local result, lhs, rhs = case[1], case[2], case[3]
      assert.equal(2, #result.parents)
      assert.is_true(rawequal(result.parents[1], lhs))
      assert.is_true(rawequal(result.parents[2], rhs))
    end
  end)
end)

describe("tensor backward propagation", function()
  local function clone_data(data)
    local out = {}
    for i, v in ipairs(data) do out[i] = v end
    return out
  end

  local function perturbed(t, idx, delta)
    local data = clone_data(t.data)
    data[idx] = data[idx] + delta
    return tensor.new(t.shape, data)
  end

  -- Central-difference directional derivative of forward_fn's output with
  -- respect to a single element of one input, dotted against seed_grad
  -- (dL/dOut). This is the numerical stand-in for what an analytic
  -- backward pass should compute, so `_backward` can be checked against it
  -- without ever encoding the calculus by hand in the test.
  local function numeric_partial(forward_fn, inputs, input_idx, elem_idx, seed_grad, eps)
    local plus, minus = {}, {}
    for i, t in ipairs(inputs) do
      plus[i], minus[i] = t, t
    end
    plus[input_idx] = perturbed(inputs[input_idx], elem_idx, eps)
    minus[input_idx] = perturbed(inputs[input_idx], elem_idx, -eps)
    local out_plus, out_minus = forward_fn(plus), forward_fn(minus)
    local diff = 0
    for k = 1, #out_plus.data do
      diff = diff + seed_grad.data[k] * (out_plus.data[k] - out_minus.data[k])
    end
    return diff / (2 * eps)
  end

  -- Runs forward_fn on fresh copies of `raw_inputs` (so each op under test
  -- starts from a clean, zero-initialized `.gradient`), seeds the output
  -- gradient, calls `_backward`, then checks every analytic parent
  -- gradient against its numerical estimate. This validates whatever
  -- chain-rule formula gets implemented, rather than hand-encoding the
  -- formula here too (and possibly making the same mistake twice).
  local function assert_gradcheck(forward_fn, raw_inputs, seed_data, eps, tolerance)
    local inputs = {}
    for i, t in ipairs(raw_inputs) do
      inputs[i] = tensor.new(t.shape, clone_data(t.data))
    end
    local out = forward_fn(inputs)
    local seed_grad = tensor.new(out.shape, seed_data)
    out.gradient = seed_grad
    out:_backward()

    for i, t in ipairs(inputs) do
      for elem = 1, #t.data do
        local numeric = numeric_partial(forward_fn, inputs, i, elem, seed_grad, eps)
        local analytic = t.gradient.data[elem]
        assert.near(numeric, analytic, tolerance)
      end
    end
  end

  local a = tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 })
  local c = tensor.new({ 2, 3 }, { 6, 5, 4, 3, 2, 1 })
  local b34 = tensor.new({ 3, 4 }, { 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 })
  local s = tensor.scalar(2)

  describe("invariants that should hold for every op", function()
    it("initializes every fresh tensor's gradient to zeros matching its shape", function()
      for _, t in ipairs({ a, a:add(c), a:sum(), a:scale(s), a:pow(s) }) do
        assert.is_not_nil(t.gradient)
        assert.same(t.shape, t.gradient.shape)
        assert.equal(#t.data, #t.gradient.data)
        for i = 1, #t.gradient.data do
          assert.equal(0, t.gradient.data[i])
        end
      end
    end)

    it("gives every non-leaf tensor a callable _backward", function()
      local results = {
        a:matmul(b34), a:transpose(), a:mul(c), a:add(c), a:sub(c),
        a:scale(s), a:pow(s), a:mean(), a:sum(),
      }
      for _, result in ipairs(results) do
        assert.equal("function", type(result._backward))
      end
    end)

    it("does not mutate data or shape when _backward runs", function()
      local x = tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 })
      local y = tensor.new({ 2, 3 }, { 6, 5, 4, 3, 2, 1 })
      local out = x:add(y)
      out.gradient = tensor.new(out.shape, { 1, 1, 1, 1, 1, 1 })
      out:_backward()
      assert.same({ 1, 2, 3, 4, 5, 6 }, x.data)
      assert.same({ 6, 5, 4, 3, 2, 1 }, y.data)
      assert.same({ 2, 3 }, x.shape)
    end)

    it("accumulates into existing parent gradient instead of overwriting it", function()
      local x = tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 })
      local y = tensor.new({ 2, 3 }, { 6, 5, 4, 3, 2, 1 })
      local out = x:add(y)
      x.gradient = tensor.new(x.shape, { 10, 10, 10, 10, 10, 10 })
      out.gradient = tensor.new(out.shape, { 1, 1, 1, 1, 1, 1 })
      out:_backward()
      for i = 1, #x.gradient.data do
        assert.equal(10 + 1, x.gradient.data[i])
      end
    end)

    it("sums contributions when a tensor is used as more than one input to the same op", function()
      local x = tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 })
      local out = x:add(x)
      out.gradient = tensor.new(out.shape, { 1, 2, 3, 4, 5, 6 })
      out:_backward()
      -- d(x+x)/dx = 2, applied once per occurrence, so both contributions
      -- land in the same x.gradient and should sum to 2 * out.gradient.
      for i = 1, #x.gradient.data do
        assert.equal(2 * out.gradient.data[i], x.gradient.data[i])
      end
    end)

    -- matmul and transpose write same-shaped gradients, so like every other
    -- op they must add onto whatever a parent already holds, not clobber it.
    -- (Overwriting only "works" when a parent has exactly one consumer and
    -- silently drops a contribution the moment it feeds two paths.)
    it("accumulates into existing parent gradients for matmul instead of overwriting", function()
      local x = tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 })
      local w = tensor.new({ 3, 2 }, { 1, 0, 0, 1, 1, 1 })
      local out = x:matmul(w)
      x.gradient = tensor.new(x.shape, { 100, 100, 100, 100, 100, 100 })
      w.gradient = tensor.new(w.shape, { 100, 100, 100, 100, 100, 100 })
      out.gradient = tensor.new(out.shape, { 1, 1, 1, 1 })
      out:_backward()
      -- contribution to x is out.grad · wᵀ = { 1, 1, 2, 1, 1, 2 }
      assert.same({ 101, 101, 102, 101, 101, 102 }, x.gradient.data)
      -- contribution to w is xᵀ · out.grad = { 5, 5, 7, 7, 9, 9 }
      assert.same({ 105, 105, 107, 107, 109, 109 }, w.gradient.data)
    end)

    it("accumulates into an existing parent gradient for transpose instead of overwriting", function()
      local y = tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 })
      local t = y:transpose()
      y.gradient = tensor.new(y.shape, { 100, 100, 100, 100, 100, 100 })
      t.gradient = tensor.new(t.shape, { 1, 2, 3, 4, 5, 6 })
      t:_backward()
      -- contribution to y is transpose(t.grad) = { 1, 3, 5, 2, 4, 6 }
      assert.same({ 101, 103, 105, 102, 104, 106 }, y.gradient.data)
    end)
  end)

  describe("gradient correctness (numerical gradient check per op)", function()
    it("add", function()
      assert_gradcheck(function(t) return t[1]:add(t[2]) end,
        { a, c }, { 1, -2, 0.5, 3, -1, 2 }, 1e-4, 1e-3)
    end)

    it("sub", function()
      assert_gradcheck(function(t) return t[1]:sub(t[2]) end,
        { a, c }, { 1, -2, 0.5, 3, -1, 2 }, 1e-4, 1e-3)
    end)

    it("mul (elementwise)", function()
      assert_gradcheck(function(t) return t[1]:mul(t[2]) end,
        { a, c }, { 1, -2, 0.5, 3, -1, 2 }, 1e-4, 1e-3)
    end)

    it("scale", function()
      assert_gradcheck(function(t) return t[1]:scale(t[2]) end,
        { a, s }, { 1, -2, 0.5, 3, -1, 2 }, 1e-4, 1e-3)
    end)

    it("transpose", function()
      assert_gradcheck(function(t) return t[1]:transpose() end,
        { a }, { 1, -2, 0.5, 3, -1, 2 }, 1e-4, 1e-3)
    end)

    it("pow", function()
      assert_gradcheck(function(t) return t[1]:pow(t[2]) end,
        { a, s }, { 1, -2, 0.5, 3, -1, 2 }, 1e-4, 1e-3)
    end)

    -- The squared-error loss (mean((pred - target)^2)) routinely raises a
    -- negative base to a constant power. The base gradient d(x^2)/dx = 2x is
    -- perfectly well-defined there; the exponent gradient involves log(x),
    -- which is not, so it must be kept finite rather than poisoning the graph
    -- with NaN (a scalar exponent's own gradient is discarded in practice).
    it("keeps pow gradients finite for negative bases", function()
      local x = tensor.new({ 1, 3 }, { -3, -1, 2 })
      local two = tensor.scalar(2)
      local out = x:pow(two)
      out.gradient = tensor.new(out.shape, { 1, 1, 1 })
      out:_backward()
      -- base gradient is 2x, valid regardless of sign
      assert.near(-6, x.gradient.data[1], 1e-9)
      assert.near(-2, x.gradient.data[2], 1e-9)
      assert.near(4, x.gradient.data[3], 1e-9)
      -- exponent gradient must not be NaN (NaN is the only value != itself)
      for i = 1, #two.gradient.data do
        assert.is_true(two.gradient.data[i] == two.gradient.data[i])
      end
    end)

    it("matmul", function()
      assert_gradcheck(function(t) return t[1]:matmul(t[2]) end,
        { a, b34 }, { 1, -2, 0.5, 3, -1, 2, 0.25, 4 }, 1e-4, 1e-2)
    end)

    it("sum", function()
      assert_gradcheck(function(t) return t[1]:sum() end,
        { a }, { 1.7 }, 1e-4, 1e-3)
    end)

    it("mean", function()
      assert_gradcheck(function(t) return t[1]:mean() end,
        { a }, { 1.7 }, 1e-4, 1e-3)
    end)
  end)

  describe("scalar to scalar gradient", function()
    it("add", function()
      assert_gradcheck(function(t) return t[1]:add(t[2]) end,
        { s, tensor.scalar(5) }, { 1.3 }, 1e-4, 1e-3)
    end)

    it("sub", function()
      assert_gradcheck(function(t) return t[1]:sub(t[2]) end,
        { s, tensor.scalar(5) }, { 1.3 }, 1e-4, 1e-3)
    end)

    it("mul", function()
      assert_gradcheck(function(t) return t[1]:mul(t[2]) end,
        { s, tensor.scalar(5) }, { 1.3 }, 1e-4, 1e-3)
    end)
  end)

  describe("broadcasting", function()
    describe("scalar to matrix gradient", function()
      describe("add", function()
        assert_gradcheck(function(t) return t[1]:add(t[2]) end,
          { s, c }, { 1, -2, 0.5, 3, -1, 2 }, 1e-4, 1e-3)
      end)
      describe("sub", function()
        assert_gradcheck(function(t) return t[1]:sub(t[2]) end,
          { s, c }, { 1, -2, 0.5, 3, -1, 2 }, 1e-4, 1e-3)
      end)
      describe("mul", function()
        assert_gradcheck(function(t) return t[1]:mul(t[2]) end,
          { s, c }, { 1, -2, 0.5, 3, -1, 2 }, 1e-4, 1e-3)
      end)
    end)

    describe("matrix to scalar gradient", function()
      describe("add", function()
        assert_gradcheck(function(t) return t[1]:add(t[2]) end,
          { c, s }, { 1, -2, 0.5, 3, -1, 2 }, 1e-4, 1e-3)
      end)
      describe("sub", function()
        assert_gradcheck(function(t) return t[1]:sub(t[2]) end,
          { c, s }, { 1, -2, 0.5, 3, -1, 2 }, 1e-4, 1e-3)
      end)
      describe("mul", function()
        assert_gradcheck(function(t) return t[1]:mul(t[2]) end,
          { c, s }, { 1, -2, 0.5, 3, -1, 2 }, 1e-4, 1e-3)
      end)
    end)

    -- Broadcasting a scalar out to a matrix-shaped result must not leak
    -- that matrix shape back into the scalar's own gradient: a parent's
    -- gradient should always match that parent's own shape, not the
    -- output's, regardless of which op produced the output.
    describe("keeps a scalar operand's gradient scalar-shaped after", function()
      for _, op in ipairs({ "add", "sub", "mul" }) do
        describe(op .. " with a matrix", function()
          local scalar = tensor.scalar(3)
          local matrix = tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 })
          local out = scalar[op](scalar, matrix)
          out.gradient = tensor.new(out.shape, { 1, 1, 1, 1, 1, 1 })
          out:_backward()
          assert.same({}, scalar.gradient.shape)
          assert.equal(1, #scalar.gradient.data)
        end)
      end
    end)
  end)
end)

describe("Tensor:backwards", function()
  local function clone_data(data)
    local out = {}
    for i, v in ipairs(data) do out[i] = v end
    return out
  end

  local function perturbed(t, idx, delta)
    local data = clone_data(t.data)
    data[idx] = data[idx] + delta
    return tensor.new(t.shape, data)
  end

  -- Builds a fresh, independent copy of raw_inputs (so backwards() never
  -- mutates the values under test), runs forward_fn to build the graph,
  -- calls backwards() on the (scalar) result, and checks every leaf's
  -- accumulated gradient against a central-difference numerical estimate
  -- obtained by re-running forward_fn from scratch on perturbed inputs.
  -- This validates full multi-op graph traversal (ordering, accumulation,
  -- shape handling) without hand-encoding any calculus here.
  local function assert_full_gradcheck(forward_fn, raw_inputs, eps, tolerance)
    local inputs = {}
    for i, t in ipairs(raw_inputs) do
      inputs[i] = tensor.new(t.shape, clone_data(t.data))
    end
    local out = forward_fn(inputs)
    out:backwards()

    for i, t in ipairs(inputs) do
      for elem = 1, #t.data do
        local plus, minus = {}, {}
        for j, raw in ipairs(raw_inputs) do
          plus[j] = tensor.new(raw.shape, clone_data(raw.data))
          minus[j] = tensor.new(raw.shape, clone_data(raw.data))
        end
        plus[i] = perturbed(inputs[i], elem, eps)
        minus[i] = perturbed(inputs[i], elem, -eps)
        local numeric = (forward_fn(plus).data[1] - forward_fn(minus).data[1]) / (2 * eps)
        assert.near(numeric, t.gradient.data[elem], tolerance)
      end
    end
  end

  it("errors when called on a non-scalar tensor", function()
    local m = tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 })
    assert.has_error(function() m:backwards() end)
  end)

  it("does not error when called on a scalar tensor", function()
    local s = tensor.scalar(5)
    assert.has_no.errors(function() s:backwards() end)
  end)

  it("seeds its own gradient to 1", function()
    local x = tensor.scalar(3)
    local y = tensor.scalar(4)
    local loss = x:add(y)
    loss:backwards()
    assert.equal(1, loss.gradient.data[1])
  end)

  it("leaves the gradient of a leaf with no path to the loss at zero", function()
    local x = tensor.scalar(3)
    local y = tensor.scalar(4)
    local unrelated = tensor.scalar(100)
    local loss = x:add(y)
    loss:backwards()
    assert.equal(0, unrelated.gradient.data[1])
  end)

  it("propagates correctly through a diamond dependency (one leaf feeding two paths that reconverge)", function()
    assert_full_gradcheck(function(t)
      local x = t[1]
      local a = x:mul(x)
      local b = x:add(tensor.scalar(5))
      return a:add(b)
    end, { tensor.scalar(3) }, 1e-4, 1e-3)
  end)

  it("propagates correctly through a diamond where the shared node has further parents of its own", function()
    -- w -> q -> {p1, p2} -> loss. A correct traversal must fully accumulate
    -- q's gradient (from both p1 and p2) before using it to propagate into
    -- w; visiting q's backward too early would silently drop a contribution.
    assert_full_gradcheck(function(t)
      local w = t[1]
      local q = w:mul(w)
      local p1 = q:add(tensor.scalar(2))
      local p2 = q:mul(tensor.scalar(3))
      return p1:add(p2)
    end, { tensor.scalar(2) }, 1e-4, 1e-3)
  end)

  it("propagates correctly through a matrix-valued diamond reduced to a scalar loss", function()
    assert_full_gradcheck(function(t)
      local x = t[1]
      local a = x:mul(x)
      local b = x:sub(tensor.scalar(1))
      return a:add(b):sum()
    end, { tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 }) }, 1e-4, 1e-3)
  end)

  it("propagates correctly through a long chain of ops (deep graph traversal)", function()
    assert_full_gradcheck(function(t)
      local x = t[1]
      local y = x
      for _ = 1, 8 do
        y = y:add(x):mul(tensor.scalar(0.5))
      end
      return y
    end, { tensor.scalar(1.5) }, 1e-4, 1e-2)
  end)

  it("propagates correctly when a tensor is reused across more than two consumers", function()
    assert_full_gradcheck(function(t)
      local x = t[1]
      local a = x:mul(tensor.scalar(2))
      local b = x:mul(tensor.scalar(3))
      local c = x:mul(tensor.scalar(5))
      return a:add(b):add(c)
    end, { tensor.scalar(4) }, 1e-4, 1e-3)
  end)

  it("propagates correctly through a mix of matmul and elementwise ops in one graph", function()
    assert_full_gradcheck(function(t)
      local x, w = t[1], t[2]
      local h = x:matmul(w)
      local y = h:mul(h)
      return y:sum()
    end, {
      tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 }),
      tensor.new({ 3, 2 }, { 1, 0, 0, 1, 1, 1 }),
    }, 1e-4, 1e-2)
  end)

  it("sums matmul and elementwise contributions when one leaf feeds both paths", function()
    -- x flows into a matmul path AND an elementwise path that reconverge. A
    -- matmul _backward that overwrote (rather than accumulated) x's gradient
    -- would silently drop whichever contribution the traversal wrote first.
    assert_full_gradcheck(function(t)
      local x, w = t[1], t[2]
      local h = x:matmul(w):sum()
      local g = x:mul(x):sum()
      return h:add(g)
    end, {
      tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 }),
      tensor.new({ 3, 2 }, { 1, 0, 0, 1, 1, 1 }),
    }, 1e-4, 1e-2)
  end)

  it("does not mutate leaf data or shapes when traversing the graph", function()
    local x = tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 })
    local loss = x:mul(x):sum()
    loss:backwards()
    assert.same({ 1, 2, 3, 4, 5, 6 }, x.data)
    assert.same({ 2, 3 }, x.shape)
  end)

  it("accumulates onto pre-existing parent gradients rather than overwriting them", function()
    local x = tensor.scalar(3)
    x.gradient = tensor.scalar(100)
    local loss = x:add(tensor.scalar(1))
    loss:backwards()
    assert.equal(101, x.gradient.data[1])
  end)
end)

describe("Tensor:zero_grad", function()
  it("resets a matrix tensor's gradient to zeros matching its shape", function()
    local m = tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 })
    m.gradient = tensor.new(m.shape, { 1, 2, 3, 4, 5, 6 })
    m:zero_grad()
    assert.same(m.shape, m.gradient.shape)
    assert.equal(#m.data, #m.gradient.data)
    for i = 1, #m.gradient.data do
      assert.equal(0, m.gradient.data[i])
    end
  end)

  it("resets a scalar tensor's gradient to zero", function()
    local s = tensor.scalar(5)
    s.gradient = tensor.scalar(42)
    s:zero_grad()
    assert.same({}, s.gradient.shape)
    assert.equal(0, s.gradient.data[1])
  end)

  it("does not mutate the tensor's own data or shape", function()
    local m = tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 })
    m.gradient = tensor.new(m.shape, { 9, 9, 9, 9, 9, 9 })
    m:zero_grad()
    assert.same({ 1, 2, 3, 4, 5, 6 }, m.data)
    assert.same({ 2, 3 }, m.shape)
  end)

  it("replaces the gradient tensor rather than mutating the old one in place", function()
    local m = tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 })
    local old_gradient = m.gradient
    old_gradient.data[1] = 7
    m:zero_grad()
    assert.is_false(rawequal(old_gradient, m.gradient))
    assert.equal(7, old_gradient.data[1])
  end)
end)
