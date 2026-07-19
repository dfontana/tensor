local tensor = require("lua.tensor")
local assert = require("luassert")

local a = tensor.new({ 2, 3 }, { 1, 2, 3, 4, 5, 6 })

describe("tensor", function()
  it("indexes elements", function()
    for _, case in ipairs({ { 1, 1, 1 }, { 1, 3, 3 }, { 2, 2, 5 } }) do
      assert.equal(case[3], a:index(case[1], case[2]))
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
      { a:scale(tensor.new({}, { 1 })), a },
      { a:scale(tensor.new({}, { 0 })), zero },
      { a:scale(tensor.new({}, { 2 })), a:add(a) },
    }) do
      assert.equal(case[2], case[1])
    end
  end)

  it("reduces tensors to scalar tensors", function()
    local scalar = tensor.new({}, { 21 })
    assert.equal(scalar, a:sum())
    assert.equal(a:sum(), a:transpose():sum())
  end)

  it("computes means as scalar tensors", function()
    local scalar = tensor.new({}, { 3.5 })
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
  local s = tensor.new({}, { 2 })

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
      { a:sum(),        a },
      { a:mean(),       a },
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
      a:scale(s), a:mean(), a:sum(),
    }
    for _, result in ipairs(results) do
      assert.is_true(#result.parents > 0)
    end
  end)
end)
