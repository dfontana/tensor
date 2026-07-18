local tensor = require("lua.tensor")

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
      { a:matmul(i), a },
      { a:matmul(b):transpose(), b:transpose():matmul(a:transpose()) },
    }) do
      assert.equal(case[2], case[1])
    end
  end)

  it("satisfies element-wise addition and subtraction identities", function()
    local b = tensor.new({ 2, 3 }, { 6, 5, 4, 3, 2, 1 })
    local zero = tensor.new({ 2, 3 }, { 0, 0, 0, 0, 0, 0 })
    for _, case in ipairs({
      { a:add(b), b:add(a) },
      { a:add(b):sub(b), a },
      { a:sub(a), zero },
    }) do
      assert.equal(case[2], case[1])
    end
  end)

  it("satisfies element-wise multiplication identities", function()
    local b = tensor.new({ 2, 3 }, { 6, 5, 4, 3, 2, 1 })
    local zero = tensor.new({ 2, 3 }, { 0, 0, 0, 0, 0, 0 })
    local one = tensor.new({ 2, 3 }, { 1, 1, 1, 1, 1, 1 })
    for _, case in ipairs({
      { a:mul(b), b:mul(a) },
      { a:mul(one), a },
      { a:mul(zero), zero },
    }) do
      assert.equal(case[2], case[1])
    end
  end)

  it("satisfies scalar multiplication identities", function()
    local zero = tensor.new({ 2, 3 }, { 0, 0, 0, 0, 0, 0 })
    for _, case in ipairs({
      { a:scale(1), a },
      { a:scale(0), zero },
      { a:scale(2), a:add(a) },
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
