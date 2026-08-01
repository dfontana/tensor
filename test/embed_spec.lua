local tensor = require("lua.tensor")
local embed = require("lua.embed")
local assert = require("luassert")

local Vocab = embed.Vocab

describe("Vocab:size", function()
  it("reports the number of tokens it was constructed with", function()
    assert.equal(3, Vocab:new({ "a", "b", "c" }):size())
    assert.equal(0, Vocab:new({}):size())
  end)
end)

describe("Vocab:encode", function()
  it("maps each token to its 1-based position in the vocabulary", function()
    local vocab = Vocab:new({ "cat", "dog", "fish" })
    assert.equal(1, vocab:encode("cat"))
    assert.equal(2, vocab:encode("dog"))
    assert.equal(3, vocab:encode("fish"))
  end)

  it("errors on a token that is not in the vocabulary", function()
    local vocab = Vocab:new({ "cat", "dog" })
    assert.has_error(function() vocab:encode("bird") end, "Token not in vocab: bird")
  end)
end)

describe("Vocab:one_hot", function()
  it("encodes a token as a 1-D one-hot vector the size of the vocabulary", function()
    local vocab = Vocab:new({ "cat", "dog", "fish" })
    local hot = vocab:one_hot("dog")
    assert.same({ 3 }, hot.shape)
    -- "dog" is id 2, so only the second slot is set.
    assert.same({ 0, 1, 0 }, hot.data)
    -- built with require_grad = false, so it carries no gradient buffer
    assert.is_nil(hot.gradient)
  end)

  it("errors on a token that is not in the vocabulary", function()
    assert.has_error(function() Vocab:new({ "cat" }):one_hot("bird") end)
  end)
end)
