-- End-to-end training regression over the whole Fennel stack: tensor + model +
-- optimizer + vocab + backward, driven through fnl/main.fnl's scenarios.
--
-- Before phase 8 a sibling "oracle" spec cross-checked this against the Lua
-- reference for bit-identical loss curves. With the reference removed, the port
-- IS the implementation, so the meaningful regression is simply that the whole
-- stack still trains and the loss falls.
local assert = require("luassert")
local main = require("fnl.main")

describe("end-to-end training (whole Fennel stack)", function()
  it("drives the xor loss down over training (relu net + mean-squared)", function()
    math.randomseed(42)
    local losses = main["learn-xor"](300)
    assert.equal(300, #losses)
    -- Same threshold the seeded oracle used to hold: loss more than halves.
    assert.is_true(losses[#losses] < losses[1] * 0.5)
  end)

  it("learns the bigram next-token mapping (linear + cross-entropy)", function()
    math.randomseed(42)
    local losses = main["learn-bigram"](300)
    assert.is_true(losses[#losses] < losses[1])
  end)

  it("learns next-token prediction through an embedding (embedding + linear)", function()
    math.randomseed(42)
    local losses = main["learn-embeds"](300)
    assert.is_true(losses[#losses] < losses[1])
  end)
end)
