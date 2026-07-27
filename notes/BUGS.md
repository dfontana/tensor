# Known issues

Two findings from reading the library ahead of the Fennel port. Both are
independent of that port and can be fixed in the current Lua code. The first is a
genuine bug; the second is a hazard rather than a defect, but it blocks using
`main.lua` as a regression signal.

---

## 1. `Linear`'s bias is created without a gradient, crashing backward

**Status:** real bug, currently masked
**Files:** `lua/tensor.lua`, `lua/model.lua`

### Repro

```sh
lua -e '
local t = require("lua.tensor")
local m = require("lua.model")
local lin = m.Linear(2, 4)
lin:forward(t.new({4,2},{0,0,0,1,1,0,1,1})):sum():backward()
'
```

```
lua: ./lua/tensor.lua:473: attempt to index a nil value (local 'g')
stack traceback:
	./lua/tensor.lua:473: in function 'lua.tensor._accumulate_grad'
	./lua/tensor.lua:243: in method '_backward'
	./lua/tensor.lua:165: in function 'lua.tensor.backward'
```

### Cause

The two constructors disagree on what `require_grad` defaults to.

`Tensor.new` (`lua/tensor.lua:92`) defaults it **true**:

```lua
options = options or { require_grad = true }
```

`Tensor.fill` (`lua/tensor.lua:35`) — which `Tensor.zeroes` delegates to —
defaults it **false**:

```lua
options = options or {}
```

Both then do `gradient = options.require_grad and Tensor.zeroes(...) or nil`, so
`Tensor.zeroes(shape)` with no options yields a tensor whose `gradient` is `nil`.

`model.lua:21` builds the bias with exactly that call:

```lua
local bias = tensor.zeroes({ out_feats })
```

So every `Linear` has a bias with no gradient tensor. On the backward pass,
`add`'s rule calls `t:_accumulate_grad(ret.gradient)` with `t` = bias. Inside
`_accumulate_grad`, `local g = self.gradient` is `nil`; the bias shape `{4}` does
not match the output shape `{4,4}`, neither side is scalar, so control reaches the
vector-unbroadcast branch at line 473 and indexes `g`.

Confirmed directly — for the `main.lua` model, both biases are born gradient-less:

```
1  shape=(2,4)  grad=(2,4) [0,0,0,0,0,0,0,0]
2  shape=(4)    grad=nil
3  shape=(4,1)  grad=(4,1) [0,0,0,0]
4  shape=(1)    grad=nil
```

### Why nothing caught it

Two independent reasons:

- **The training loop heals it by accident.** `main.lua` calls `sgd:zero()` as the
  first statement of every iteration, before the forward pass. `zero_grad`
  *replaces* the gradient (`self.gradient = Tensor.zeroes(...)`) rather than
  mutating it in place, so the `nil` is gone before the first backward ever runs.
  Reorder those two lines, or use a `Linear` outside a training loop, and it
  crashes.
- **The specs never construct a parameter this way.** `optimizer_spec.lua` builds
  its params with `tensor.new` (which defaults `require_grad` true) and assigns
  `.gradient` by hand. No spec exercises `model.Linear` at all.

### Suggested fix

Route every constructor through one path that applies one default, rather than
having `new` and `fill` each decide. Then add the spec that would have caught it:
build a `Linear`, run forward → backward with no `optimizer:zero()` first, assert
the bias gradient is populated and correct.

Worth deciding at the same time: should `require_grad` default to true or false?
True matches `Tensor.new` and "parameters learn by default". False is cheaper for
intermediates, but every intermediate currently allocates a gradient tensor
anyway, so the flag is doing less work than it appears to.

### Note for the port

Under ops-as-data a leaf is simply a node with no `:op`, so this flag may be
better expressed as a single question — "does this node carry a `gradient` field
at all" — answered in one place. See `PORTING.md`.

---

## 2. `main.lua` is nondeterministic and cannot serve as a regression signal

**Status:** not a defect in the library; a hazard for the port
**File:** `lua/main.lua`

### Observation

Five consecutive runs of `mise run main`, final loss at step 5000:

```
1.0498062254369e-29
1.3089126832992e-29
3.6631986183936e-30
0.125
0.125
```

Three converge; two stall at exactly `0.125`.

### Cause

Nothing seeds the RNG, so `tensor.uniform` draws a different initialisation every
run. `0.125` is the classic XOR failure mode — the network gets two of the four
cases right and the loss is `mean{0, 0.25, 0.25, 0}`. With `uniform(-0.1, 0.1)`
init and ReLU, units die easily and the model settles into that local minimum.

This is a property of the training setup, not of the autograd. Verified: with
`math.randomseed(42)` fixed, the same model converges to ~1e-30 at both
`±0.1` and `±1.0` init scales, so the gradients are fine.

### Why it matters

`main.lua` is the only end-to-end exercise of the whole stack, and in its current
form it cannot tell you whether a change broke something — a stalled run is
indistinguishable from a bad draw. That is a problem well before the Fennel port;
any refactor of the autograd core has the same blind spot today.

### Suggested fix

Seed it. `math.randomseed(42)` at the top of `main.lua` makes the run
reproducible, which is enough on its own to make the loss curve meaningful.

Two follow-ups worth considering separately:

- **As a port oracle.** Fennel compiles to Lua and calls the same `math.random`,
  so identical seed *and* identical order of draws gives bit-identical loss curves
  across implementations. Any divergence pinpoints either changed math or a
  changed number/order of random draws. This requires porting `uniform` and
  `normal` without reordering their loops.
- **The underlying fragility.** Even seeded, a 2-4-1 ReLU net on XOR with
  `uniform(-0.1, 0.1)` and plain SGD is genuinely brittle. If you want it robust
  rather than merely reproducible, the usual levers are a wider init (scaled to
  fan-in, e.g. Xavier/He) or more hidden units. Worth doing as a learning exercise
  on initialisation rather than as a bug fix — the current behaviour is a correct
  implementation of a bad setup.
