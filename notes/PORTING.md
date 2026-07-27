# Porting the tensor library to Fennel

A plan, not an implementation. Every code block here is a sketch or a signature —
the bodies are deliberately left blank. Fill them in; that is the point.

## Scope

**In scope:** a faithful port of today's semantics to Fennel, plus exactly one
design change — the autograd graph becomes **data instead of closures**.

**Out of scope**, deliberately, until learning demands them:

| Deferred | Why not now |
| --- | --- |
| Strides / views | Big structural change. Belongs with the N-D task. See ROADMAP. |
| Tape-based autograd | Replaces the topo sort you just wrote and understood. See ROADMAP. |
| Operator overloading | Ambiguous against numeric `+`. Not worth it. |
| Pluggable SIMD/Vulkan backend | Needs the op graph to exist first. Ops-as-data is the prerequisite. |

The port should end with the same behaviour, the same test suite passing, and a
graph you can *inspect* rather than only *call*.

---

## The one design change: ops as data

Today a node carries a closure:

```lua
ret.parents = { self, t }
ret._backward = function() ... end
```

After the port a node carries an **op tag** and its parents:

```fennel
{:shape [2 3] :data [...] :gradient g :op :matmul :parents [a b]}
```

and the whole backward pass is one dispatch:

```fennel
(fn backward-step! [node]
  (let [g node.gradient]
    (case node
      {:op :add    :parents [a b]} ...
      {:op :sub    :parents [a b]} ...
      {:op :mul    :parents [a b]} ...
      {:op :matmul :parents [a b]} ...
      {:op :relu   :parents [a]}   ...
      ;; ... one clause per op; leaves fall through
      _ nil)))
```

Fennel's `case` destructures by key and by position in the same pattern, so each
clause reads as the chain rule for that op — the table in `learnings.md` becomes
literally the code.

### Why this is the change worth making

It closes roadmap items **N+3** and **N+4** at once:

- **N+4 (wasted graphs).** Today `mul`'s backward calls `ret.gradient:mul(t)` —
  a *public, tracked* op. Every backward pass allocates a fresh subgraph plus a
  gradient tensor for it, forever, and then throws it away. Inside
  `backward-step!` you instead call the **untracked kernels**, so no graph is
  built. This is the real payoff and it requires the two-layer split below.
- **N+3 (opcode not closure).** You get something better than a switch on an
  integer: a pattern that identifies the op *and* binds its parents in one form.

It also makes the graph inspectable, which is the entry ticket to the Vulkan and
SIMD work — a closure is opaque, `{:op :matmul :parents [...]}` is a program you
can walk, fuse, and schedule.

### The two-layer split this forces

Every operation exists twice:

```
kernels   pure math on {shape, data}. No graph, no gradient. Suffix `*`.
          add* sub* mul* matmul* transpose* relu* pow* scale* sum* mean*

ops       kernel + record the node. What callers use.
          add sub mul matmul transpose relu pow scale sum mean
```

`backward-step!` calls **only** kernels. Public ops call the kernel then attach
`:op` and `:parents`. This split is also exactly where a future backend plugs in:
swap the kernel table, leave everything else alone.

---

## Layout and conventions

```
fnl/
  tensor.fnl      shape helpers, kernels, ops, backward-step!, backward!
  optimizer.fnl
  model.fnl
  main.fnl
lua/              the existing implementation — keep until the port is green
test/
  impl.lua        selects which implementation the specs run against
  compat.lua      temporary method-shim over the Fennel core
```

Naming: kebab-case; `!` suffix for mutation; `?` for predicates.

| Lua | Fennel |
| --- | --- |
| `Tensor.numel` | `numel` |
| `Tensor:_is_scalar` | `scalar?` |
| `Tensor:_eq_shape` | `same-shape?` |
| `Tensor:zero_grad` | `zero-grad!` |
| `Tensor:_accumulate_grad` | `accum-grad!` |
| `Tensor:backward` | `backward!` |
| `Tensor:_backward` | *(gone — replaced by `backward-step!`)* |

Free functions taking the tensor first, so threading works:

```fennel
(-> x (matmul w) (add b) relu)
```

Keep a metatable for `__eq` and `__tostring` only. No methods in the core.

### One hard rule: do not rename fields

Keep `shape`, `data`, `gradient`, `parents` **exactly as they are**. The entire
regression strategy below depends on the Lua specs being able to read a Fennel
tensor's fields directly. `gradient` → `grad` is nicer Fennel, and it is not
worth what it costs you here. Rename it after the port, in one commit, if at all.

---

## Keeping the tests working throughout

This is the part that matters most, so it comes with the most detail.

### What you already have

The suite is a much better safety net than it looks. `test/tensor_spec.lua`
already contains:

- `assert_gradcheck` — central-difference numerical gradient check, **per op**
- `assert_full_gradcheck` — the same across a whole multi-op graph
- identity/algebraic properties for every forward op
- parent-identity tracking tests using `rawequal`
- accumulate-don't-overwrite tests, including the diamond cases

Crucially, the gradcheck helpers **never encode the calculus by hand** — they
compare against finite differences. That means they validate a *reimplementation*
on its own merits, not merely agreement with the old code. They will catch a
wrong `case` clause in `backward-step!` immediately. Do not rewrite them; make
them run against Fennel.

### Step 1: an implementation switch

Add `test/impl.lua`, resolving each module independently so you can run **mixed
mode** — a ported Fennel tensor underneath the still-Lua optimizer:

```lua
-- Resolve one module: prefer the Fennel port when TENSOR_IMPL=fnl and the
-- ported module exists, else fall back to the Lua original.
local function pick(name)  -- e.g. pick("tensor")
  -- ... blank: try require("test.compat")/require("fnl."..name), pcall,
  --     fall back to require("lua."..name)
end

return { tensor = pick("tensor"), optimizer = pick("optimizer") }
```

Then change the specs' `require("lua.tensor")` to read from `test.impl`. Do this
**first, while everything is still Lua**, and confirm the suite is green. You now
have a switch you trust.

Mixed mode works because `optimizer.lua` and `model.lua` only ever touch `.data`,
`.gradient.data`, and a couple of methods. Port `tensor` completely and validate
it before anything else moves.

### Step 2: a throwaway method shim

The specs call methods (`a:matmul(b)`, `out:_backward()`); the Fennel core exports
free functions. Bridge them once, in `test/compat.lua`, rather than editing 900
lines of spec:

```lua
local core = require("fnl.tensor")

-- The core already puts a metatable on every node (for __eq/__tostring).
-- Decorate its __index with method wrappers; the core stays method-free.
local methods = {
  matmul = function(self, t) return core.matmul(self, t) end,
  -- ... add, sub, mul, scale, pow, relu, transpose, sum, mean
  -- _index, _sshape, zero_grad, backward
  _backward = function(self) return core["backward-step!"](self) end,
}
-- ... blank: merge `methods` into core.mt.__index, re-export new/scalar/zeroes
```

Three things to note:

- `_backward` maps cleanly onto `backward-step!`. Today's closure reads
  `ret.gradient` and accumulates into parents; so does the dispatch. The
  ~40 tests that call `out:_backward()` directly keep working unchanged.
- Fennel exports kebab-case names, which from Lua means `core["zero-grad!"]`.
  Ugly, but `compat.lua` is the only Lua consumer and it is temporary.
- `test/helper.lua` checks `getmetatable(value) == tensor` for its formatter.
  Point it at the core's metatable.

Delete `compat.lua` at the end, when the specs move to Fennel (optional, last).

### Step 3: a seeded end-to-end oracle

`main.lua` is **nondeterministic** and unusable as a signal until seeded — see
`BUGS.md` #2.

Add `math.randomseed(42)` at the top. Fennel compiles to Lua and calls the same
`math.random`, so **identical seed + identical order of draws ⇒ bit-identical
loss curves** between the two implementations. Compare a few hundred steps of
Lua output against Fennel output; any divergence means you changed the number or
order of random draws, or changed the math. That is a very sharp check, and it
covers the whole stack at once.

It only holds if draw order is preserved — so port `uniform` and `normal` without
reordering their loops.

---

## Suggested order

Each phase names the spec blocks that should go green. Run
`TENSOR_IMPL=fnl` after every one and watch the failure count shrink.

**0 — Toolchain.** Add fennel to `mise.toml` (mise suggests
`mise use -g http:fennel@1.5.3`). Install `fennel.searcher` in `test/helper.lua`
so `require("fnl.tensor")` resolves `.fnl` transparently. Add a `test:fnl` task.
No library code yet.

**1 — The switch.** `test/impl.lua` + spec `require` changes, still all Lua.
Suite must be green before you continue.

**2 — Shape helpers.** `numel`, `scalar?`, `same-shape?`, `shape->string`,
`index-of`, `broadcast-shape`, `broadcast-get`. Pure, no autograd. Note that
`broadcast-shape` is where Fennel pattern matching pays off — the scalar /
same-shape / vector-against-last-dim cases are `case` clauses with guards, not a
ladder of `if`s.

**3 — Kernels (`*` layer).** Every forward op as pure math on `{shape data}`.
→ green: the whole `describe("tensor")` block, plus the forward assertions in the
scalar-operand and vector-operand blocks, including the rejection cases.

**4 — Nodes and tracked ops.** Constructors (`tensor`, `scalar`, `zeros`, `fill`,
`uniform`, `normal`) and the public ops that wrap kernels with `:op`/`:parents`.
→ green: `describe("tensor parent tracking")` in full, the parent-recording tests
in the broadcast blocks, and the fresh-gradient invariant.

**5 — `backward-step!` and `accum-grad!`.** The `case` dispatch, one clause per
op, calling only kernels. `accum-grad!` keeps today's unbroadcast logic.
→ green: `describe("tensor backward propagation")` — every invariant and every
gradcheck. This is the big one; expect to spend most of your time here.

**6 — `backward!`.** Port the existing iterative DFS topo sort as-is. Do not
change the algorithm in the same commit that changes the node representation.
→ green: `describe("Tensor:backward")`, `describe("Tensor:zero_grad")`.

**7 — Optimizer, then model, then main.** Small and mechanical by comparison.
`model.fnl` gets simpler: params are static after construction, so make them a
value rather than a method, and a `relu` layer becomes `{:forward relu :params []}`
with no wrapper closure.
→ green: `optimizer_spec`, then the seeded main comparison.

**8 — Cleanup.** Delete `compat.lua` and `lua/`, optionally port the specs to
Fennel.

---

## Decisions to make while porting

**`require_grad`'s inconsistent default** (see `BUGS.md` #1 — `Tensor.new` and
`Tensor.fill` disagree, so `Linear`'s bias is born with `gradient = nil`). Fix it
before or during the port, not by transliterating the inconsistency. Under
ops-as-data a leaf is just a node with no `:op`, so you may find the flag is
better expressed as a single question — "does this node carry a `gradient` field
at all" — answered in one place.

**Preserve the `pow` exponent-gradient guard.** `log(x)` is undefined for `x <= 0`,
so today's code contributes `0` there rather than NaN-poisoning the graph. The
`:pow` clause in `backward-step!` must reproduce this; the spec at
`tensor_spec.lua:545` pins it.

**Resist macros early.** A `defop` macro that generates the forward wrapper and
registers the backward clause is tempting. Write the ops as plain functions first.
Fennel macros are compile-time, live in a separate file, and worsen error
messages; extract one only if the boilerplate genuinely repeats. Prefer functions
to macros is Fennel's own guidance.

**You will lose the LuaLS `---@` annotations.** Fennel has no equivalent. Partial
compensation: `lambda`/`λ` gives arity checking, and Fennel has first-class
docstrings you can query from the REPL with `doc`. It is a trade — static types
for runtime introspection — not a pure win. Go in knowing that.
