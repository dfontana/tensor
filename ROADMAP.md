5. Next Token predictors
  - Train on a small real token sequence rather than a hand-built cycle
  - Increase context size so predictions can depend on multiple earlier tokens.
5a. Attention / Simply GPT
  - Learn and implement attention as a way for tokens in a context to influence one another.
  - Add causal behavior: each position can only use earlier positions.
  - Build a small sequence next-token predictor around that.
  - Train and generate from a toy character- or word-level corpus.
5c. Tokenizer? tiktoken
6. Simd via rust + mlua + neon (plugable backend to the autograd core). Triton?
N-1. N-D Tensors -> Batching Optimizations, Structured Data (batches of sentences, attention heads, BatchSize=1 is just 2D).
  - Broadcasting beyond 2D limits
  - MatMul, Transpose, Gather
N+1. Sparse tensors?
N+2. Operator overloading? (Meh -- decided against during the Fennel port)
N+3. _backward stores an op-code not function to avoid allocating function on heap? Still have to capture parents in forward pass
N+4. Backward functions calling public methods create graphs that go to waste instead of just math functions

## Go learn about: strides and views (solves N-1 and N)

Right now a tensor is `{shape, data}` with hardcoded 2D indexing (`index_of`) and
a hand-rolled special case for broadcasting (`broadcast_output_shape` /
`broadcast_get`). N-D and broadcasting look like two roadmap items but they are
really one, and strides solve both.

A strided tensor carries a little more metadata:

```
{shape [2 3] strides [3 1] offset 0 data buf}
```

`strides[i]` is how many flat elements to step to advance one position along
dimension `i`. The element at `[i j k]` lives at
`offset + i*strides[1] + j*strides[2] + k*strides[3]`. Contiguous row-major
strides are the reverse cumulative product of the shape.

What falls out:

- **Transpose becomes free.** Swap two entries in `strides`. No data movement,
  no copy. Today `Tensor:transpose` allocates and copies every element.
- **Reshape becomes free** (when contiguous). Recompute `strides` from the new
  shape; `data` is untouched.
- **Broadcasting becomes a stride of 0.** A dimension with stride 0 reads the
  same element for every index along it — that *is* repetition. `{3}` against
  `{2,3}` becomes `shape [2 3], strides [0 1]`. The whole of `broadcast_get`
  disappears, and NumPy's real rule (size-1 stretching, which is currently
  rejected) comes along for free.
- **Unbroadcasting in the backward pass becomes mechanical:** sum over exactly
  the axes whose stride was 0. That replaces the shape-case ladder in
  `_accumulate_grad`.
- **Slicing and views** become possible at all, via `offset` plus adjusted shape.

The cost, and the thing to actually go understand: kernels now receive possibly
**non-contiguous** views, so they can no longer walk `data` linearly. The standard
answer is an explicit `contiguous` materialization step before a kernel runs, with
a fast path when the strides are already dense. This is what tinygrad's
ShapeTracker does, and it is also why a strided core is a *prerequisite* for the
SIMD/Vulkan backend (item 6) rather than a distraction from it — a bulk kernel
wants contiguous memory, and something has to decide when to produce it.

Questions worth answering by building it: what does `matmul` do with a transposed
(stride-swapped) input — handle it, or force `contiguous` first? Can two views
share one buffer safely when the optimizer mutates parameters in place?

## Go learn about: tape-based autograd (alternative to the topo sort)

`Tensor:backward` currently walks the graph with an iterative DFS, pushing each
node twice (unexpanded/expanded) to get a post-order, then replays it in reverse.
That is correct and explicit, and it is worth keeping while learning.

The alternative: once ops are data, append every node to a shared **tape** as it
is created. Node creation order is *already* a valid topological order — a node
cannot be built before its parents exist. So the entire backward pass becomes:

```
for i = #tape, 1, -1 do backward_step(tape[i]) end
```

The DFS, the `seen` set, and the expanded/unexpanded bookkeeping all disappear.
This is how autograd and JAX-style systems work.

What it costs, and the real reason to think before adopting it:

- **Implicit state.** Something has to hold "the current tape". In Fennel that
  means a module-level `var` plus a `with-tape` macro that saves and restores it —
  which is, incidentally, one of the few places a macro is genuinely the right
  tool (save/restore around a body is exactly what macros are for).
- **Lifetime.** The tape is per-forward-pass. Reusing a node across two tapes, or
  forgetting to reset, gives wrong answers quietly rather than loudly.
- **It hides the thing you were learning.** The explicit graph walk makes
  "why must a node's gradient be fully accumulated before it propagates further"
  visible; the tape makes it true by construction, which is great engineering and
  worse pedagogy.

Also see the note in `learnings.md` on the streaming alternative: track a child
count per node during the forward pass and process a node once all its children
have reported in. That trades a list of pointers for a counter per node, reducing
peak memory at the cost of more per-node metadata. Three designs, same result —
worth implementing at least two to feel the difference.
