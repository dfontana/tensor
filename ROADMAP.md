For the rest of the roadmap we can use [tiny shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) as the training data.

5. Strides
  - Support N-D tensors through strides (slicing/offsets not needed)
  - Generalized Broadcasting
6. self-attention & transformer blocks
  - positions
  - casual masking
  - Q/K/V
  - transformer blocks
  - Train on actual data, but simple word splitting "tokenization"
  - Increase context size to 3 then 16 (etc)
6a. Grouped Query Heads attention
6b. KV Caching
7. tokenization (tiktoken, etc)
4. Native CPU v1
  - Fennel tensors keep Lua metadata/autograd tables with optional GC-owned f32 Storage data
  - Exact-shape contiguous elementwise kernels, reductions, rank-2 matmul, same-shape gradient accumulation, and SGD use the native module
  - Broadcasting, views/transpose, gather, softmax, and other complex operations remain Fennel until the stride milestone
  - The explicit `TENSOR_FORCE_NO_NATIVE=1` test task verifies the genuine Lua-table fallback
N-1. Batching matrix ops
N+1. Sparse tensors?
N+2. SIMD / GPU support
  - Kernels may need to materialize/make data contiguous when it's necessary/more efficient. This requires tracking if the data is contiguous and deciding when it's better to do so than strided access
