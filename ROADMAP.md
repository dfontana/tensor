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
N+2. Operator overloading? (Meh)
N+3. _backward stores an op-code not function to avoid allocating function on heap? Still have to capture parents in forward pass
N+4. Backward functions calling public methods create graphs that go to waste instead of just math functions
