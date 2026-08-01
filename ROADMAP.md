5. Next Token predictors
  - A training dataset, which means a vocabulary and correct next token mappings. If we assume a context size of 1 that's token -> token rather than list[token] -> token. The model's input size would also change (padded with 0s?) but never the less
5a. Attention / Simply GPT
5b. Embeddings?
 5bi. What does this mean for vocabulary?
5c. Tokenizer? tiktoken
6. Simd via rust + mlua + neon (plugable backend to the autograd core)
N-1. N-D Tensors -> Batching Optimizations, Structured Data (batches of sentences, attention heads, BatchSize=1 is just 2D).
N. Broadcasting?
N+1. Sparse tensors?
N+2. Operator overloading? (Meh)
N+3. _backward stores an op-code not function to avoid allocating function on heap? Still have to capture parents in forward pass
N+4. Backward functions calling public methods create graphs that go to waste instead of just math functions
