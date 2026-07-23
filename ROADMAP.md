4. Tinygrad: Linear, relu, loss function -- layers!
5. Attention / Next Token predictors (Simple gpt)
6. Simd via rust + mlua + neon (plugable backend to the autograd core)
N-1. N-D Tensors -> Batching Optimizations, Structured Data (batches of sentences, attention heads, BatchSize=1 is just 2D).
N. Broadcasting?
N+1. Sparse tensors?
N+2. Operator overloading? (Meh)
N+3. _backward stores an op-code not function to avoid allocating function on heap? Still have to capture parents in forward pass
N+4. Backward functions calling public methods create graphs that go to waste instead of just math functions
