3. Autograd: Topo sort + backprop
  - Sort graph, walk backwards calling each back function.
  -> Add tiny optimizer (sgd)
  - (Consider reading Micrograd)
4. Tinygrad: Linear, relu, loss function -- layers!
5. Attention / Next Token predictors (Simple gpt)
6. Simd via rust + mlua + neon (plugable backend to the autograd core)
N-1. N-D Tensors -> Batching Optimizations, Structured Data (batches of sentences, attention heads, BatchSize=1 is just 2D).
N. Broadcasting?

