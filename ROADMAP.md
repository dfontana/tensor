1. Element wise ops (add, sub, sum, element-wise multiply)
2. Graph rec (parents + back function for each op)
  - Parents: Each op returns a new tensor pointing back to it's parents that made it (so tensors start storing list of tensors). Order matters.
  - Back Func: Sends gradient back to parents ("given change in output, how much should each input change. Local derivatives, chain rule.").
  - Need to start storing grad (0) - the loss tensor starts at 1 (this tracks the error), and then walks backwards additively
3. Autograd: Topo sort + backprop
  - Sort graph, walk backwards calling each back function.
  -> Add tiny optimizer (sgd)
  - (Consider reading Micrograd)
4. Tinygrad: Linear, relu, loss function -- layers!
5. Attention / Next Token predictors (Simple gpt)
6. Simd via rust + mlua + neon (plugable backend to the autograd core)
N-1. N-D Tensors -> Batching Optimizations, Structured Data (batches of sentences, attention heads, BatchSize=1 is just 2D).
N. Broadcasting?

