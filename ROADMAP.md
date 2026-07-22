3. Zerograd: Basic SGD
  - Add learning function to a tensor's data (d[i] -= rate * grad[i])
  - An optimizer class should hold parameter tensors, so it can:
    - step: applies learning to parameter tensors
    - zero: clears the gradients of the parameter tensors
  - Goal:
    - init(params) -> zero -> build net `loss=forward()` -> step()
    - On a linear regression:
      - prediction = w*x + b
      - loss = mean((prediction - target)^2)
      - Where loss should decrease over steps, w/b change, gradients/graphs reset
4. Tinygrad: Linear, relu, loss function -- layers!
5. Attention / Next Token predictors (Simple gpt)
6. Simd via rust + mlua + neon (plugable backend to the autograd core)
N-1. N-D Tensors -> Batching Optimizations, Structured Data (batches of sentences, attention heads, BatchSize=1 is just 2D).
N. Broadcasting?

