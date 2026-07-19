## On Backprop Functions (Chain Rule)

Every back function is just the chain rule for that operation, accumulating on each parent. So in the operation of c = a + b, the question is how much does each parent - relative to any other parent - influence how c changes? Not actual delta -- the gradient in c already describes how 'sensitive' it is to changes. The back-prop question is "how much of that sensitive can we attribute to each parent?". For addition, both terms equally contribute, so we add c's gradient to both a & b's. It doesn't _matter_ how large each is -- they have _equal opportunity_ to influence.

So for:
 Addition both terms contribute equally, we add back c.grad
 Subtraction, a adds positively, but b is subtracted sub c.grad there
 Multiplication, it'd added still but we *weigh* by the opposite term's data; because a is added b times, and b is added a times (in a * b)
 Sum is the same thing as Add
 Mean is like Add, but divided by total elements (as their influence)
 Transpose doesn't scale, it just moves
 MatMul is the dot product of rows and columns yielding an output "coordinate", so when we apply c.grad backwards the addition part says all terms in that dot product are equally influential, but then for each pair-wise index we have to apply the multiplication rule. since the same index gets involved in multiple dot products, it mostly means we're going to update grads multiple times by "chance"
Chain rule: Take upstream gradient and multiply by local derivative, acc to each parent gradient

### Deeper on mat-mul backwards function

Maintaining the shape is the interesting part. It's fairly evident that each index in the output C is contributed "equally" by each Row.i*Col.i during the dot product calculation -- so it looks JUST like the elementwise multiplication. The main difference is multiple fields get touched by each inbound index since it's dot-product.

So the gradient update of A.i is effectively taking the sum of C.i * B.i for the entire Row/Col of C/B. In otherwords, it's the dot product again, but flipping the other matrix so the shape "fits".

```lua
-- AAA   BB   CC
-- AAA   BB   CC
--       BB
--
-- A^T dot C = B
-- AA  CC  BB
-- AA  CC  BB
-- AA      BB
--
-- C dot B^T = A
-- CC  BBB  AAA
-- CC  BBB  AAA
```
