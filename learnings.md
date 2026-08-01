# On Backprop Functions (Chain Rule)

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

## On Topological Sort & Backwards Pass

The naive approach materializes a full list of references in the graph. If I wanted to stream it, I'd have to track during the forward pass how many children each node has (count) so when children start backprop'ing, I can know which nodes are "ready" based on how many children counts hit 0. A child count decreases on the parent each time its child calls it's back_prop function.

This isn't free, it still requires tracking extra metadata per node (child ready count) vs a list of pointers, but does mean we can reduce peak memory consumption in favor of a larger total graph size "on disk".

# Next Token Predictors

Key components:
- A vocabulary (where each item is assigned some reverse-mappable, int ID)
- A model whose final output layer emits a 'logit' (score) per vocab entry
- A transformation converting those logits into a probability dist (sums to 1, min value 0) where the highest value is the next selected token.

During training the idea is based on the input so far the rewarded item should be the next more likely tokens. For example if we are learnig 'cat' from vocab 'a,c,t' we we are trying to predict:

(assuming context of size 1)
- input: c -> output a
- input: a -> output t

If our context was size 2 it would look like:
- input: c -> output a
- input: ca -> output t

## Input Vectors

Representing inputs:
- Can be one-hot, where our input vector == length of vocab, and we assign '1' to the token ('c') and '0' to everything else
- Embedded (this is what actual LLMs use); which is a shorter vector for each token that is _learned_.

### Embedding

If 'cat' in out one-hot representation is 10,000 numbers long (because we have 10k tokens in our vocab), then it's stating that only those tokens 'exist' and no other tokens relate to them. That's not _really_ true --> embeddeding picks a different length agnostic to the vocabularly (say 32 numbers) and each word ("cat", "dog") is trained upon. Those vectors in N-dimensional space can now convey similarity to each other (like a "dog" and "cat" are animals, or common household pets, etc) vs dissimilar (a "cat" is not a "car" but in training it may reveal a "cat" can be jokingly refered to as a "car" and there-fore a little more similar than a "cat" is to a "cabinet").

This embedding becomes a trainable table: {vocab-size, embed-length} -> in this example that would be 10k rows of 32 length instead of 10k rows of 10k length. What makes this interesting (and important for getting an embedding back to a token) is that you could multiply the one-shot represetation by the embedding table and get back the embedding for that token. That's because only one index has weight 1 and rest are 0 -- so you get just that one row back.

When it comes to learning, the interesting part is a token repeating in your input gets it's gradients updated multiple times; and any token NOT in your input gets nothing because all their weights are 0. This reveals a stark inefficiency -- if we just naively backprop we're updating a bunch of 0 gradients needlessly.

So the 'Embedding' layer does a avoids the mat-mul waste by instead doing an _index_ operation -- indexing the gradient matrix by the token's row (to access just that token's embedding) makes it the only gradient that need updating. All others can be skipped. It's almost like selecting the Embedding table's rows from your input batch and stacking them before feeding to the model, but importantly it's _one_ tensor node not N independent nodes. This if called _gather_ in many cases.

Because embeddings get learned during training, as similar tokens get used over and over the model naturally keeps updating their vectors similarly which pushes them closer together in vector space. 

#### Application

- You build a matrix of Vocab size rows x Embed dimension width, of random values. These will get trained while running model training -- they don't get trained separately. But this table is a parameter to the model itself (returned by parameters()).
  - In some cases they'll be pretrained and either frozen (not updated) or allowed to keep training ("fine-tuning")
- With your input tokens you one-shot them into a matrix -> pass them into the Embedding layer to avoid sparse matrix problem -> forward pass through the model -> final layer is then the length of your vocab so you get your logits.
- Softmax on the logit layer to then sample for the next token.

```
token ID
-> embedding row
-> rest of model
-> vocabulary logits
-> cross-entropy loss
```

## Logits & Softmax, Loss

Logits are not probabilities and can be any real number, relative to each other (rank). For example its possible one set of logits would be:

```
# a, c, t
 [2, 1, 0]
```

To actually make these a probability distribution we'd need to apply the `softmax` function on this 1D tensor -> exponentiate each element (removing negatives and signal boosting the higher values even higher) & divide by their total (making it a total out of 1).

- `exponentiate` means Euler's number `e` -> so you raise `e` to the logit value.
- `temperature` is a parameter you can use to divide each logit by to reduce how high disproportionately that probability rises -- a T of `1.0` does nothing, but (for example) a T of `0.6` "flattens" it a little bit. Fully this looks like `e^(Li/T)` (where `Li` is the element-wise logit)

When it comes to rewarding the model, loss needs to reward a correct next token much more when it's probability is high and punish it when the probability is lower. _Cross-entropy_ achieves that by computing loss as `-log(P)` where `P` is the probabillity of the correct next token that was assigned. A _low_ `P` has much higher loss than a _high_ `P`. This has to be generally computed across your target & prediction one-shot vectors element-wise. Target would be a one-hot vector embedding of the correct next term (eg '1' and not '0' at every other position) so this function would effectively select the target's probability:

```
-loss = sum(target[i]*log(P[i]))
```

Typically models stop at the "Logit" output and the softmax operation is paired with the cross-entropy function when computing loss (rather than having the model emit probabilities which aren't needed).

Depending on what you need really dictates when softmax is applied:
- Training: Logits -> Softmax+CrossEntropy
- Greedy inference (always pick highest) -> Just need logits
- Sampled generation -> Just apply softmax so you can sample the distribution rather than pick highest

## Context Size

Context size ultimately implicates how your training data is structured (how many tokens are referenced as input mapping to singular output token) AND how much the model takes in per training pass (likely padded with 0?)
