(local tensor (require :fnl.tensor))
(local assert (require :luassert))
(local assert-data (. (require :test.helper) :assert-data))

; Native f32 values quantize a perturbation smaller than one ulp. Keep the
; fallback tests at their original step while using a stable finite difference
; step for native Storage.
(local finite-difference-step
  (fn [t eps]
    (if (= (type t.data) "userdata")
        (math.max eps 0.01)
        eps)))

(local a (tensor.new [2 3] [1 2 3 4 5 6]))

(describe "tensor"
  (fn []
    (it "multiplies matrices"
      (fn []
        (let [b (tensor.new [3 4] [7 8 9 10 11 12 13 14 15 16 17 18])]
          (assert.equal
            (tensor.new [2 4] [74 80 86 92 173 188 203 218])
            (tensor.matmul a b)))))

    (it "transposes matrices"
      (fn []
        (assert.equal
          (tensor.new [3 2] [1 4 2 5 3 6])
          (tensor.transpose a 1 2))))

    (it "satisfies matrix identities"
      (fn []
        (let [b (tensor.new [3 4] [7 8 9 10 11 12 13 14 15 16 17 18])
              i (tensor.new [3 3] [1 0 0 0 1 0 0 0 1])
              cases [[(tensor.transpose (tensor.transpose a 1 2) 1 2) a]
                     [(tensor.matmul a i) a]
                     [(tensor.transpose (tensor.matmul a b) 1 2)
                      (tensor.matmul (tensor.transpose b 1 2) (tensor.transpose a 1 2))]]]
          (each [_ entry (ipairs cases)]
            (assert.equal (. entry 2) (. entry 1))))))

    (it "satisfies element-wise addition and subtraction identities"
      (fn []
        (let [b (tensor.new [2 3] [6 5 4 3 2 1])
              zero (tensor.new [2 3] [0 0 0 0 0 0])
              cases [[(tensor.add a b) (tensor.add b a)]
                     [(tensor.sub (tensor.add a b) b) a]
                     [(tensor.sub a a) zero]]]
          (each [_ entry (ipairs cases)]
            (assert.equal (. entry 2) (. entry 1))))))

    (it "satisfies element-wise multiplication identities"
      (fn []
        (let [b (tensor.new [2 3] [6 5 4 3 2 1])
              zero (tensor.new [2 3] [0 0 0 0 0 0])
              one (tensor.new [2 3] [1 1 1 1 1 1])
              cases [[(tensor.mul a b) (tensor.mul b a)]
                     [(tensor.mul a one) a]
                     [(tensor.mul a zero) zero]]]
          (each [_ entry (ipairs cases)]
            (assert.equal (. entry 2) (. entry 1))))))

    (it "satisfies scalar multiplication identities"
      (fn []
        (let [zero (tensor.new [2 3] [0 0 0 0 0 0])
              cases [[(tensor.scale a (tensor.scalar 1)) a]
                     [(tensor.scale a (tensor.scalar 0)) zero]
                     [(tensor.scale a (tensor.scalar 2)) (tensor.add a a)]]]
          (each [_ entry (ipairs cases)]
            (assert.equal (. entry 2) (. entry 1))))))

    (it "raises elements to a scalar power"
      (fn []
        (assert-data [1 4 9 16 25 36]
                     (. (tensor.pow a (tensor.scalar 2)) :data))))

    (it "satisfies power identities"
      (fn []
        (let [ones (tensor.new [2 3] [1 1 1 1 1 1])
              cases [[(tensor.pow a (tensor.scalar 1)) a]
                     [(tensor.pow a (tensor.scalar 0)) ones]]]
          (each [_ entry (ipairs cases)]
            (assert.equal (. entry 2) (. entry 1))))))

    (it "applies relu elementwise, clamping negatives to zero"
      (fn []
        (let [m (tensor.new [2 3] [-3 -1 0 2 -4 5])]
          (assert.equal
            (tensor.new [2 3] [0 0 0 2 0 5])
            (tensor.relu m))
          ; relu is the identity on an all-nonnegative tensor
          (assert.equal a (tensor.relu a)))))

    (it "reduces tensors to scalar tensors"
      (fn []
        (let [scalar (tensor.scalar 21)]
          (assert.equal scalar (tensor.sum a))
          (assert.equal (tensor.sum a) (tensor.sum (tensor.transpose a 1 2))))))

    (it "computes means as scalar tensors"
      (fn []
        (let [scalar (tensor.scalar 3.5)]
          (assert.equal scalar (tensor.mean a))
          (assert.equal (tensor.mean a) (tensor.mean (tensor.transpose a 1 2))))))

    (it "compares tensors"
      (fn []
        (let [cases [[a true]
                     [(tensor.new [3 2] [7 8 9 10 11 12]) false]
                     [1 false]]]
          (each [_ entry (ipairs cases)]
            (assert.equal (. entry 2) (= a (. entry 1)))))))))

(describe "tensor parent tracking"
  (fn []
    (local a (tensor.new [2 3] [1 2 3 4 5 6]))
    (local b (tensor.new [3 4] [7 8 9 10 11 12 13 14 15 16 17 18]))
    (local c (tensor.new [2 3] [6 5 4 3 2 1]))
    (local s (tensor.scalar 2))

    ; A dedicated tensor identical in value to `a` but a distinct object, to
    ; distinguish identity tracking (what backprop needs) from value equality
    ; (what __eq checks).
    (local a-lookalike (tensor.new [2 3] [1 2 3 4 5 6]))

    (it "gives leaf tensors no parents"
      (fn []
        (assert.equal 0 (length a.parents))
        (assert.equal 0 (length s.parents))))

    (it "records exactly the inputs, by identity, as parents for unary ops"
      (fn []
        (let [cases [[(tensor.transpose a 1 2) a]
                     [(tensor.sum a) a]
                     [(tensor.mean a) a]
                     [(tensor.relu a) a]]]
          (each [_ entry (ipairs cases)]
            (let [result (. entry 1)
                  input (. entry 2)]
              (assert.equal 1 (length result.parents))
              (assert.is_true (rawequal (. result.parents 1) input)))))))

    (it "records exactly the inputs, by identity and in order, as parents for binary ops"
      (fn []
        (let [cases [[(tensor.matmul a b) a b]
                     [(tensor.add a c) a c]
                     [(tensor.sub a c) a c]
                     [(tensor.mul a c) a c]
                     [(tensor.scale a s) a s]
                     [(tensor.pow a s) a s]]]
          (each [_ entry (ipairs cases)]
            (let [result (. entry 1)
                  lhs (. entry 2)
                  rhs (. entry 3)]
              (assert.equal 2 (length result.parents))
              (assert.is_true (rawequal (. result.parents 1) lhs))
              (assert.is_true (rawequal (. result.parents 2) rhs)))))))

    (it "distinguishes parent identity from value equality"
      (fn []
        (let [result (tensor.add a c)]
          ; Value-equal but distinct object: must NOT be accepted as the parent.
          (assert.is_true (= a a-lookalike))
          (assert.is_false (rawequal (. result.parents 1) a-lookalike))
          (assert.is_true (rawequal (. result.parents 1) a)))))

    (it "only tracks immediate parents, not the full ancestry, for chained ops"
      (fn []
        (let [sum-ab (tensor.add a c)
              chained (tensor.sub sum-ab c)]
          (assert.equal 2 (length chained.parents))
          (assert.is_true (rawequal (. chained.parents 1) sum-ab))
          (assert.is_true (rawequal (. chained.parents 2) c))

          ; The original leaves are reachable through the parent graph, but are
          ; not direct parents of the chained result.
          (assert.is_false (rawequal (. chained.parents 1) a))
          (assert.is_true
            (rawequal (. (. (. chained.parents 1) :parents) 1) a))
          (assert.is_true
            (rawequal (. (. (. chained.parents 1) :parents) 2) c)))))

    (it "gives every non-leaf tensor a nonzero-length parents table"
      (fn []
        (let [results [(tensor.matmul a b) (tensor.transpose a 1 2) (tensor.mul a c)
                       (tensor.add a c) (tensor.sub a c) (tensor.scale a s)
                       (tensor.pow a s) (tensor.mean a) (tensor.sum a) (tensor.relu a)]]
          (each [_ result (ipairs results)]
            (assert.is_true (> (length result.parents) 0))))))))

(describe "tensor operations with scalar operands"
  (fn []
    (local m (tensor.new [2 3] [1 2 3 4 5 6]))
    (local three (tensor.scalar 3))
    (local four (tensor.scalar 4))

    (it "broadcasts a scalar against a matrix, and the result matches the shape of the non-scalar side"
      (fn []
        (let [cases [[(tensor.add m three) [4 5 6 7 8 9]]
                     [(tensor.add three m) [4 5 6 7 8 9]]
                     [(tensor.mul m three) [3 6 9 12 15 18]]
                     [(tensor.mul three m) [3 6 9 12 15 18]]
                     [(tensor.sub m three) [-2 -1 0 1 2 3]]]]
          (each [_ entry (ipairs cases)]
            (let [result (. entry 1)
                  expected (. entry 2)]
              (assert.same m.shape result.shape)
              (assert-data expected result.data))))))

    (it "keeps scalar minus tensor and tensor minus scalar as distinct, non-commutative results"
      (fn []
        (let [left (tensor.sub three m)
              right (tensor.sub m three)]
          ; three - m (scalar on the left) must differ from m - three (scalar on
          ; the right): subtraction is not commutative, broadcasting shouldn't
          ; change that.
          (assert-data [2 1 0 -1 -2 -3] left.data)
          (assert-data [-2 -1 0 1 2 3] right.data)
          ; The two are exact negations of each other.
          (for [i 1 (length m.data)]
            (assert.equal (* -1 (. left.data i)) (. right.data i))))))

    (it "keeps add and mul commutative under broadcasting (identical to the non-scalar case)"
      (fn []
        (assert.equal (tensor.add m three) (tensor.add three m))
        (assert.equal (tensor.mul m three) (tensor.mul three m))))

    (it "combines two scalars using ordinary scalar arithmetic"
      (fn []
        (assert.equal (tensor.scalar 7) (tensor.add three four))
        (assert.equal (tensor.scalar 12) (tensor.mul three four))
        ; four - three = 1, not three - four = -1: direction must be preserved
        ; even when both operands happen to be scalars.
        (assert.equal (tensor.scalar 1) (tensor.sub four three))
        (assert.equal (tensor.scalar -1) (tensor.sub three four))))

    (it "treats scalar add/sub/mul by a scalar identity the same way non-scalar identities work"
      (fn []
        (let [zero (tensor.scalar 0)
              one (tensor.scalar 1)
              cases [[(tensor.add m zero) m]
                     [(tensor.mul m one) m]
                     [(tensor.sub m zero) m]
                     [(tensor.mul m zero) (tensor.new [2 3] [0 0 0 0 0 0])]]]
          (each [_ entry (ipairs cases)]
            (assert.equal (. entry 2) (. entry 1))))))

    (it "leaves a scalar unchanged when transposed"
      (fn []
        (assert.equal three (tensor.transpose three 1 2))))

    (it "reduces a scalar tensor to itself under sum and mean"
      (fn []
        (assert.equal three (tensor.sum three))
        (assert.equal three (tensor.mean three))))

    (it "records both operands as parents, by identity, for scalar-broadcast ops"
      (fn []
        (let [cases [[(tensor.add m three) m three]
                     [(tensor.add three m) three m]
                     [(tensor.sub m three) m three]
                     [(tensor.mul m three) m three]]]
          (each [_ entry (ipairs cases)]
            (let [result (. entry 1)
                  lhs (. entry 2)
                  rhs (. entry 3)]
              (assert.equal 2 (length result.parents))
              (assert.is_true (rawequal (. result.parents 1) lhs))
              (assert.is_true (rawequal (. result.parents 2) rhs)))))))))

(describe "tensor operations with vector operands (broadcasting)"
  (fn []
    ; The only vector broadcast we support is a bare vector {N} against a tensor
    ; whose last dimension is N. The vector is repeated along the (contiguous)
    ; last dimension for every leading position, and the result takes the fuller
    ; tensor shape. There is no NumPy-style size-1 stretching: {N,1} column
    ; vectors and mismatched last dimensions are rejected.
    (local m (tensor.new [2 3] [1 2 3 4 5 6]))
    (local v (tensor.new [3] [10 20 30]))

    (it "broadcasts a vector across the last dimension of a matrix"
      (fn []
        (let [cases [[(tensor.add m v) [11 22 33 14 25 36]]
                     [(tensor.add v m) [11 22 33 14 25 36]]
                     [(tensor.mul m v) [10 40 90 40 100 180]]
                     [(tensor.mul v m) [10 40 90 40 100 180]]
                     [(tensor.sub m v) [-9 -18 -27 -6 -15 -24]]]]
          (each [_ entry (ipairs cases)]
            (let [result (. entry 1)
                  expected (. entry 2)]
              (assert.same m.shape result.shape)
              (assert-data expected result.data))))))

    (it "broadcasts a vector across the last dimension of a 3-D tensor"
      (fn []
        ; {2,2,3} against {3}: the vector repeats for all four leading rows.
        (let [t (tensor.new [2 2 3] [1 2 3 4 5 6 7 8 9 10 11 12])
              result (tensor.add t v)]
          (assert.same [2 2 3] result.shape)
          (assert-data [11 22 33 14 25 36 17 28 39 20 31 42] result.data))))

    (it "keeps vector minus matrix distinct from matrix minus vector (non-commutative)"
      (fn []
        (let [left (tensor.sub v m)
              right (tensor.sub m v)]
          ; v - m is the exact negation of m - v: broadcasting must not quietly
          ; symmetrize a non-commutative op.
          (assert-data [9 18 27 6 15 24] left.data)
          (for [i 1 (length m.data)]
            (assert.equal (* -1 (. right.data i)) (. left.data i))))))

    (it "rejects broadcasts outside the scalar / last-dimension-vector forms"
      (fn []
        ; last dimension mismatch, and NumPy-style size-1 stretching, are errors.
        (assert.has_error
          (fn [] (tensor.add m (tensor.new [2] [1 2]))))
        (assert.has_error
          (fn [] (tensor.add m (tensor.new [2 1] [1 2]))))
        (assert.has_error
          (fn [] (tensor.add m (tensor.new [2 2] [1 2 3 4]))))))

    (it "records both operands as parents, by identity, for vector-broadcast ops"
      (fn []
        (let [cases [[(tensor.add m v) m v]
                     [(tensor.add v m) v m]
                     [(tensor.mul m v) m v]
                     [(tensor.sub m v) m v]]]
          (each [_ entry (ipairs cases)]
            (let [result (. entry 1)
                  lhs (. entry 2)
                  rhs (. entry 3)]
              (assert.equal 2 (length result.parents))
              (assert.is_true (rawequal (. result.parents 1) lhs))
              (assert.is_true (rawequal (. result.parents 2) rhs)))))))))

(describe "tensor softmax"
  (fn []
    (local slice-sum
      (fn [data from to]
        (faccumulate [total 0 i from to]
          (+ total (. data i)))))

    (it "turns a vector into a probability distribution that sums to 1"
      (fn []
        (let [v (tensor.new [3] [1 2 3])
              p (tensor.softmax v 1)]
          (assert.same [3] p.shape)
          (assert.near 1 (slice-sum p.data 1 3) 0.000001)
          (each [_ x (ipairs p.data)]
            (assert.is_true (and (> x 0) (< x 1)))))))

    (it "matches the closed-form softmax for a small vector"
      (fn []
        ; softmax({1,2,3}) = exp(x) / sum(exp(x))
        (let [v (tensor.new [3] [1 2 3])
              denom (+ (+ (math.exp 1) (math.exp 2)) (math.exp 3))
              expected [(/ (math.exp 1) denom)
                        (/ (math.exp 2) denom)
                        (/ (math.exp 3) denom)]
              p (tensor.softmax v 1)]
          (for [i 1 3]
            (assert.near (. expected i) (. p.data i) 0.000001)))))

    (it "preserves the order of the inputs (monotonic in the logits)"
      (fn []
        (let [p (tensor.softmax (tensor.new [4] [-1 0 2 5]) 1)]
          (for [i 2 4]
            (assert.is_true (> (. p.data i) (. p.data (- i 1))))))))

    (it "is invariant to adding a constant to every logit (shift invariance)"
      (fn []
        (let [base (tensor.softmax (tensor.new [3] [1 2 3]) 1)
              shifted (tensor.softmax (tensor.new [3] [1001 1002 1003]) 1)]
          (for [i 1 3]
            (assert.near (. base.data i) (. shifted.data i) 0.000001)))))

    (it "stays numerically stable for large logits (no overflow to nan/inf)"
      (fn []
        ; max-subtraction inside softmax must keep exp() finite even here.
        (let [p (tensor.softmax (tensor.new [3] [1000 1001 1002]) 1)]
          (assert.near 1 (slice-sum p.data 1 3) 0.000001)
          (each [_ x (ipairs p.data)]
            (assert.is_true (and (= x x) (< x math.huge)))))))

    (it "normalizes each slice independently along the chosen axis"
      (fn []
        (let [m (tensor.new [2 3] [1 2 3 4 5 6])
              ; axis 2: every row is its own distribution
              rows (tensor.softmax m 2)
              ; axis 1: every column is its own distribution
              cols (tensor.softmax m 1)]
          (assert.near 1 (slice-sum rows.data 1 3) 0.000001)
          (assert.near 1 (slice-sum rows.data 4 6) 0.000001)
          (assert.near 1 (+ (. cols.data 1) (. cols.data 4)) 0.000001)
          (assert.near 1 (+ (. cols.data 2) (. cols.data 5)) 0.000001)
          (assert.near 1 (+ (. cols.data 3) (. cols.data 6)) 0.000001))))

    (it "accepts a negative axis, resolving it from the end"
      (fn []
        (let [m (tensor.new [2 3] [1 2 3 4 5 6])]
          (assert.equal (tensor.softmax m 2) (tensor.softmax m -1)))))

    (it "sharpens toward the argmax as temperature drops and flattens as it rises"
      (fn []
        (let [logits (tensor.new [3] [1 2 3])
              cold (tensor.softmax logits 1 0.1)
              hot (tensor.softmax logits 1 100)]
          ; low temperature concentrates mass on the largest logit
          (assert.is_true (> (. cold.data 3) 0.99))
          ; high temperature pushes toward the uniform 1/3
          (for [i 1 3]
            (assert.near (/ 1 3) (. hot.data i) 0.01)))))

    (it "rejects a non-positive temperature"
      (fn []
        (let [logits (tensor.new [3] [1 2 3])]
          (assert.has_error (fn [] (tensor.softmax logits 1 0)))
          (assert.has_error (fn [] (tensor.softmax logits 1 -1))))))

    (it "records its single input as parent, by identity"
      (fn []
        (let [v (tensor.new [3] [1 2 3])
              p (tensor.softmax v 1)]
          (assert.equal 1 (length p.parents))
          (assert.is_true (rawequal (. p.parents 1) v)))))))

(describe "tensor argmax"
  (fn []
    (it "returns the index of the largest value in a vector, reducing to a scalar shape"
      (fn []
        (let [v (tensor.new [4] [-1 3 2 0])
              idx (tensor.argmax v 1)]
          (assert.same [] idx.shape)
          (assert-data [2] idx.data))))

    (it "breaks ties toward the first occurrence"
      (fn []
        (let [v (tensor.new [4] [5 5 1 5])]
          (assert-data [1] (. (tensor.argmax v 1) :data))))

    (it "reduces each row independently along the last axis"
      (fn []
        (let [m (tensor.new [2 3] [1 2 3 6 5 4])
              idx (tensor.argmax m 2)]
          (assert.same [2] idx.shape)
          (assert-data [3 1] idx.data))))

    (it "reduces each column independently along the first axis"
      (fn []
        (let [m (tensor.new [2 3] [1 9 3 4 5 6])
              idx (tensor.argmax m 1)]
          (assert.same [3] idx.shape)
          ; columns: {1,4}->2, {9,5}->1, {3,6}->2
          (assert-data [2 1 2] idx.data))))

    (it "accepts a negative axis, resolving it from the end"
      (fn []
        (let [m (tensor.new [2 3] [1 2 3 4 5 6])]
          (assert.equal (tensor.argmax m 2) (tensor.argmax m -1)))))

    (it "reduces the chosen axis of a 3-D tensor"
      (fn []
        (let [t (tensor.new [2 2 2] [2 1 3 4 8 7 5 6])
              idx (tensor.argmax t 3)]
          (assert.same [2 2] idx.shape)
          ; pairs along the last axis: {2,1}->1, {3,4}->2, {8,7}->1, {5,6}->2
          (assert-data [1 2 1 2] idx.data))))

    (it "does not participate in autograd (no parents, no gradient)"
      (fn []
        (let [v (tensor.new [3] [1 2 3])
              idx (tensor.argmax v 1)]
          (assert.same [] idx.parents)
          (assert.is_nil idx.gradient))))

    (it "rejects an axis outside the tensor's rank"
      (fn []
        (let [m (tensor.new [2 3] [1 2 3 4 5 6])]
          (assert.has_error (fn [] (tensor.argmax m 3)))
          (assert.has_error (fn [] (tensor.argmax m 0)))))))))

(describe "tensor cross_entropy"
  (fn []
    (it "reduces to a scalar loss"
      (fn []
        (let [logits (tensor.new [1 3] [0 0 0])
              target (tensor.new [1 3] [1 0 0])
              loss ((. tensor :cross-entropy) logits target 2)]
          (assert.same [] loss.shape))))

    (it "equals -log(prob of the true class) for a one-hot target on a single slice"
      (fn []
        ; uniform logits -> each class prob 1/3 -> loss = -log(1/3) = log 3
        (let [logits (tensor.new [1 3] [0 0 0])
              target (tensor.new [1 3] [1 0 0])]
          (assert.near (math.log 3)
                       (. (. ((. tensor :cross-entropy) logits target 2) :data) 1)
                       0.000001))))

    (it "matches -sum(target * log_softmax) for arbitrary logits"
      (fn []
        (let [logits (tensor.new [1 3] [2 1 0])
              target (tensor.new [1 3] [1 0 0])
              denom (+ (+ (math.exp 2) (math.exp 1)) (math.exp 0))
              expected (- (math.log (/ (math.exp 2) denom)))]
          (assert.near expected
                       (. (. ((. tensor :cross-entropy) logits target 2) :data) 1)
                       0.000001))))

    (it "shrinks as the logit for the true class grows (more confidence, less loss)"
      (fn []
        (let [target (tensor.new [1 3] [1 0 0])
              unsure ((. tensor :cross-entropy) (tensor.new [1 3] [0 0 0]) target 2)
              confident ((. tensor :cross-entropy) (tensor.new [1 3] [5 0 0]) target 2)]
          (assert.is_true (< (. confident.data 1) (. unsure.data 1))))))

    (it "averages the loss across independent slices"
      (fn []
        ; Two identical rows -> the mean loss equals a single row's loss.
        (let [logits (tensor.new [2 3] [0 0 0 0 0 0])
              target (tensor.new [2 3] [1 0 0 1 0 0])]
          (assert.near (math.log 3)
                       (. (. ((. tensor :cross-entropy) logits target 2) :data) 1)
                       0.000001))))

    (it "rejects a target whose shape does not match the logits"
      (fn []
        (let [logits (tensor.new [1 3] [0 0 0])
              target (tensor.new [1 2] [1 0])]
          (assert.has_error (fn [] ((. tensor :cross-entropy) logits target 2))))))

    (it "records only the logits as parent, by identity (not the target)"
      (fn []
        (let [logits (tensor.new [1 3] [0 0 0])
              target (tensor.new [1 3] [1 0 0])
              loss ((. tensor :cross-entropy) logits target 2)]
          (assert.equal 1 (length loss.parents))
          (assert.is_true (rawequal (. loss.parents 1) logits)))))))

(describe "tensor gather"
  (fn []
    ; gather treats the 2D tensor as a table of rows (an embedding table
    ; {vocab, embed_size}) and selects one row per entry of the 1D index
    ; tensor, stacking them into a {num_indices, embed_size} result.
    (local embed (tensor.new [3 2] [10 20 30 40 50 60]))

    (it "selects the indexed rows, producing {num_indices, embed_size}"
      (fn []
        (let [out (tensor.gather embed (tensor.new [2] [3 1]))]
          (assert.same [2 2] out.shape)
          ; row 3 then row 1
          (assert-data [50 60 10 20] out.data))))

    (it "preserves the order of the index tensor and repeats rows on demand"
      (fn []
        (let [out (tensor.gather embed (tensor.new [4] [2 2 1 3]))]
          (assert.same [4 2] out.shape)
          (assert-data [30 40 30 40 10 20 50 60] out.data))))

    (it "records only the source table as parent, by identity (not the index tensor)"
      (fn []
        (let [idx (tensor.new [2] [3 1])
              out (tensor.gather embed idx)]
          (assert.equal 1 (length out.parents))
          (assert.is_true (rawequal (. out.parents 1) embed))
          (assert.is_false (rawequal (. out.parents 1) idx)))))

    (it "keeps the index tensor's shape and appends embed_size for N-D indices"
      (fn []
        ; a {2,2} index tensor gathers 4 rows and yields a {2,2,embed_size} result
        (let [out (tensor.gather embed (tensor.new [2 2] [1 2 3 1]))]
          (assert.same [2 2 2] out.shape)
          (assert-data [10 20 30 40 50 60 10 20] out.data))))

    (it "rejects gathering from a non-2D table"
      (fn []
        (assert.has_error
          (fn [] (tensor.gather (tensor.new [3] [1 2 3])
                                 (tensor.new [1] [1]))))))

    (it "scatters the upstream gradient back onto the selected rows"
      (fn []
        (let [out (tensor.gather embed (tensor.new [2] [3 1]))]
          (set out.gradient (tensor.new out.shape [1 2 3 4]))
          (tensor.backward-step! out)
          ; row 3 gets {1,2}, row 1 gets {3,4}, row 2 (never selected) stays zero
          (assert.same [3 2] embed.gradient.shape)
          (assert-data [3 4 0 0 1 2] embed.gradient.data))))

    (it "sums the upstream gradient when the same row is gathered more than once"
      (fn []
        (let [table (tensor.new [3 2] [10 20 30 40 50 60])
              out (tensor.gather table (tensor.new [2] [1 1]))]
          (set out.gradient (tensor.new out.shape [1 2 3 4]))
          (tensor.backward-step! out)
          ; row 1 is selected twice, so its gradient is the sum {1+3, 2+4}
          (assert-data [4 6 0 0 0 0] table.gradient.data))))

    (it "accumulates onto a pre-existing table gradient instead of overwriting it"
      (fn []
        (let [table (tensor.new [3 2] [10 20 30 40 50 60])]
          (set table.gradient (tensor.new table.shape [100 100 100 100 100 100]))
          (let [out (tensor.gather table (tensor.new [1] [2]))]
            (set out.gradient (tensor.new out.shape [1 2]))
            (tensor.backward-step! out)
            (assert-data [100 100 101 102 100 100] table.gradient.data)))))))

(describe "tensor backward propagation"
  (fn []
    (local clone-data
      (fn [data]
        (icollect [_ value (ipairs data)] value)))

    (local perturbed
      (fn [t idx delta]
        (tensor.new t.shape
          (icollect [i value (ipairs t.data)]
            (if (= i idx) (+ value delta) value)))))

    ; Central-difference directional derivative of forward-fn's output with
    ; respect to a single element of one input, dotted against seed-grad
    ; (dL/dOut). This is the numerical stand-in for what an analytic
    ; backward pass should compute, so `backward-step!` can be checked against it
    ; without ever encoding the calculus by hand in the test.
    (local numeric-partial
      (fn [forward-fn inputs input-idx elem-idx seed-grad eps]
        (let [step (finite-difference-step (. inputs input-idx) eps)
              plus
                (icollect [i t (ipairs inputs)]
                  (if (= i input-idx)
                    (perturbed t elem-idx step)
                    t))
              minus
                (icollect [i t (ipairs inputs)]
                  (if (= i input-idx)
                    (perturbed t elem-idx (- step))
                    t))
              out-plus (forward-fn plus)
              out-minus (forward-fn minus)
              diff
                (faccumulate [total 0 k 1 (length out-plus.data)]
                  (+ total
                     (* (. seed-grad.data k)
                        (- (. out-plus.data k) (. out-minus.data k)))))]
          (/ diff (* 2 step)))))

    ; Runs forward-fn on fresh copies of `raw-inputs` (so each op under test
    ; starts from a clean, zero-initialized `.gradient`), seeds the output
    ; gradient, calls `backward-step!`, then checks every analytic parent
    ; gradient against its numerical estimate. This validates whatever
    ; chain-rule formula gets implemented, rather than hand-encoding the
    ; formula here too (and possibly making the same mistake twice).
    (local assert-gradcheck
      (fn [forward-fn raw-inputs seed-data eps tolerance]
        (let [inputs
                (icollect [_ t (ipairs raw-inputs)]
                  (tensor.new t.shape (clone-data t.data)))
              out (forward-fn inputs)
              seed-grad (tensor.new out.shape seed-data)]
          (set out.gradient seed-grad)
          (tensor.backward-step! out)
          (each [i t (ipairs inputs)]
            (for [elem 1 (length t.data)]
              (let [numeric (numeric-partial forward-fn inputs i elem seed-grad eps)
                    analytic (. t.gradient.data elem)]
                (assert.near numeric analytic tolerance)))))))

    (local a (tensor.new [2 3] [1 2 3 4 5 6]))
    (local c (tensor.new [2 3] [6 5 4 3 2 1]))
    (local b34 (tensor.new [3 4] [7 8 9 10 11 12 13 14 15 16 17 18]))
    (local s (tensor.scalar 2))

    (describe "invariants that should hold for every op"
      (fn []
        ; Gradient allocation is lazy: a fresh tensor's gradient is nil, and the
        ; zero buffer is materialized on demand the first time a gradient flows into
        ; it. This avoids allocating buffers for forward-only / eval passes.
        (it "defers gradient allocation: a fresh tensor's gradient is nil"
          (fn []
            (each [_ t (ipairs [a (tensor.add a c) (tensor.sum a)
                                (tensor.scale a s) (tensor.pow a s)])]
              (assert.is_nil t.gradient))))

        (it "allocates a correctly-shaped zero gradient on first accumulation"
          (fn []
            (let [x (tensor.new [2 3] [1 2 3 4 5 6])
                  out (tensor.add x (tensor.new [2 3] [1 1 1 1 1 1]))]
              (assert.is_nil x.gradient) ; still deferred before backward
              (set out.gradient (tensor.new out.shape [1 1 1 1 1 1]))
              (tensor.backward-step! out)
              (assert.is_not_nil x.gradient) ; materialized by accumulation
              (assert.same x.shape x.gradient.shape)
              (assert.equal (length x.data) (length x.gradient.data)))))

        (it "gives every non-leaf tensor an op tag that drives its backward rule"
          (fn []
            ; Ops-as-data: instead of a per-node `_backward` closure, a non-leaf
            ; carries an `:op` tag that backward-step! dispatches on (leaves are :noop).
            (let [results [(tensor.matmul a b34) (tensor.transpose a 1 2)
                           (tensor.mul a c) (tensor.add a c) (tensor.sub a c)
                           (tensor.scale a s) (tensor.pow a s) (tensor.mean a)
                           (tensor.sum a) (tensor.relu a)]]
              (each [_ result (ipairs results)]
                (assert.equal "string" (type result.op))
                (assert.not_equal "noop" result.op)))))

        (it "does not mutate data or shape when _backward runs"
          (fn []
            (let [x (tensor.new [2 3] [1 2 3 4 5 6])
                  y (tensor.new [2 3] [6 5 4 3 2 1])
                  out (tensor.add x y)]
              (set out.gradient (tensor.new out.shape [1 1 1 1 1 1]))
              (tensor.backward-step! out)
              (assert-data [1 2 3 4 5 6] x.data)
              (assert-data [6 5 4 3 2 1] y.data)
              (assert.same [2 3] x.shape))))

        (it "accumulates into existing parent gradient instead of overwriting it"
          (fn []
            (let [x (tensor.new [2 3] [1 2 3 4 5 6])
                  y (tensor.new [2 3] [6 5 4 3 2 1])
                  out (tensor.add x y)]
              (set x.gradient (tensor.new x.shape [10 10 10 10 10 10]))
              (set out.gradient (tensor.new out.shape [1 1 1 1 1 1]))
              (tensor.backward-step! out)
              (assert-data [11 11 11 11 11 11] x.gradient.data))))

        (it "sums contributions when a tensor is used as more than one input to the same op"
          (fn []
            (let [x (tensor.new [2 3] [1 2 3 4 5 6])
                  out (tensor.add x x)]
              (set out.gradient (tensor.new out.shape [1 2 3 4 5 6]))
              (tensor.backward-step! out)
              ; d(x+x)/dx = 2, applied once per occurrence, so both contributions
              ; land in the same x.gradient and should sum to 2 * out.gradient.
              (assert-data [2 4 6 8 10 12] x.gradient.data))))

        ; matmul and transpose write same-shaped gradients, so like every other
        ; op they must add onto whatever a parent already holds, not clobber it.
        ; (Overwriting only "works" when a parent has exactly one consumer and
        ; silently drops a contribution the moment it feeds two paths.)
        (it "accumulates into existing parent gradients for matmul instead of overwriting"
          (fn []
            (let [x (tensor.new [2 3] [1 2 3 4 5 6])
                  w (tensor.new [3 2] [1 0 0 1 1 1])
                  out (tensor.matmul x w)]
              (set x.gradient (tensor.new x.shape [100 100 100 100 100 100]))
              (set w.gradient (tensor.new w.shape [100 100 100 100 100 100]))
              (set out.gradient (tensor.new out.shape [1 1 1 1]))
              (tensor.backward-step! out)
              ; contribution to x is out.grad · wᵀ = { 1, 1, 2, 1, 1, 2 }
              (assert-data [101 101 102 101 101 102] x.gradient.data)
              ; contribution to w is xᵀ · out.grad = { 5, 5, 7, 7, 9, 9 }
              (assert-data [105 105 107 107 109 109] w.gradient.data))))

        (it "accumulates into an existing parent gradient for transpose instead of overwriting"
          (fn []
            (let [y (tensor.new [2 3] [1 2 3 4 5 6])
                  t (tensor.transpose y 1 2)]
              (set y.gradient (tensor.new y.shape [100 100 100 100 100 100]))
              (set t.gradient (tensor.new t.shape [1 2 3 4 5 6]))
              (tensor.backward-step! t)
              ; contribution to y is transpose(t.grad) = { 1, 3, 5, 2, 4, 6 }
              (assert-data [101 103 105 102 104 106] y.gradient.data))))))

    (describe "gradient correctness (numerical gradient check per op)"
      (fn []
        (it "add"
          (fn []
            (assert-gradcheck
              (fn [ts] (tensor.add (. ts 1) (. ts 2)))
              [a c] [1 -2 0.5 3 -1 2] 0.0001 0.001)))

        (it "sub"
          (fn []
            (assert-gradcheck
              (fn [ts] (tensor.sub (. ts 1) (. ts 2)))
              [a c] [1 -2 0.5 3 -1 2] 0.0001 0.001)))

        (it "mul (elementwise)"
          (fn []
            (assert-gradcheck
              (fn [ts] (tensor.mul (. ts 1) (. ts 2)))
              [a c] [1 -2 0.5 3 -1 2] 0.0001 0.001)))

        (it "scale"
          (fn []
            (assert-gradcheck
              (fn [ts] (tensor.scale (. ts 1) (. ts 2)))
              [a s] [1 -2 0.5 3 -1 2] 0.0001 0.001)))

        (it "transpose"
          (fn []
            (assert-gradcheck
              (fn [ts] (tensor.transpose (. ts 1) 1 2))
              [a] [1 -2 0.5 3 -1 2] 0.0001 0.001)))

        (it "pow"
          (fn []
            (assert-gradcheck
              (fn [ts] (tensor.pow (. ts 1) (. ts 2)))
              [a s] [1 -2 0.5 3 -1 2] 0.0001 0.01)))

        (it "relu"
          (fn []
            ; Mixed-sign input kept well clear of 0 so the central-difference
            ; estimate never straddles relu's non-differentiable kink at 0, where
            ; the numerical and analytic gradients would legitimately disagree.
            (let [mixed (tensor.new [2 3] [-3 -1 0.5 2 -4 5])]
              (assert-gradcheck
                (fn [ts] (tensor.relu (. ts 1)))
                [mixed] [1 -2 0.5 3 -1 2] 0.0001 0.001))))

        ; The gradient flows through only where the input was positive: relu is
        ; flat (slope 0) on negatives and passes the upstream gradient through
        ; unchanged on positives.
        (it "gates the gradient by the sign of the input"
          (fn []
            (let [x (tensor.new [1 4] [-2 -0.5 1 3])
                  out (tensor.relu x)]
              (set out.gradient (tensor.new out.shape [5 6 7 8]))
              (tensor.backward-step! out)
              (assert-data [0 0 7 8] x.gradient.data))))

        ; The squared-error loss (mean((pred - target)^2)) routinely raises a
        ; negative base to a constant power. The base gradient d(x^2)/dx = 2x is
        ; perfectly well-defined there; the exponent gradient involves log(x),
        ; which is not, so it must be kept finite rather than poisoning the graph
        ; with NaN (a scalar exponent's own gradient is discarded in practice).
        (it "keeps pow gradients finite for negative bases"
          (fn []
            (let [x (tensor.new [1 3] [-3 -1 2])
                  two (tensor.scalar 2)
                  out (tensor.pow x two)]
              (set out.gradient (tensor.new out.shape [1 1 1]))
              (tensor.backward-step! out)
              ; base gradient is 2x, valid regardless of sign
              (assert.near -6 (. x.gradient.data 1) 0.000000001)
              (assert.near -2 (. x.gradient.data 2) 0.000000001)
              (assert.near 4 (. x.gradient.data 3) 0.000000001)
              ; exponent gradient must not be NaN (NaN is the only value != itself)
              (each [_ value (ipairs two.gradient.data)]
                (assert.is_true (= value value))))))

        (it "matmul"
          (fn []
            (assert-gradcheck
              (fn [ts] (tensor.matmul (. ts 1) (. ts 2)))
              [a b34] [1 -2 0.5 3 -1 2 0.25 4] 0.0001 0.01)))

        (it "sum"
          (fn []
            (assert-gradcheck
              (fn [ts] (tensor.sum (. ts 1)))
              [a] [1.7] 0.0001 0.001)))

        (it "mean"
          (fn []
            (assert-gradcheck
              (fn [ts] (tensor.mean (. ts 1)))
              [a] [1.7] 0.0001 0.001)))

        ; softmax and cross_entropy leave the target/actual out of the graph, so
        ; the gradcheck only differentiates the logits (the sole parent).
        (it "softmax (single slice)"
          (fn []
            (let [logits (tensor.new [1 3] [0.5 -1 2])]
              (assert-gradcheck
                (fn [ts] (tensor.softmax (. ts 1) 2))
                [logits] [1 -2 0.5] 0.0001 0.001))))

        (it "softmax (multiple slices along the axis)"
          (fn []
            (assert-gradcheck
              (fn [ts] (tensor.softmax (. ts 1) 2))
              [a] [1 -2 0.5 3 -1 2] 0.0001 0.001)))

        (it "softmax (non-default temperature)"
          (fn []
            (let [logits (tensor.new [1 3] [0.5 -1 2])]
              (assert-gradcheck
                (fn [ts] (tensor.softmax (. ts 1) 2 0.5))
                [logits] [1 -2 0.5] 0.0001 0.001))))

        (it "cross_entropy (single slice)"
          (fn []
            (let [target (tensor.new [1 3] [1 0 0])]
              (assert-gradcheck
                (fn [ts]
                  ((. tensor :cross-entropy) (. ts 1) target 2))
                [(tensor.new [1 3] [0.5 -1 2])] [1] 0.0001 0.001))))

        (it "cross_entropy (multiple slices, soft targets)"
          (fn []
            ; Each row's target is a valid distribution summing to 1, which is what
            ; the analytic gradient (prob - target)/slices assumes.
            (let [target (tensor.new [2 3] [1 0 0 0.2 0.3 0.5])]
              (assert-gradcheck
                (fn [ts]
                  ((. tensor :cross-entropy) (. ts 1) target 2))
                [a] [1.3] 0.0001 0.001))))))

    (describe "scalar to scalar gradient"
      (fn []
        (it "add"
          (fn []
            (assert-gradcheck
              (fn [ts] (tensor.add (. ts 1) (. ts 2)))
              [s (tensor.scalar 5)] [1.3] 0.0001 0.001)))

        (it "sub"
          (fn []
            (assert-gradcheck
              (fn [ts] (tensor.sub (. ts 1) (. ts 2)))
              [s (tensor.scalar 5)] [1.3] 0.0001 0.001)))

        (it "mul"
          (fn []
            (assert-gradcheck
              (fn [ts] (tensor.mul (. ts 1) (. ts 2)))
              [s (tensor.scalar 5)] [1.3] 0.0001 0.001)))))

    (describe "broadcasting"
      (fn []
        (describe "scalar to matrix gradient"
          (fn []
            (it "add"
              (fn []
                (assert-gradcheck
                  (fn [ts] (tensor.add (. ts 1) (. ts 2)))
                  [s c] [1 -2 0.5 3 -1 2] 0.0001 0.001)))
            (it "sub"
              (fn []
                (assert-gradcheck
                  (fn [ts] (tensor.sub (. ts 1) (. ts 2)))
                  [s c] [1 -2 0.5 3 -1 2] 0.0001 0.001)))
            (it "mul"
              (fn []
                (assert-gradcheck
                  (fn [ts] (tensor.mul (. ts 1) (. ts 2)))
                  [s c] [1 -2 0.5 3 -1 2] 0.0001 0.001)))))

        (describe "matrix to scalar gradient"
          (fn []
            (it "add"
              (fn []
                (assert-gradcheck
                  (fn [ts] (tensor.add (. ts 1) (. ts 2)))
                  [c s] [1 -2 0.5 3 -1 2] 0.0001 0.001)))
            (it "sub"
              (fn []
                (assert-gradcheck
                  (fn [ts] (tensor.sub (. ts 1) (. ts 2)))
                  [c s] [1 -2 0.5 3 -1 2] 0.0001 0.001)))
            (it "mul"
              (fn []
                (assert-gradcheck
                  (fn [ts] (tensor.mul (. ts 1) (. ts 2)))
                  [c s] [1 -2 0.5 3 -1 2] 0.0001 0.001)))))

        ; Broadcasting a scalar out to a matrix-shaped result must not leak
        ; that matrix shape back into the scalar's own gradient: a parent's
        ; gradient should always match that parent's own shape, not the
        ; output's, regardless of which op produced the output.
        (describe "keeps a scalar operand's gradient scalar-shaped after"
          (fn []
            (each [_ op (ipairs ["add" "sub" "mul"])]
              (it (.. op " with a matrix")
                (fn []
                  (let [scalar (tensor.scalar 3)
                        matrix (tensor.new [2 3] [1 2 3 4 5 6])
                        out ((. tensor op) scalar matrix)]
                    (set out.gradient (tensor.new out.shape [1 1 1 1 1 1]))
                    (tensor.backward-step! out)
                    (assert.same [] scalar.gradient.shape)
                    (assert.equal 1 (length scalar.gradient.data))))))))

        (describe "vector to matrix gradient"
          (fn []
            ; A vector {N} is repeated along the last dimension of the {m,N} matrix.
            ; The backward pass must "unbroadcast": each vector element's gradient
            ; is the sum of the upstream gradients at every position it was copied to,
            ; while the matrix gradient passes straight through. The gradcheck
            ; confirms both operands' analytic gradients match the numerical ones.
            (local matrix (tensor.new [2 3] [1 2 3 4 5 6]))
            (local v (tensor.new [3] [10 20 30]))
            (local seed [1 -2 0.5 3 -1 2])
            (it "add (matrix + vector)"
              (fn []
                (assert-gradcheck
                  (fn [ts] (tensor.add (. ts 1) (. ts 2)))
                  [matrix v] seed 0.0001 0.001)))
            (it "add (vector + matrix)"
              (fn []
                (assert-gradcheck
                  (fn [ts] (tensor.add (. ts 1) (. ts 2)))
                  [v matrix] seed 0.0001 0.001)))
            (it "sub (matrix - vector)"
              (fn []
                (assert-gradcheck
                  (fn [ts] (tensor.sub (. ts 1) (. ts 2)))
                  [matrix v] seed 0.0001 0.001)))
            (it "mul (matrix * vector)"
              (fn []
                (assert-gradcheck
                  (fn [ts] (tensor.mul (. ts 1) (. ts 2)))
                  [matrix v] seed 0.0001 0.001)))
            (it "mul (vector * matrix)"
              (fn []
                (assert-gradcheck
                  (fn [ts] (tensor.mul (. ts 1) (. ts 2)))
                  [v matrix] seed 0.0001 0.001)))))

        ; Broadcasting a vector out to a matrix-shaped result must not leak
        ; the matrix shape back into the vector's own gradient: the vector's
        ; gradient must keep the vector's shape, holding the summed contributions
        ; from every row it was broadcast across.
        (describe "keeps a vector operand's gradient vector-shaped and unbroadcast"
          (fn []
            (it "sums a vector's gradient down the broadcast rows"
              (fn []
                (let [matrix (tensor.new [2 3] [1 2 3 4 5 6])
                      v (tensor.new [3] [10 20 30])
                      out (tensor.add matrix v)]
                  (set out.gradient (tensor.new out.shape [1 2 3 4 5 6]))
                  (tensor.backward-step! out)
                  ; v was copied onto both rows, so its gradient is the column-wise
                  ; sum of the upstream: { 1+4, 2+5, 3+6 }.
                  (assert.same [3] v.gradient.shape)
                  (assert-data [5 7 9] v.gradient.data)
                  ; the matrix sees the upstream gradient unchanged
                  (assert-data [1 2 3 4 5 6] matrix.gradient.data))))))))))

(describe "Tensor:backward"
  (fn []
    (local clone-data
      (fn [data]
        (icollect [_ value (ipairs data)] value)))

    (local perturbed
      (fn [t idx delta]
        (tensor.new t.shape
          (icollect [i value (ipairs t.data)]
            (if (= i idx) (+ value delta) value)))))

    ; Builds a fresh, independent copy of raw-inputs (so backwards() never
    ; mutates the values under test), runs forward-fn to build the graph,
    ; calls backwards() on the (scalar) result, and checks every leaf's
    ; accumulated gradient against a central-difference numerical estimate
    ; obtained by re-running forward-fn from scratch on perturbed inputs.
    ; This validates full multi-op graph traversal (ordering, accumulation,
    ; shape handling) without hand-encoding any calculus here.
    (local assert-full-gradcheck
      (fn [forward-fn raw-inputs eps tolerance]
        (let [inputs
                (icollect [_ t (ipairs raw-inputs)]
                  (tensor.new t.shape (clone-data t.data)))
              out (forward-fn inputs)]
          (tensor.backward! out)
          (each [i t (ipairs inputs)]
            (let [step (finite-difference-step t eps)]
              (for [elem 1 (length t.data)]
                (let [plus
                        (icollect [j raw (ipairs raw-inputs)]
                          (if (= j i)
                            (perturbed (. inputs i) elem step)
                            (tensor.new raw.shape (clone-data raw.data))))
                      minus
                        (icollect [j raw (ipairs raw-inputs)]
                          (if (= j i)
                            (perturbed (. inputs i) elem (- step))
                            (tensor.new raw.shape (clone-data raw.data))))
                      plus-out (forward-fn plus)
                      minus-out (forward-fn minus)
                      numeric
                        (/ (- (. plus-out.data 1) (. minus-out.data 1))
                           (* 2 step))]
                  (assert.near numeric (. t.gradient.data elem) tolerance))))))))

    (it "errors when called on a non-scalar tensor"
      (fn []
        (let [m (tensor.new [2 3] [1 2 3 4 5 6])]
          (assert.has_error (fn [] (tensor.backward! m))))))

    (it "does not error when called on a scalar tensor"
      (fn []
        (let [s (tensor.scalar 5)]
          (assert.has_no.errors (fn [] (tensor.backward! s))))))

    (it "seeds its own gradient to 1"
      (fn []
        (let [x (tensor.scalar 3)
              y (tensor.scalar 4)
              loss (tensor.add x y)]
          (tensor.backward! loss)
          (assert.equal 1 (. loss.gradient.data 1)))))

    (it "leaves the gradient of a leaf with no path to the loss at zero"
      (fn []
        (let [x (tensor.scalar 3)
              y (tensor.scalar 4)
              unrelated (tensor.scalar 100)
              loss (tensor.add x y)]
          (tensor.backward! loss)
          ; A leaf off the backward path receives no contribution. With lazy gradients
          ; no buffer is ever allocated for it, so nil IS the "still zero" state.
          (assert.is_nil unrelated.gradient))))

    (it "propagates correctly through a diamond dependency (one leaf feeding two paths that reconverge)"
      (fn []
        (assert-full-gradcheck
          (fn [ts]
            (let [x (. ts 1)
                  aa (tensor.mul x x)
                  b (tensor.add x (tensor.scalar 5))]
              (tensor.add aa b)))
          [(tensor.scalar 3)] 0.0001 0.001)))

    (it "propagates correctly through a diamond where the shared node has further parents of its own"
      (fn []
        ; w -> q -> {p1, p2} -> loss. A correct traversal must fully accumulate
        ; q's gradient (from both p1 and p2) before using it to propagate into
        ; w; visiting q's backward too early would silently drop a contribution.
        (assert-full-gradcheck
          (fn [ts]
            (let [w (. ts 1)
                  q (tensor.mul w w)
                  p1 (tensor.add q (tensor.scalar 2))
                  p2 (tensor.mul q (tensor.scalar 3))]
              (tensor.add p1 p2)))
          [(tensor.scalar 2)] 0.0001 0.001)))

    (it "propagates correctly through a matrix-valued diamond reduced to a scalar loss"
      (fn []
        (assert-full-gradcheck
          (fn [ts]
            (let [x (. ts 1)
                  aa (tensor.mul x x)
                  b (tensor.sub x (tensor.scalar 1))]
              (tensor.sum (tensor.add aa b))))
          [(tensor.new [2 3] [1 2 3 4 5 6])] 0.0001 0.001)))

    (it "propagates correctly through a long chain of ops (deep graph traversal)"
      (fn []
        (assert-full-gradcheck
          (fn [ts]
            (let [x (. ts 1)
                  y (faccumulate [current x _ 1 8]
                      (tensor.mul (tensor.add current x) (tensor.scalar 0.5)))]
              y))
          [(tensor.scalar 1.5)] 0.0001 0.01)))

    (it "propagates correctly when a tensor is reused across more than two consumers"
      (fn []
        (assert-full-gradcheck
          (fn [ts]
            (let [x (. ts 1)
                  aa (tensor.mul x (tensor.scalar 2))
                  b (tensor.mul x (tensor.scalar 3))
                  c (tensor.mul x (tensor.scalar 5))]
              (tensor.add (tensor.add aa b) c)))
          [(tensor.scalar 4)] 0.0001 0.001)))

    (it "propagates correctly through a mix of matmul and elementwise ops in one graph"
      (fn []
        (assert-full-gradcheck
          (fn [ts]
            (let [x (. ts 1)
                  w (. ts 2)
                  h (tensor.matmul x w)
                  y (tensor.mul h h)]
              (tensor.sum y)))
          [(tensor.new [2 3] [1 2 3 4 5 6])
           (tensor.new [3 2] [1 0 0 1 1 1])]
          0.0001 0.01)))

    (it "sums matmul and elementwise contributions when one leaf feeds both paths"
      (fn []
        ; x flows into a matmul path AND an elementwise path that reconverge. A
        ; matmul backward that overwrote (rather than accumulated) x's gradient
        ; would silently drop whichever contribution the traversal wrote first.
        (assert-full-gradcheck
          (fn [ts]
            (let [x (. ts 1)
                  w (. ts 2)
                  h (tensor.sum (tensor.matmul x w))
                  g (tensor.sum (tensor.mul x x))]
              (tensor.add h g)))
          [(tensor.new [2 3] [1 2 3 4 5 6])
           (tensor.new [3 2] [1 0 0 1 1 1])]
          0.0001 0.01)))

    (it "does not mutate leaf data or shapes when traversing the graph"
      (fn []
        (let [x (tensor.new [2 3] [1 2 3 4 5 6])
              loss (tensor.sum (tensor.mul x x))]
          (tensor.backward! loss)
          (assert-data [1 2 3 4 5 6] x.data)
          (assert.same [2 3] x.shape))))

    (it "accumulates onto pre-existing parent gradients rather than overwriting them"
      (fn []
        (let [x (tensor.scalar 3)]
          (set x.gradient (tensor.scalar 100))
          (let [loss (tensor.add x (tensor.scalar 1))]
            (tensor.backward! loss)
            (assert.equal 101 (. x.gradient.data 1))))))))

(describe "Tensor:zero_grad"
  (fn []
    (it "resets a matrix tensor's gradient to zeros matching its shape"
      (fn []
        (let [m (tensor.new [2 3] [1 2 3 4 5 6])]
          (set m.gradient (tensor.new m.shape [1 2 3 4 5 6]))
          (tensor.zero-grad! m)
          (assert.same m.shape m.gradient.shape)
          (assert.equal (length m.data) (length m.gradient.data))
          (assert-data [0 0 0 0 0 0] m.gradient.data))))

    (it "resets a scalar tensor's gradient to zero"
      (fn []
        (let [s (tensor.scalar 5)]
          (set s.gradient (tensor.scalar 42))
          (tensor.zero-grad! s)
          (assert.same [] s.gradient.shape)
          (assert.equal 0 (. s.gradient.data 1)))))

    (it "does not mutate the tensor's own data or shape"
      (fn []
        (let [m (tensor.new [2 3] [1 2 3 4 5 6])]
          (set m.gradient (tensor.new m.shape [9 9 9 9 9 9]))
          (tensor.zero-grad! m)
          (assert-data [1 2 3 4 5 6] m.data)
          (assert.same [2 3] m.shape))))

    (it "replaces the gradient tensor rather than mutating the old one in place"
      (fn []
        (let [m (tensor.new [2 3] [1 2 3 4 5 6])]
          (tensor.zero-grad! m) ; materialize a first buffer (lazy impls start from nil)
          (let [old-gradient m.gradient]
            (tset old-gradient.data 1 7)
            (tensor.zero-grad! m)
            (assert.is_false (rawequal old-gradient m.gradient))
            (assert.equal 7 (. old-gradient.data 1))))))))

