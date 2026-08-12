(local tensor (require :fnl.tensor))
(local assert (require :luassert))
(local assert-data (. (require :test.helper) :assert-data))

; These are contract tests for the stride-based N-D milestone. They intentionally
; exercise public behavior rather than asserting storage layout or allocations.
; They fail until strides, generalized broadcasting, and their backward rules exist.
(describe "N-D stride tensor behavior"
  (fn []
    (it "transposes arbitrary axis pairs and preserves logical values through an elementwise op"
      (fn []
        (let [x (tensor.new [2 2 3] [1 2 3 4 5 6 7 8 9 10 11 12])
              out (tensor.relu (tensor.transpose x 1 3))]
          (assert.same [3 2 2] out.shape)
          (assert-data [1 7 4 10 2 8 5 11 3 9 6 12] out.data))))

    (it "normalizes negative transpose axes and rejects duplicate axes"
      (fn []
        (let [x (tensor.new [2 2 3] [1 2 3 4 5 6 7 8 9 10 11 12])]
          (assert.equal (tensor.transpose x 1 3) (tensor.transpose x 1 -1))
          (assert.has_error (fn [] (tensor.transpose x 2 2)))
          (assert.has_error (fn [] (tensor.transpose (tensor.new [2 2] [1 2 3 4]))))))

    (it "keeps axis operations correct after transposing N-D logical dimensions"
      (fn []
        (let [x (tensor.new [2 3] [1 2 3 4 5 6])
              transposed (tensor.transpose x 1 2)
              probs (tensor.softmax transposed 1)
              target (tensor.new [3 2] [0 0 0 0 1 1])
              loss ((. tensor :cross-entropy) transposed target 1)]
          (assert.same [3 2] probs.shape)
          (assert.near (/ 1 (+ 1 (math.exp 1) (math.exp 2))) (. probs.data 1) 0.000000000001)
          (assert.near (/ (math.exp 1) (+ 1 (math.exp 1) (math.exp 2))) (. probs.data 3) 0.000000000001)
          (assert.near (/ (math.exp 2) (+ 1 (math.exp 1) (math.exp 2))) (. probs.data 5) 0.000000000001)
          (assert-data [3 3] (. (tensor.argmax transposed 1) :data))
          (assert.near (- (math.log (/ (math.exp 2) (+ 1 (math.exp 1) (math.exp 2)))))
                       (. loss.data 1) 0.000000000001))))

    (it "keeps unary operations and transpose backward in logical coordinate order"
      (fn []
        (let [x (tensor.new [2 3] [1 2 3 4 5 6])
              transposed (tensor.transpose x 1 2)
              transformed (tensor.pow (tensor.scale transposed (tensor.scalar 2)) (tensor.scalar 2))
              weights (tensor.new [3 2] [1 2 3 4 5 6])
              loss (tensor.sum (tensor.mul transposed weights))]
          (assert-data [4 64 16 100 36 144] transformed.data)
          (tensor.backward! loss)
          (assert-data [1 3 5 2 4 6] x.gradient.data))))

    (it "broadcasts size-one dimensions at any aligned N-D axis"
      (fn []
        (let [left (tensor.new [2 1 3] [1 2 3 4 5 6])
              right (tensor.new [1 4 1] [10 20 30 40])
              out (tensor.add left right)]
          (assert.same [2 4 3] out.shape)
          (assert-data [11 12 13 21 22 23 31 32 33 41 42 43
                        14 15 16 24 25 26 34 35 36 44 45 46]
                       out.data))))

    (it "broadcasts correctly when an operand has transposed logical strides"
      (fn []
        (let [x (tensor.new [2 3] [1 2 3 4 5 6])
              rows (tensor.new [3 1] [10 20 30])
              out (tensor.add (tensor.transpose x 1 2) rows)]
          (assert.same [3 2] out.shape)
          (assert-data [11 14 22 25 33 36] out.data))))

    (it "unbroadcasts gradients over every size-one expanded dimension"
      (fn []
        (let [left (tensor.new [2 1 3] [1 2 3 4 5 6])
              right (tensor.new [1 4 1] [10 20 30 40])
              loss (tensor.sum (tensor.add left right))]
          (tensor.backward! loss)
          (assert.same [2 1 3] left.gradient.shape)
          (assert-data [4 4 4 4 4 4] left.gradient.data)
          (assert.same [1 4 1] right.gradient.shape)
          (assert-data [6 6 6 6] right.gradient.data))))

    (it "multiplies batched matrices while broadcasting their leading dimensions"
      (fn []
        (let [left (tensor.new [2 2 3] [1 2 3 4 5 6 7 8 9 10 11 12])
              right (tensor.new [1 3 2] [1 2 3 4 5 6])
              out (tensor.matmul left right)]
          (assert.same [2 2 2] out.shape)
          (assert-data [22 28 49 64 76 100 103 136] out.data))))

    (it "unbroadcasts the broadcast batch operand in batched matmul backward"
      (fn []
        (let [left (tensor.new [2 2 3] [1 2 3 4 5 6 7 8 9 10 11 12])
              right (tensor.new [1 3 2] [1 2 3 4 5 6])
              loss (tensor.sum (tensor.matmul left right))]
          (tensor.backward! loss)
          (assert.same [2 2 3] left.gradient.shape)
          (assert-data [3 7 11 3 7 11 3 7 11 3 7 11] left.gradient.data)
          (assert.same [1 3 2] right.gradient.shape)
          (assert-data [22 22 26 26 30 30] right.gradient.data))))

    (it "gathers whole slices along a specified source axis"
      (fn []
        (let [source (tensor.new [2 3 2] [1 2 3 4 5 6 7 8 9 10 11 12])
              indices (tensor.new [2] [3 1])
              out (tensor.gather source indices 2)]
          (assert.same [2 2 2] out.shape)
          (assert-data [5 6 1 2 11 12 7 8] out.data)
          (assert.equal out (tensor.gather source indices -2)))))

    (it "scatters generalized gather gradients into the selected source slices"
      (fn []
        (let [source (tensor.new [2 3 2] [1 2 3 4 5 6 7 8 9 10 11 12])
              indices (tensor.new [2] [3 1])
              out (tensor.gather source indices 2)]
          (set out.gradient (tensor.new out.shape [1 1 1 1 1 1 1 1]))
          (tensor.backward-step! out)
          (assert.same [2 3 2] source.gradient.shape)
          (assert-data [1 1 0 0 1 1 1 1 0 0 1 1] source.gradient.data)))))))
