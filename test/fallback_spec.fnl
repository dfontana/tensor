(local tensor (require :fnl.tensor))
(local optimizer (require :fnl.optimizer))
(local assert (require :luassert))
(local assert-data (. (require :test.helper) :assert-data))

(describe "forced no-native fallback"
  (fn []
    (it "keeps tensor data as genuine Lua tables"
      (fn []
        (let [a (tensor.new [2] [1 2])
              b (tensor.new [2] [3 4])
              out (tensor.add a b)
              broadcast (tensor.add a (tensor.scalar 2))
              probabilities (tensor.softmax a 1)]
          (assert.equal "table" (type a.data))
          (assert.equal "table" (type out.data))
          (assert.equal "table" (type broadcast.data))
          (assert.equal "table" (type probabilities.data))
          (assert-data [4 6] out.data)
          (assert-data [3 4] broadcast.data)
          (assert.has_error (fn [] (tensor.mean (tensor.new [0] [])))))))

    (it "keeps autograd and optimizer behavior on the table path"
      (fn []
        (let [x (tensor.new [2] [1 2])
              y (tensor.new [2] [3 4])
              out (tensor.add x y)]
          (set out.gradient (tensor.new out.shape [1 2]))
          (tensor.backward-step! out)
          (assert.equal "table" (type x.gradient.data))
          (assert-data [1 2] x.gradient.data))
        (let [parameter (tensor.new [2] [1 2])
              gradient (tensor.new [2] [1 1])
              params {:parameter parameter}]
          (set parameter.gradient gradient)
          (optimizer.step! params 0.1)
          (assert-data [0.9 1.9] parameter.data))))))
