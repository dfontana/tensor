(local native (require :tensor_native))
(local tensor (require :fnl.tensor))
(local optimizer (require :fnl.optimizer))
(local assert (require :luassert))
(local assert-data (. (require :test.helper) :assert-data))

(describe "tensor_native"
  (fn []
    (it "imports the Rust module and returns its hello-world greeting"
      (fn []
        (assert.equal "Hello, world!" (native.hello))))))

(describe "native Storage"
  (fn []
    (it "copies dense tables and preserves existing Storage identity"
      (fn []
        (let [storage (native.storage [1 2 3])
              same (native.storage storage)
              copy (storage:clone)]
          (assert.is_true (native.is_storage storage))
          (assert.is_false (native.is_storage [1 2 3]))
          (assert.is_true (rawequal storage same))
          (assert.is_true (native.is_storage copy))
          (assert.is_false (rawequal storage copy))
          (assert.same [1 2 3] (storage:to_table))
          (tset copy 1 99)
          (assert.equal 1 (. storage 1))
          (assert.equal 99 (. copy 1)))))

    (it "supports zero allocation, fill, and fixed-size indexing"
      (fn []
        (let [storage (native.storage_zeros 3)
              empty (native.storage_zeros 0)]
          (assert.equal 3 (length storage))
          (assert.equal 0 (length empty))
          (assert.equal 3 (storage:len))
          (assert.same [0 0 0] (storage:to_table))
          (storage:fill 2.25)
          (assert.same [2.25 2.25 2.25] (storage:to_table))
          (tset storage 2 4.5)
          (assert.equal 4.5 (. storage 2))
          (assert.is_nil (. storage 0))
          (assert.is_nil (. storage 4))
          (assert.is_nil (. storage :unknown))
          (assert.equal 2.25 (. storage 1.0))
          (assert.has_error (fn [] (. storage 1.5)))
          (assert.has_error (fn [] (tset storage 0 1)))
          (assert.has_error (fn [] (tset storage 4 1)))
          (assert.has_error (fn [] (tset storage 1 "1")))
          (assert.has_error (fn [] (tset storage 1 1e39)))
          (assert.has_error (fn [] (storage:fill "1"))))))

    (it "iterates through ipairs and pairs and has a compact representation"
      (fn []
        (let [storage (native.storage [1 2 3])
              ipairs-values []
              pairs-values []
              pairs-keys []]
          (each [index value (ipairs storage)]
            (tset ipairs-values index value))
          (each [index value (pairs storage)]
            (table.insert pairs-keys index)
            (tset pairs-values index value))
          (assert.same [1 2 3] ipairs-values)
          (assert.same [1 2 3] pairs-values)
          (assert.same [1 2 3] pairs-keys)
          (assert.is_not_nil (string.match (tostring storage) "Storage")))))

    (it "narrows values to f32 while accepting non-finite numbers"
      (fn []
        (let [storage (native.storage [16777217 math.huge (/ 0 0)])]
          (assert.equal 16777216 (. storage 1))
          (assert.equal math.huge (. storage 2))
          (assert.is_false (= (. storage 3) (. storage 3)))
          (assert.has_error (fn [] (native.storage [1e39]))))))

    (it "rejects coercible strings and non-sequence tables"
      (fn []
        (let [hole {}]
          (tset hole 1 1)
          (tset hole 3 3)
          (assert.has_error (fn [] (native.storage hole))))
        (assert.has_error (fn [] (native.storage ["1" 2])))
        (assert.has_error (fn [] (native.storage {1 2 :named 3})))
        (assert.has_error (fn [] (native.storage "1")))
        (assert.has_error (fn [] (native.storage_zeros -1)))
        (assert.has_error (fn [] (native.storage_zeros "3")))))))

(describe "native kernels"
  (fn []
    (it "returns outputs and handles binary output aliases"
      (fn []
        (let [lhs (native.storage [1 2 3 4])
              rhs (native.storage [10 20 30 40])
              out (native.storage_zeros 4)]
          (assert.is_true (rawequal out (native.add_into lhs rhs out)))
          (assert.same [11 22 33 44] (out:to_table))
          (native.sub_into lhs rhs lhs)
          (assert.same [-9 -18 -27 -36] (lhs:to_table))
          (native.mul_into lhs rhs rhs)
          (assert.same [-90 -360 -810 -1440] (rhs:to_table))
          (let [alt-out (native.storage_zeros 4)]
            (native.scale_into out alt-out 2)
            (assert.same [22 44 66 88] (alt-out:to_table)))
          (let [both (native.storage [2 3 4])]
            (native.add_into both both both)
            (assert.same [4 6 8] (both:to_table))))))

    (it "handles unary aliases, scalar conversion, and power"
      (fn []
        (let [data (native.storage [-2 0 3])]
          (assert.is_true (rawequal data (native.relu_into data data)))
          (assert.same [0 0 3] (data:to_table))
          (native.scale_into data -2 data)
          (assert.same [0 0 -6] (data:to_table))
          (native.pow_into data 2 data)
          (assert.same [0 0 36] (data:to_table))
          (assert.has_error (fn [] (native.scale_into data 1e39 (native.storage_zeros 3)))))))

    (it "computes reductions and rejects an empty mean"
      (fn []
        (let [data (native.storage [1.5 -2 4.25])
              empty (native.storage_zeros 0)]
          (assert.equal 3.75 (native.sum data))
          (assert.equal 1.25 (native.mean data))
          (assert.equal 0 (native.sum empty))
          (assert.has_error (fn [] (native.mean empty))))))

    (it "handles row-major matmul aliases and zero dimensions"
      (fn []
        (let [lhs (native.storage [1 2 3 4])
              rhs (native.storage [5 6 7 8])]
          (assert.is_true (rawequal lhs (native.matmul_into lhs rhs 2 2 2 lhs)))
          (assert.same [19 22 43 50] (lhs:to_table)))
        (let [lhs (native.storage_zeros 0)
              rhs (native.storage [1 2 3 4 5 6])
              out (native.storage_zeros 0)]
          (native.matmul_into lhs rhs 0 3 2 out)
          (assert.same [] (out:to_table)))
        (let [lhs (native.storage_zeros 0)
              rhs (native.storage_zeros 0)
              out (native.storage [9 9 9 9 9 9])]
          (native.matmul_into lhs rhs out 2 0 3)
          (assert.same [0 0 0 0 0 0] (out:to_table)))
        (let [lhs (native.storage_zeros 6)
              rhs (native.storage_zeros 0)
              out (native.storage_zeros 0)]
          (native.matmul_into lhs rhs 2 3 0 out)
          (assert.same [] (out:to_table))))

    (it "uses an original gradient when SGD aliases parameter and gradient"
      (fn []
        (let [parameter (native.storage [2 -3 4])]
          (native.sgd_step parameter parameter 0.5)
          (assert.same [1 -1.5 2] (parameter:to_table)))))

    (it "validates types, lengths, dimensions, and preserves outputs on errors"
      (fn []
        (let [lhs (native.storage [1 2])
              rhs (native.storage [3])
              out (native.storage [9 9])]
          (assert.has_error (fn [] (native.add_into lhs rhs out)))
          (assert.same [9 9] (out:to_table))
          (assert.has_error (fn [] (native.relu_into [1 2] out)))
          (assert.has_error (fn [] (native.sum [1 2])))
          (assert.has_error (fn [] (native.scale_into lhs "2" out)))
          (assert.has_error (fn [] (native.matmul_into lhs rhs -1 1 1 out)))
          (assert.has_error (fn [] (native.matmul_into lhs rhs 1.0 1 1 out)))
          (assert.has_error (fn [] (native.matmul_into lhs rhs 9223372036854775807 2 2 out)))))))))

(describe "Fennel native integration"
  (fn []
    (it "coerces tensor data and preserves Storage identity"
      (fn []
        (let [storage (native.storage [1 2 3])
              tensor-value (tensor.new [3] storage)]
          (assert.is_true (native.is_storage tensor-value.data))
          (assert.is_true (rawequal storage tensor-value.data)))))

    (it "uses native kernels only for eligible same-shape and rank-2 operations"
      (fn []
        (let [a (tensor.new [2 2] [1 2 3 4])
              b (tensor.new [2 2] [5 6 7 8])
              scalar (tensor.scalar 2)
              added (tensor.add a b)
              subtracted (tensor.sub a b)
              multiplied (tensor.mul a b)
              rectified (tensor.relu (tensor.new [2 2] [-1 2 -3 4]))
              scaled (tensor.scale a scalar)
              powered (tensor.pow a scalar)
              summed (tensor.sum a)
              averaged (tensor.mean a)
              product (tensor.matmul a b)]
          (each [_ value (ipairs [added subtracted multiplied rectified scaled powered
                                  summed averaged product])]
            (assert.is_true (native.is_storage value.data)))
          (assert.same [6 8 10 12] (added.data:to_table))
          (assert.same [-4 -4 -4 -4] (subtracted.data:to_table))
          (assert.same [5 12 21 32] (multiplied.data:to_table))
          (assert.same [0 2 0 4] (rectified.data:to_table))
          (assert.same [2 4 6 8] (scaled.data:to_table))
          (assert.same [1 4 9 16] (powered.data:to_table))
          (assert.equal 10 (. (summed.data:to_table) 1))
          (assert.equal 2.5 (. (averaged.data:to_table) 1))
          (assert.same [19 22 43 50] (product.data:to_table)))))

    (it "keeps broadcasting and unsupported Fennel operations correct"
      (fn []
        (let [matrix (tensor.new [2 3] [1 2 3 4 5 6])
              vector (tensor.new [3] [10 20 30])
              broadcasted (tensor.add matrix vector)
              transposed (tensor.transpose matrix 1 2)
              probabilities (tensor.softmax matrix 2)]
          (each [_ value (ipairs [broadcasted transposed probabilities])]
            (assert.is_true (native.is_storage value.data)))
          (assert.same [11 22 33 14 25 36] (broadcasted.data:to_table))
          (assert.same [1 4 2 5 3 6] (transposed.data:to_table))
          (assert.near 1 (faccumulate [sum 0 i 1 3] (+ sum (. probabilities.data i))) 0.000001)))))

    (it "accumulates gradients in native Storage and updates native parameters"
      (fn []
        (let [x (tensor.new [2] [1 2])
              y (tensor.new [2] [3 4])
              out (tensor.add x y)]
          (set x.gradient (tensor.new x.shape [10 10]))
          (set out.gradient (tensor.new out.shape [1 2]))
          (tensor.backward-step! out)
          (assert.is_true (native.is_storage x.gradient.data))
          (assert.same [11 12] (x.gradient.data:to_table)))
        (let [parameter (tensor.new [3] [1 2 3])
              gradient (tensor.new [3] [1 1 1])
              params {:parameter parameter}]
          (set parameter.gradient gradient)
          (optimizer.step! params 0.1)
          (assert.is_true (native.is_storage parameter.data))
          (assert-data [0.9 1.9 2.9] parameter.data)))))
