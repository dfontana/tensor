(local tensor (require :fnl.tensor))
(local optimizer (require :fnl.optimizer))
(local assert (require :luassert))
(local assert-data (. (require :test.helper) :assert-data))

; Free-function API. The kebab / `!` names are reached through keyword field
; access because Lua receives those exact exported names.
(local zero (. optimizer :zero!))
(local step (. optimizer :step!))
(local mean-squared (. optimizer :mean-squared))
(local backward (. tensor :backward!))

(describe "optimizer.zero!"
  (fn []
    (it "resets the gradients of every parameter in the (string-keyed) param map"
      (fn []
        (let [params {:m (tensor.new [2 3] [1 2 3 4 5 6])
                      :b (tensor.scalar 0.2)}]
          (set params.m.gradient (tensor.new params.m.shape [9 9 9 9 9 9]))
          (set params.b.gradient (tensor.scalar 9))
          (zero params)
          (assert-data [0 0 0 0 0 0] params.m.gradient.data)
          (assert.equal 0 (. params.b.gradient.data 1)))))))

(describe "optimizer.step!"
  (fn []
    (it "moves each parameter one learning-rate step down its gradient"
      (fn []
        (let [params {:m (tensor.new [1 3] [1 2 3])
                      :b (tensor.scalar 10)}]
          (set params.m.gradient (tensor.new params.m.shape [1 1 1]))
          (set params.b.gradient (tensor.scalar 2))
          (step params 0.1)
          ; p := p - lr * grad
          (assert-data [0.9 1.9 2.9] params.m.data)
          (assert.near 9.8 (. params.b.data 1) 0.000001))))))

(describe "optimizer.mean-squared"
  (fn []
    (it "reduces to a scalar tensor holding the mean of the squared errors"
      (fn []
        (let [pred (tensor.new [1 3] [1 2 3])
              actual (tensor.new [1 3] [0 0 0])
              loss (mean-squared pred actual)]
          (assert.same [] loss.shape)
          ; mean(1^2, 2^2, 3^2) = 14 / 3
          (assert.near (/ 14 3) (. loss.data 1) 0.000001))))

    (it "is zero exactly when the prediction equals the target"
      (fn []
        (let [pred (tensor.new [2 2] [1 2 3 4])
              actual (tensor.new [2 2] [1 2 3 4])]
          (assert.equal 0 (. (mean-squared pred actual) :data 1)))))

    (it "is symmetric in prediction and target (the error is squared)"
      (fn []
        (let [a (tensor.new [1 3] [1 2 3])
              b (tensor.new [1 3] [4 0 5])
              first-loss (. (mean-squared a b) :data 1)
              second-loss (. (mean-squared b a) :data 1)]
          (assert.near first-loss second-loss 0.000000001))))

    (it "backpropagates the gradient 2*(pred - actual)/n onto the prediction"
      (fn []
        ; Composed from sub/pow/mean, each already gradchecked in tensor_spec;
        ; this pins the composition's end-to-end gradient at the leaf.
        (let [pred (tensor.new [1 3] [1 2 3])
              actual (tensor.new [1 3] [0 0 0])
              loss (mean-squared pred actual)]
          (backward loss)
          ; 2 * {1, 2, 3} / 3
          (assert.near (/ 2 3) (. pred.gradient.data 1) 0.000001)
          (assert.near (/ 4 3) (. pred.gradient.data 2) 0.000001)
          (assert.near (/ 6 3) (. pred.gradient.data 3) 0.000001))))))

(describe "Optimizer end-to-end (linear regression, the milestone goal)"
  (fn []
    ; Fit y = m*x + b to the single point (x=3, y=1) starting from m=b=0.
    ; With gradients reset each step, the loss must fall monotonically.
    (local forward
      (fn [params]
        (let [pred (tensor.add (tensor.mul params.m (tensor.scalar 3)) params.b)]
          (mean-squared pred (tensor.scalar 1)))))

    (it "decreases the loss on every step and drives it toward zero"
      (fn []
        (let [params {:m (tensor.new [] [0])
                      :b (tensor.new [] [0])}
              final-state
                (faccumulate [state [math.huge nil] _ 1 8]
                  (let [previous (. state 1)]
                    (zero params)
                    (let [loss (forward params)
                          current (. loss.data 1)]
                      (assert.is_true (< current previous))
                      (backward loss)
                      (step params 0.02)
                      [current current])))]
          (assert.is_true (< (. final-state 2) 0.01)))))

    (it "would stall without zero!(): stale gradients accumulate across steps"
      (fn []
        ; Same loop but never zeroing. Gradients from earlier steps pile onto
        ; later ones, so the trajectory is no longer plain gradient descent and
        ; the clean monotonic descent above is lost. This pins down *why* zero!()
        ; is part of the step/zero/forward/backward cycle.
        (let [params {:m (tensor.new [] [0])
                      :b (tensor.new [] [0])}
              final-state
                (faccumulate [state [true math.huge] _ 1 8]
                  (let [monotonic (. state 1)
                        previous (. state 2)
                        loss (forward params)
                        current (. loss.data 1)
                        next-monotonic (and monotonic (< current previous))]
                    (backward loss)
                    (step params 0.02)
                    [next-monotonic current]))]
          (assert.is_false (. final-state 1)))))))
