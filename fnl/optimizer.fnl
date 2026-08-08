(local {: scalar : sub : pow : mean : zero-grad!} (require :fnl.tensor))

; Reset every parameter's gradient.
(fn zero! [params]
  (each [_ p (pairs params)]
    (zero-grad! p)))

; One SGD step, in place: p := p - lr * grad. Parameters that never received a gradient (off the backward path) are skipped.
(fn step! [params lr]
  (each [_ p (pairs params)]
    (when p.gradient
      (for [i 1 (length p.data)]
        (tset p.data i (- (. p.data i) (* lr (. p.gradient.data i))))))))

; Mean-squared error, composed from already-gradchecked ops.
(fn mean-squared [pred actual]
  (mean (pow (sub pred actual) (scalar 2))))

{: zero!
 : step!
 :mean-squared mean-squared}
