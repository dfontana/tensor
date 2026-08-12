(local {: scalar : sub : pow : mean : zero-grad!} (require :fnl.tensor))

(local native
  (if (= (os.getenv "TENSOR_FORCE_NO_NATIVE") "1")
      nil
      (let [(ok? module) (pcall require :tensor_native)]
        (and ok? module))))

(fn native-storage? [data]
  (and native (native.is_storage data)))

; Reset every parameter's gradient.
(fn zero! [params]
  (each [_ p (pairs params)]
    (zero-grad! p)))

; One SGD step, in place: p := p - lr * grad. Parameters that never received a gradient (off the backward path) are skipped.
(fn step! [params lr]
  (each [_ p (pairs params)]
    (when p.gradient
      (if (and (native-storage? p.data) (native-storage? p.gradient.data))
          (native.sgd_step p.data p.gradient.data lr)
          (for [i 1 (length p.data)]
            (tset p.data i (- (. p.data i) (* lr (. p.gradient.data i)))))))))

; Mean-squared error, composed from already-gradchecked ops.
(fn mean-squared [pred actual]
  (mean (pow (sub pred actual) (scalar 2))))

{: zero!
 : step!
 :mean-squared mean-squared}
