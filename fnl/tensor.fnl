(local {
  : numel
  : last
  : scalar?
  : same-shape?
  : add*
  : sub*
  : mul*
  : relu*
  : scale*
  : pow*
  : sum*
  : mean*
  : matmul*
  : transpose*
  : softmax*
  : argmax*
  : cross-entropy*
  : index-select*
  : softmax-backward*
  : cross-entropy-backward*
  : index-select-backward*
  : relu-backward*
  : pow-exponent-backward*
} (require :fnl.kernels))

(fn shape->string [s] (.. "(" (table.concat s ",") ")"))
(local mt {
  :__eq (fn [self t] (if
    (not (same-shape? self.shape t.shape)) false
    (not= (length self.data) (length t.data)) false
    (faccumulate [v true i 1 (length self.data)]
      (and v (= (. self.data i) (. t.data i))))
  ))
  :__tostring (fn [self] (..
    (shape->string self.shape)
    " [" (table.concat self.data ",") "]"
  ))
})

; TODO rename new-tensor
(fn new [shape data options]
  (local tensor {
    : shape
    : data
    :parents (or (?. options :parents) [])
    ; TODO rename to tracked
    :tracked? (not= (?. options :require_grad) false)
    :gradient nil
    :op (or (?. options :op) :noop)
    ; Optional per-op backward context (axis/temp/actual/idx). Set atomically with
    ; :op so a node can never carry an op tag without the state that op needs.
    :ctx (?. options :ctx)
  })
  (setmetatable tensor mt)
  tensor
)
(fn tracked [sd parents op ctx] (new sd.shape sd.data {:parents parents :op op :ctx ctx}))
(fn untracked [sd op] (new sd.shape sd.data {:require_grad false :op op}))

(fn fill [shape v options] (new shape (fcollect [_ 1 (numel shape)] v) options))
(fn zeroes [shape options] (fill shape 0 options))
(fn scalar [v options] (new {} [ v ] options))
(fn uniform [shape low high options] (new
  shape
  (fcollect [_ 1 (numel shape)] (+ (* (math.random) (- high low)) low))
  options))
(fn normal [shape mean stddev options] (new
  shape
  (fcollect [_ 1 (numel shape)] (do
    (local z (*
      (math.sqrt (* -2 (math.log (math.random))))
      (math.cos (* 2 math.pi (math.random)))
    ))
    (+ mean (* stddev z))
  ))
  options))

(fn add [a b] (tracked (add* a b) [ a b ] :add))
(fn sub [a b] (tracked (sub* a b) [ a b ] :sub))
(fn mul [a b] (tracked (mul* a b) [ a b ] :mul))
(fn relu [a] (tracked (relu* a) [a] :relu))
(fn scale [a b] (tracked (scale* a b) [ a b ] :scale))
(fn pow [a b] (tracked (pow* a b) [ a b ] :pow))
(fn sum [a] (tracked (sum* a) [a] :sum))
(fn mean [a] (tracked (mean* a) [a] :mean))
(fn matmul [a b] (tracked (matmul* a b) [ a b ] :matmul))
(fn transpose [a] (tracked (transpose* a) [a] :transpose))

(fn softmax [a axis temp] (tracked (softmax* a axis temp) [a] :softmax {:axis axis :temp (or temp 1.0)}))
(fn argmax [a axis] (untracked (argmax* a axis) :argmax))
(fn cross-entropy [a actual axis] (tracked (cross-entropy* a actual axis) [a] :cross-entropy {:actual actual :axis axis}))
; Select the given indicies (selectt is a tensor used for accessing) from a
; This is mostly relevant for extracting embeddings, etc
(fn index-select [a selectt] (tracked (index-select* a selectt) [a] :index-select {:idx selectt}))

(fn zero-grad! [a] (set a.gradient (zeroes a.shape {:require_grad false})))
(fn accum-grad! [a grad]
  ; Untracked don't backprop, skip
  (when a.tracked?
    (when (= a.gradient nil)
      ; Lazy init gradients when missing
      (set a.gradient (zeroes a.shape {:require_grad false})))
    (let [ag a.gradient
          n (last a.shape)]
      (if
        ; same shape -> straight elementwise accumulation
        (same-shape? a.shape grad.shape)
          (for [i 1 (length ag.data)]
            (tset ag.data i (+ (. ag.data i) (. grad.data i))))
        ; scalar grad -> broadcast against a
        (scalar? grad)
            (for [i 1 (length ag.data)]
              (tset ag.data i (+ (. ag.data i) (. grad.data 1))))
        ; self is scalar so collapse to one
        (scalar? a)
            (tset ag.data 1 (faccumulate [t (. ag.data 1) i 1 (length grad.data)] (+ t (. grad.data i))))
        ; Self is a vector, broadcast along the tensor's last dimension
        ; sum every leading (broadcast) dimension back onto the vector's elements
        (for [i 1 (length grad.data)]
            (let [j (+ (% (- i 1) n) 1)]
              (tset ag.data j (+ (. ag.data j) (. grad.data i)))))
      ))))
; One node's local gradient rule: read this node's OWN gradient (the upstream,
; already fully accumulated by the time we get here) and push a contribution
; into each parent via accum-grad!
(fn backward-step! [a]
  (local grad a.gradient)
  (fn par [n] (. a.parents n))
  (fn acc-par [...] (each [n v (ipairs [...])] (accum-grad! (par n) v)))
  (case a.op
  :add (acc-par grad grad)
  :sub (acc-par grad (scale* grad (scalar -1)))
  :mul (acc-par (mul* grad (par 2)) (mul* grad (par 1)))
  :scale (acc-par (scale* grad (par 2)) (mul* (par 1) grad))
  :mean (acc-par (scale* grad (scalar (/ 1 (numel (. (par 1) :shape))))))
  :sum  (acc-par grad)
  :matmul (acc-par
    (matmul* grad (transpose* (par 2)))
    (matmul* (transpose* (par 1)) grad))
  :transpose  (acc-par (transpose* grad))
  :pow (acc-par
        ; base grad is chain-rule plumbing over existing kernels: grad ⊙ t ⊙ self^(t-1)
        (mul* (mul* grad (par 2))
          (pow* (par 1) (sub* (par 2) (scalar 1))))
        ; exponent grad is new elementwise math (the log guard) -> its own kernel;
        ; accum-grad! then collapses the self-shaped result onto the scalar exponent
        (pow-exponent-backward* (par 1) a grad))
  :relu (acc-par (relu-backward* a grad))
  ; ctx stashed at forward carries axis/temp; probs come from the node's own data
  :softmax (accum-grad! (par 1) (softmax-backward* a grad a.ctx.axis a.ctx.temp))
  ; ctx carries actual + axis; the backward kernel recomputes the probabilities
  :cross-entropy (accum-grad! (par 1) (cross-entropy-backward* (par 1) grad a.ctx.actual a.ctx.axis))
  ; ctx carries the index tensor (deliberately not a parent); scatter-add onto the source
  :index-select (accum-grad! (par 1) (index-select-backward* (par 1) grad a.ctx.idx))
  _ nil
))

(fn backward! [a]
  (assert (scalar? a) "Can only run backwards on a scalar")
  (set a.gradient (scalar 1 {:require_grad false}))
  ; Iterative DFS building a post-order; walk it in reverse so every node's
  ; gradient is fully accumulated before it feeds its parents. `seen`, keyed by
  ; node identity, dedupes diamonds (a node reused by more than one child).
  (let [order []
        seen {a true}
        frontier [{:node a :expanded false}]]
    (while (not= (length frontier) 0)
      (let [item (table.remove frontier)
            node item.node]
        (if (not item.expanded)
          (do
            ; Re-push expanded, then push any unseen parents to expand first.
            (table.insert frontier {:node node :expanded true})
            (each [_ p (ipairs node.parents)]
              (when (not (. seen p))
                (tset seen p true)
                (table.insert frontier {:node p :expanded false}))))
          ; Second pop (expanded): all parents already queued, record the node.
          (table.insert order node))))
    (for [i (length order) 1 -1]
      (backward-step! (. order i)))))

{
  : new
  : fill
  : zeroes
  : scalar
  : uniform
  : normal
  
  : scalar?
  : same-shape?
  : shape->string

  : numel

  : add
  : sub
  : mul
  : relu
  : scale
  : pow
  : sum
  : mean
  : matmul
  : transpose
  : softmax
  : argmax
  : cross-entropy
  ; TODO fix export name
  :gather index-select

  : zero-grad!
  : backward!
  : backward-step!
}
