; Kernels take and return plain {shape data} - Nothing else

; Loading the optional native module is deliberately local to the Fennel
; integration. A test-only environment switch keeps the table implementation
; reachable even when the shared object is installed.
(local native
  (if (= (os.getenv "TENSOR_FORCE_NO_NATIVE") "1")
      nil
      (let [(ok? module) (pcall require :tensor_native)]
        (and ok? module))))

(fn last [a] (. a (length a)))

(fn scalar? [at] (= (length at.shape) 0))
; TODO N-D: equality remains shape-only, but all logical reads below must translate
; N-D coordinates through tensor strides rather than assume dense data positions.
(fn same-shape? [a-shape b-shape] (if
  (not= (length a-shape) (length b-shape)) false
  (faccumulate [v true i 1 (length a-shape)]
    (and v (= (. a-shape i) (. b-shape i))))
))

(fn numel [shape] (faccumulate [n 1 i 1 (length shape)] (* n (. shape i))))

; TODO N-D: replace this contiguous, special-case reader with coordinate-to-storage
; addressing via strides. Broadcast-aligned dimensions use stride 0.
; Read an operand's value for the i-th element of a broadcast result of `size`
; elements. Scalars repeat their single value; a vector {N} repeats along the
; (contiguous) last dimension; a full-shaped operand indexes directly.
(fn broadcast-get [shape data i size] (if
  (= (length shape) 0) (. data 1)
  (= (length data) size) (. data i)
  (. data (+ (% (- i 1) (last shape)) 1))))

; TODO N-D: align dimensions from the right and allow equal sizes or either size 1;
; construct aligned stride metadata with leading/broadcast dimensions set to zero.
; Compute the broadcast output shape against the given Tensors. Only two forms are
; supported: scalar {} against anything, and a vector {N} against a tensor whose
; last dimension is N. Anything else (e.g. {4,3}+{4}, {4,3}+{4,1}) is an error.
(fn broadcast-shape [a b] (if
  (= (length a) 0) b
  (= (length b) 0) a
  (same-shape? a b) a
  (and (= (length a) 1) (= (last b) (. a 1))) b
  (and (= (length b) 1) (= (last a) (. b 1))) a
  (error ( .. "cannot broadcast (" (table.concat a ",") ") with (" (table.concat b ",") ")"))))

; TODO N-D: iterate output coordinates and read both operands through their aligned
; strides. The output may remain newly allocated contiguous storage for now.
; Apply an 'op' (binary function) to each element of data
(fn binary-elementwise [a b op]
  (local shape (broadcast-shape a.shape b.shape))
  (local size (numel shape))
  {
    : shape
    :data (fcollect [i 1 size] (op
      (broadcast-get a.shape a.data i size)
      (broadcast-get b.shape b.data i size)))
  })

; TODO N-D: read `a` through its strides; preserving its logical shape does not mean
; its values occupy contiguous locations after transpose.
; Apply an 'op' (unary function) to each element of data
(fn unary-elementwise [a op] {
  :shape a.shape
  :data (fcollect [i 1 (numel a.shape)] (op (. a.data i)))
  })

; TODO N-D: reduce logical elements through strides rather than linearly walking
; storage, so reductions of transposed tensors remain correct.
; Apply a reduction operation over data, to both aggregate and finalize
(fn reduce [at reducer finish]
  (local acc (faccumulate [v 0 i 1 (numel at.shape)] (reducer v (. at.data i))))
  {
    :shape []
    :data [(finish acc)]
  })

(fn native-storage? [data]
  (and native (native.is_storage data)))

(fn native-binary [at bt kernel]
  (let [out (native.storage_zeros (numel at.shape))]
    (kernel at.data bt.data out)
    {:shape at.shape :data out}))

(fn add* [at bt]
  (if (and (same-shape? at.shape bt.shape)
           (native-storage? at.data)
           (native-storage? bt.data))
      (native-binary at bt native.add_into)
      (binary-elementwise at bt #(+ $1 $2))))
(fn sub* [at bt]
  (if (and (same-shape? at.shape bt.shape)
           (native-storage? at.data)
           (native-storage? bt.data))
      (native-binary at bt native.sub_into)
      (binary-elementwise at bt #(- $1 $2))))
(fn mul* [at bt]
  (if (and (same-shape? at.shape bt.shape)
           (native-storage? at.data)
           (native-storage? bt.data))
      (native-binary at bt native.mul_into)
      (binary-elementwise at bt #(* $1 $2))))

(fn relu* [at]
  (if (native-storage? at.data)
      (let [out (native.storage_zeros (numel at.shape))]
        (native.relu_into at.data out)
        {:shape at.shape :data out})
      (unary-elementwise at #(math.max 0 $1))))
(fn scale* [at bt]
  (assert (scalar? bt))
  (if (and (native-storage? at.data) (native-storage? bt.data))
      (let [out (native.storage_zeros (numel at.shape))]
        (native.scale_into at.data (. bt.data 1) out)
        {:shape at.shape :data out})
      (unary-elementwise at #(* $1 (. bt.data 1)))))
(fn pow* [at bt]
  (assert (scalar? bt))
  (if (and (native-storage? at.data) (native-storage? bt.data))
      (let [out (native.storage_zeros (numel at.shape))]
        (native.pow_into at.data (. bt.data 1) out)
        {:shape at.shape :data out})
      (unary-elementwise at #(^ $1 (. bt.data 1)))))

(fn sum* [at]
  (if (native-storage? at.data)
      {:shape [] :data [(native.sum at.data)]}
      (reduce at #(+ $1 $2) #$)))
(fn mean* [at]
  (assert (> (numel at.shape) 0) "mean is undefined for an empty tensor")
  (if (native-storage? at.data)
      {:shape [] :data [(native.mean at.data)]}
      (reduce at #(+ $1 $2) #(/ $1 (numel at.shape)))))

; TODO N-D: treat the final two dimensions as matrices; broadcast every leading
; batch dimension, and address both inputs by their batch/matrix strides.
(fn matmul* [at bt]
  ; TODO: 2D limited atm
  (assert (= (. at.shape 2) (. bt.shape 1)))
  (let [m (. at.shape 1)   ; rows of A / of output
        k (. at.shape 2)   ; shared inner dim
        n (. bt.shape 2)]  ; cols of B / of output
    (if (and (= (length at.shape) 2)
             (= (length bt.shape) 2)
             (native-storage? at.data)
             (native-storage? bt.data))
        (let [out (native.storage_zeros (* m n))]
          (native.matmul_into at.data bt.data m k n out)
          {:shape [m n] :data out})
        {:shape [m n]
         :data (fcollect [idx 1 (* m n)]
                 (let [r (// (- idx 1) n)
                       c (% (- idx 1) n)]
                   (faccumulate [dot 0 p 1 k]
                     (+ dot (* (. at.data (+ (* r k) p))
                               (. bt.data (+ (* (- p 1) n) c 1)))))))})))


; TODO N-D: accept two normalized axes, swap their shape and stride entries, and
; return a view sharing the existing data. Do not copy values; reject scalars
; because a rank-zero tensor has no valid axes.
(fn transpose* [at] (if
    (scalar? at) at
    (= (length at.shape) 2) {
      :shape [(. at.shape 2) (. at.shape 1)]
      :data
        (fcollect [k 1 (numel at.shape)] (do
          (local r (// (- k 1) (. at.shape 1)))
          (local c (% (- k 1) (. at.shape 1)))
          (. at.data (+ (* c (. at.shape 2)) r 1))))
    }
    ; TODO: 2D limited atm
    (error "Transpose only supports scalar or 2D")))

; TODO N-D: reuse this normalization for transpose's two axes and gather's source
; axis; transpose additionally rejects selecting the same axis twice.
; Normalize a (possibly negative) axis to a 1-based index; error if out of range.
(fn norm-axis [shape axis]
  (let [ax (if (< axis 0) (+ (length shape) axis 1) axis)]
    (assert (and (>= ax 1) (<= ax (length shape))) "axis out of range")
    ax))

; TODO N-D: derive slice starts and step sizes from tensor strides, not products
; of logical dimensions, so axis reductions work on a transposed view.
; Walk each 1-D slice along `axis`, calling (f start width stride) per slice.
; `start` is the 1-based offset of the slice's first element; successive
; elements sit at start + i*stride for i in 0..width-1.
(fn each-axis-slice [shape axis f]
  (let [width (. shape axis)
        outer (faccumulate [o 1 d 1 (- axis 1)] (* o (. shape d)))
        inner (faccumulate [n 1 d (+ axis 1) (length shape)] (* n (. shape d)))]
    (for [oi 0 (- outer 1)]
      (for [ii 0 (- inner 1)]
        (f (+ (* oi width inner) ii 1) width inner)))))

; TODO N-D: use stride-aware slice offsets and write a fresh contiguous output in
; logical coordinate order; the returned arrays must not use source storage offsets.
; Stable softmax over each slice along `axis`. Returns two offset-keyed arrays,
; dense over 1..numel: probabilities and log-probabilities.
(fn softmax-values [at axis temp]
  (let [probs {} logprobs {}]
    (each-axis-slice at.shape axis
      (fn [start width stride]
        (var maxv (- math.huge))
        (for [i 0 (- width 1)]
          (set maxv (math.max maxv (. at.data (+ start (* i stride))))))
        (var total 0)
        (for [i 0 (- width 1)]
          (let [off (+ start (* i stride))
                e (math.exp (/ (- (. at.data off) maxv) temp))]
            (tset probs off e)
            (set total (+ total e))))
        (let [logtotal (math.log total)]
          (for [i 0 (- width 1)]
            (let [off (+ start (* i stride))]
              (tset probs off (/ (. probs off) total))
              (tset logprobs off (- (/ (- (. at.data off) maxv) temp) logtotal)))))))
    (values probs logprobs)))

; TODO N-D: output retains the input's logical N-D shape while reading any view
; through strides.
(fn softmax* [at axis temp]
  (let [axis (norm-axis at.shape axis)
        temp (or temp 1.0)]
    (assert (> temp 0) "temperature must be positive")
    (let [(probs _) (softmax-values at axis temp)]
      {:shape at.shape :data probs})))

; TODO N-D: enumerate axis slices through the tensor's stride metadata. Removing
; the selected axis from shape is unchanged.
; Index of the largest value along `axis`; output shape is the input with that
; axis removed (a vector reduces to a scalar). Ties choose the first occurrence.
(fn argmax* [at axis]
  (let [axis (norm-axis at.shape axis)
        out-shape (fcollect [d 1 (length at.shape)] (if (not= d axis) (. at.shape d)))
        data {}]
    (each-axis-slice at.shape axis
      (fn [start width stride]
        (var best-i 1)
        (var best-v (. at.data start))
        (for [i 1 (- width 1)]
          (let [v (. at.data (+ start (* i stride)))]
            (when (> v best-v)
              (set best-v v)
              (set best-i (+ i 1)))))
        (table.insert data best-i)))
    {:shape out-shape :data data}))

; TODO N-D: require matching logical shapes, then read both tensors through their
; strides while accumulating loss per axis slice.
(fn cross-entropy* [at actual axis]
  (assert (same-shape? at.shape actual.shape) "shape mismatch")
  (let [axis (norm-axis at.shape axis)
        size (numel at.shape)
        slice-count (/ size (. at.shape axis))
        (_ logprobs) (softmax-values at axis 1.0)
        total-loss (faccumulate [acc 0 i 1 size]
                     (- acc (* (. actual.data i) (. logprobs i))))]
    {:shape [] :data [(/ total-loss slice-count)]}))

; TODO N-D: accept a source axis and replace it in the output shape with `idxt.shape`.
; Iterate source prefix, index coordinates, and source suffix using strides.
; Select whole rows of a 2-D table `at` by the indices in `idxt`; output shape is
; idxt.shape with the embedding width appended.
(fn index-select* [at idxt]
  (assert (= (length at.shape) 2) "gather only works on 2D tensor")
  (let [embed (. at.shape 2)
        out-shape (fcollect [i 1 (length idxt.shape)] (. idxt.shape i))]
    (table.insert out-shape embed)
    {:shape out-shape
     :data (fcollect [k 1 (* (numel idxt.shape) embed)]
             (let [row (. idxt.data (+ (// (- k 1) embed) 1))  ; which index entry
                   c   (+ (% (- k 1) embed) 1)]                ; column within the row
               (. at.data (+ (* (- row 1) embed) c))))}))

; TODO N-D: traverse each axis slice through strides for both probabilities and
; upstream gradient, producing a contiguous logical-gradient tensor.
; Gradient of softmax. `probst` is the softmax output node (its data holds the
; probabilities), `gradt` the upstream. Per slice: g[off] = (p/temp)·(up − dot),
; where dot = Σ up·p is the slice's coupling term.
(fn softmax-backward* [probst gradt axis temp]
  (let [axis (norm-axis probst.shape axis)
        out {}]
    (each-axis-slice probst.shape axis
      (fn [start width stride]
        (var dot 0)
        (for [i 0 (- width 1)]
          (let [off (+ start (* i stride))]
            (set dot (+ dot (* (. gradt.data off) (. probst.data off))))))
        (for [i 0 (- width 1)]
          (let [off (+ start (* i stride))]
            (tset out off (* (/ (. probst.data off) temp)
                             (- (. gradt.data off) dot)))))))
    {:shape probst.shape :data out}))

; TODO N-D: read logits, targets, and axis slices via strides; output is laid out
; in the logits' logical coordinate order for later accumulation.
; Gradient of cross-entropy w.r.t. the logits `logitst`. Recomputes the softmax
; probabilities (logits aren't mutated before backward), then g[i] = up·(p − y)/slices.
(fn cross-entropy-backward* [logitst gradt actual axis]
  (let [axis (norm-axis logitst.shape axis)
        size (numel logitst.shape)
        slice-count (/ size (. logitst.shape axis))
        (probs _) (softmax-values logitst axis 1.0)
        upstream (. gradt.data 1)]
    {:shape logitst.shape
     :data (fcollect [i 1 size]
             (/ (* upstream (- (. probs i) (. actual.data i))) slice-count))}))

; TODO N-D: scatter whole selected slices into a contiguous source-shaped gradient,
; using the saved source axis and source/index coordinate-to-storage mappings.
; Gradient of index-select back onto the 2-D source `sourcet`. Scatter-add: each
; selected row collects the upstream of every output slot that picked it (a row
; chosen more than once accumulates).
(fn index-select-backward* [sourcet gradt idxt]
  (let [embed (. sourcet.shape 2)
        data (fcollect [_ 1 (numel sourcet.shape)] 0)]
    (for [batch 1 (numel idxt.shape)]
      (let [row (. idxt.data batch)]
        (for [col 1 embed]
          (let [out-off (+ (* (- batch 1) embed) col)
                dst     (+ (* (- row 1) embed) col)]
            (tset data dst (+ (. data dst) (. gradt.data out-off)))))))
    {:shape sourcet.shape :data data}))

; TODO N-D: read the activation through strides and pair it with the upstream by
; logical coordinate, not matching flat storage indices.
; Gradient of relu: pass the upstream through only where the activation fired.
; Masking on the OUTPUT is valid because relu(x) > 0 ⟺ x > 0, so the pre-activation
; input never has to be saved.
(fn relu-backward* [activatedt gradt]
  {:shape activatedt.shape
   :data (icollect [i v (ipairs activatedt.data)]
           (if (> v 0) (. gradt.data i) 0))})

; TODO N-D: pair base/result/upstream values by logical coordinate through strides.
; Gradient of pow w.r.t. its scalar exponent: grad ⊙ self^t ⊙ ln(self), contributing
; 0 where self ≤ 0 (log undefined there). Returned self-shaped; accum-grad! then
; collapses it onto the scalar exponent.
(fn pow-exponent-backward* [baset resultt gradt]
  {:shape baset.shape
   :data (icollect [i x (ipairs baset.data)]
           (if (> x 0) (* (. gradt.data i) (. resultt.data i) (math.log x)) 0))})

{
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
}
