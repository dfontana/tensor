(local {: uniform : zeroes : add : matmul : relu : gather} (require :fnl.tensor))

; A model is a plain record: a `forward` function of the input tensor, and a
; `params` VALUE (the list of trainable tensors).

; 2D only (N-D comes later).
(fn linear [in-feats out-feats]
  (let [weight (uniform [in-feats out-feats] -0.1 0.1)
        bias (zeroes [out-feats] {:require_grad true})]
    {:forward (fn [x] (add (matmul x weight) bias))
     :params [weight bias]}))

(fn embedding [vocab embed-size]
  (let [embed-table (uniform [(vocab:size) embed-size] -0.1 0.1)]
    {:forward (fn [x] (gather embed-table x))
     :params [embed-table]}))

; relu carries no parameters, and its forward IS the relu op — no wrapper closure.
(fn relu-layer []
  {:forward relu :params []})

; Compose layers left to right. forward folds the input through each layer's
; forward; params are flattened once, at construction.
(fn sequential [...]
  (let [layers [...]
        params []]
    (each [_ layer (ipairs layers)]
      (each [_ p (ipairs layer.params)]
        (table.insert params p)))
    {:forward (fn [x]
                (accumulate [acc x _ layer (ipairs layers)]
                  (layer.forward acc)))
     : params}))

{:Linear linear
 :Sequential sequential
 :ReLU relu-layer
 :Embedding embedding}
