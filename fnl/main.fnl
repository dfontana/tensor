(local tensor (require :fnl.tensor))
(local model (require :fnl.model))
(local opt (require :fnl.optimizer))
(local Vocab (require :fnl.vocab))

; Train `net` on `input` against `loss-fn` for `steps` SGD steps (lr 0.05).
; Returns the loss trajectory (one scalar per step) so callers can compare curves.
(fn train [net input loss-fn steps]
  (let [params net.params
        losses []]
    (for [_ 1 steps]
      (opt.zero! params)
      (let [prediction (net.forward input)
            loss (loss-fn prediction)]
        (tensor.backward! loss)
        (opt.step! params 0.05)
        (table.insert losses (. loss.data 1))))
    losses))

; Relu example: learning the xor function.
(fn learn-xor [steps]
  (let [x (tensor.new [4 2] [0 0 0 1 1 0 1 1])
        y (tensor.new [4 1] [0 1 1 0])
        xor (model.Sequential
              (model.Linear 2 4)
              (model.ReLU)
              (model.Linear 4 1))]
    (train xor x (fn [pred] (opt.mean-squared pred y)) steps)))

; Next-token prediction over one-hot inputs (bigram).
(fn learn-bigram [steps]
  (let [vocab (Vocab.make ["red" "green" "blue"])
        inputs (vocab:one_hot_many ["red" "green" "blue"])
        targets (vocab:one_hot_many ["green" "blue" "red"])
        net (model.Linear (vocab:size) (vocab:size))]
    (train net inputs (fn [pred] (tensor.cross-entropy pred targets -1)) steps)))

; Next-token prediction through a learned embedding.
(fn learn-embeds [steps]
  (let [vocab (Vocab.make ["red" "green" "blue"])
        input-ids (vocab:encode_many ["red" "green" "blue"])
        targets (vocab:one_hot_many ["green" "blue" "red"])
        net (model.Sequential
              (model.Embedding vocab 32)
              (model.Linear 32 (vocab:size)))]
    (train net input-ids (fn [pred] (tensor.cross-entropy pred targets -1)) steps)))


(math.randomseed 42)
(let [losses (learn-embeds 2000)]
  (print (.. "loss after 2000 steps: " (. losses (length losses)))))
