(local Vocab (require :fnl.vocab))
(local assert (require :luassert))

(describe "Vocab:size"
  (fn []
    (it "reports the number of tokens it was constructed with"
      (fn []
        (let [three (Vocab.make ["a" "b" "c"])
              empty (Vocab.make [])]
          (assert.equal 3 (three:size))
          (assert.equal 0 (empty:size)))))))

(describe "Vocab:encode"
  (fn []
    (it "maps each token to its 1-based position in the vocabulary"
      (fn []
        (let [vocab (Vocab.make ["cat" "dog" "fish"])]
          (assert.equal 1 (vocab:encode "cat"))
          (assert.equal 2 (vocab:encode "dog"))
          (assert.equal 3 (vocab:encode "fish")))))

    (it "errors on a token that is not in the vocabulary"
      (fn []
        (let [vocab (Vocab.make ["cat" "dog"])]
          (assert.has_error (fn [] (vocab:encode "bird"))
                            "Token not in vocab: bird"))))))

(describe "Vocab:one_hot"
  (fn []
    (it "encodes a token as a 1-D one-hot vector the size of the vocabulary"
      (fn []
        (let [vocab (Vocab.make ["cat" "dog" "fish"])
              hot (vocab:one_hot "dog")]
          (assert.same [3] hot.shape)
          ; "dog" is id 2, so only the second slot is set.
          (assert.same [0 1 0] hot.data)
          ; built with require_grad = false, so it carries no gradient buffer
          (assert.is_nil hot.gradient))))

    (it "errors on a token that is not in the vocabulary"
      (fn []
        (let [vocab (Vocab.make ["cat"])]
          (assert.has_error (fn [] (vocab:one_hot "bird"))))))))
