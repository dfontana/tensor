(local {: new} (require :fnl.tensor))

; A vocabulary is a record of functions closing over its token list and the
; token->id map.
(fn make [tokens]
  (let [to-id (collect [i tok (ipairs tokens)] tok i)]
    {:size (fn [] (length tokens))
     :encode (fn [_ token]
               (or (. to-id token)
                   (error (.. "Token not in vocab: " token))))
     :encode_many (fn [self toks]
                    (new [(length toks)]
                         (icollect [_ tok (ipairs toks)] (self:encode tok))
                         {:require_grad false}))
     :decode (fn [_ id] (. tokens id))
     :decode_many (fn [self ids]
                    (icollect [_ id (ipairs ids.data)] (self:decode id)))
     :one_hot (fn [self token]
                (let [id (self:encode token)
                      size (length tokens)]
                  (new [size]
                       (fcollect [i 1 size] (if (= i id) 1 0))
                       {:require_grad false})))
     :one_hot_many (fn [self toks]
                     (let [size (length tokens)
                           rows (length toks)
                           data (fcollect [_ 1 (* rows size)] 0)]
                       (each [row token (ipairs toks)]
                         (let [id (self:encode token)
                               offset (+ (* (- row 1) size) id)]
                           (tset data offset 1)))
                       (new [rows size] data {:require_grad false})))}))

; Plain functional constructor: `(make tokens)` returns the vocabulary record.
{: make}
