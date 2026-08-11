(local tensor (require :fnl.tensor))
(local luassert (require :luassert))

(local tensor-mt (getmetatable (tensor.scalar 0)))
(luassert:add_formatter
  (fn [value]
    (when (= (getmetatable value) tensor-mt)
      (tostring value))))
