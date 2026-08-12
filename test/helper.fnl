(local tensor (require :fnl.tensor))
(local luassert (require :luassert))

(local tensor-mt (getmetatable (tensor.scalar 0)))
(luassert:add_formatter
  (fn [value]
    (when (= (getmetatable value) tensor-mt)
      (tostring value))))

; Native Storage is intentionally kept on tensors during the test. Only
; table-oriented assertions need a temporary view of the values.
(fn data-table [data]
  (if (= (type data) "userdata")
      (data:to_table)
      data))

(fn assert-data [expected actual]
  (local actual (data-table actual))
  (luassert.equal (length expected) (length actual))
  (for [i 1 (length expected)]
    (let [expected-value (. expected i)
          actual-value (. actual i)
          tolerance (* 0.00001 (math.max 1 (math.abs expected-value)))]
      (luassert.near expected-value actual-value tolerance))))

{:data-table data-table
 :assert-data assert-data}
