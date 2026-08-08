; Runnable demo of the library end to end. `main.fnl` is a library (requiring it
; must not train), so the executable entry point lives here.
(math.randomseed 42)
(let [m (require :fnl.main)
      losses (m.learn-xor 2000)]
  (print (.. "xor loss after 2000 steps: " (. losses (length losses)))))
