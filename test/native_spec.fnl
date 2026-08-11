(local native (require :tensor_native))
(local assert (require :luassert))

(describe "tensor_native"
  (fn []
    (it "imports the Rust module and returns its hello-world greeting"
      (fn []
        (assert.equal "Hello, world!" (native.hello))))))
