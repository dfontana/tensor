# Native CPU v1

Status: implementation, verification, and delivery complete

## Goal

Move tensor values into Lua-GC-owned Rust `f32` storage and accelerate the common contiguous CPU kernels with `fearless_simd`, while keeping tensor metadata, shapes, broadcasting policy, and autograd in Fennel.

## Non-goals

- GPU, Vulkan, Metal, or MLX support
- Generic backend, device, allocator, stream, or dtype abstractions
- Threading
- N-D strides, generalized broadcasting, batched matmul, or generalized gather
- Replacing Fennel tensor/autograd tables with Rust userdata

## Compatibility contract

A tensor remains a Fennel table whose `data` field is either native `Storage` or, when the native module is unavailable, a Lua table. Native `Storage` is fixed-size and supports the existing hot-path syntax:

- `#data`
- `data[i]`
- `data[i] = value`
- `ipairs(data)` and `pairs(data)`
- explicit `data:to_table()` for table-only APIs and tests

All writes narrow Lua numbers to finite or non-finite `f32`; finite overflow and nonnumeric values are errors. Lua GC is the sole lifetime owner: collecting userdata drops its `Vec<f32>`. There is no manual `free`, raw handle, registry, `Arc`, or custom allocator.

## Native API

Storage:

- `storage(values)` — copy a dense Lua sequence, or return existing Storage unchanged
- `storage_zeros(length)`
- `is_storage(value)`
- methods: `len`, `clone`, `to_table`, `fill`
- metamethods: 1-based fixed-size index/read/write, `__len`, `__pairs`, diagnostic `__tostring`

Kernels:

- `add_into(lhs, rhs, out)`, `sub_into(lhs, rhs, out)`, `mul_into(lhs, rhs, out)`
- `relu_into(input, out)`, `scale_into(input, scalar, out)`, scalar `pow_into(input, exponent, out)`
- `sum(input)`, `mean(input)`
- `matmul_into(lhs, rhs, m, k, n, out)` for row-major `m*k` by `k*n` matrices
- in-place `sgd_step(parameter, gradient, learning_rate)`

Kernel arguments are native `Storage` userdata. Dimensions are nonnegative Lua integers; all products and lengths are checked before writes. `matmul_into` also accepts the dimensions after `out` for direct-call compatibility.

Every output kernel validates before mutation and supports output/input aliasing correctly. SIMD is an internal implementation detail initialized once per Lua module. Scalar Rust functions are the reference implementation; SIMD handles complete lanes and scalar tails.

## Fennel boundary

Migrate exact-shape contiguous add/sub/mul, ReLU, scalar scale/pow, sum/mean, 2-D matmul, same-shape gradient accumulation, and SGD. Existing Fennel kernels remain the fallback for broadcasting, transpose, softmax, argmax, cross-entropy, gather/scatter, and specialized backward operations.

Unsupported layouts are selected explicitly; native runtime errors are not swallowed.

## Implementation checklist

### 0. Baseline

- [x] Baseline `cargo test` passes (1 test before native storage)
- [x] Record existing Fennel status: 139 pass, 1 scalar-transpose expectation failure, 9 deferred N-D errors
- [x] Keep deferred N-D work outside this v1 scope

### 1. Storage userdata

- [x] Add `Storage(Vec<f32>)` and strict conversion/index validation
- [x] Register storage constructors, methods, and metamethods
- [x] Test indexing, mutation, iteration, copying, conversion errors, and fixed size
- [x] Verify Rust formatting, clippy, and tests

Phase 1 is complete. The native CPU v1 remains in progress because Fennel integration is intentionally deferred to phase 3. The module-mode test gating keeps `cargo test` self-contained; it passes without a host Lua 5.5 link. Focused native Fennel tests load the release module directly.

### 2. Scalar and SIMD kernels

- [x] Pin `fearless_simd` 0.7
- [x] Implement scalar reference kernels
- [x] Add one-time module-initialization feature detection and dispatched SIMD kernels
- [x] Handle tails, empty inputs, shape/length errors, checked dimensions, and aliasing
- [x] Test scalar/SIMD parity and f32 tolerances across lane boundaries, odd lengths, reductions, matmul, and SGD

Phase 2 is complete. `pow_into` intentionally remains scalar; add/sub/mul, ReLU, scale, reductions, matmul, and SGD dispatch through the captured SIMD level with scalar tails and a scalar fallback. Native calls snapshot inputs before mutable output/parameter borrows, so output aliases and `sgd_step(parameter, parameter, learning_rate)` are defined.

### 3. Fennel integration

- [x] Coerce constructor data to Storage when native support is available
- [x] Preserve Lua-table fallback without the native module
- [x] Route only eligible operations to native kernels
- [x] Route same-shape gradient accumulation and SGD natively
- [x] Normalize table-only formatting and test comparisons
- [x] Keep tensor graph and autograd semantics unchanged

Phase 3 is complete. `fnl.tensor` and the kernel/optimizer modules use an
optional `pcall`-loaded `tensor_native`; `TENSOR_FORCE_NO_NATIVE=1` is only used
by the explicit fallback task. Constructors preserve existing Storage identity,
and only the documented exact-shape/rank-2 contiguous paths call native kernels.
Broadcasting, transpose, softmax, gather, and other complex paths remain in
Fennel, with their dense results coerced at the tensor boundary. Native runtime
errors are not converted into fallback execution.

The default native suite (`mise run test`) now reports 153 successes / 0
failures / 0 errors and covers the native, optimizer, tensor, and vocab specs.
The explicit fallback suite (`mise run test:fallback`) reports 2 successes / 0
failures / 0 errors. The opt-in deferred suite (`mise run test:nd`) reports 1
success / 1 failure / 9 errors: the nine errors are the unchanged deferred
N-D cases, and the remaining failure is the existing f32-precision assertion in
that opt-in contract suite.

The portable benchmark (`mise run benchmark`) compares direct native Storage
addition with an equivalent Lua-table loop. Observed output on the verification
machine (timings are informational; there is no threshold) was:

```text
length=1024 native_storage_add=0.000015s lua_table_add=0.000032s verified
length=1000003 native_storage_add=0.004741s lua_table_add=0.024688s verified
```

### 4. Verification and delivery

- [x] Run `cargo fmt --check`
- [x] Run clippy with warnings denied
- [x] Run Rust tests
- [x] Run the default native/optimizer/tensor/vocab suite (`mise run test`)
- [x] Run the opt-in deferred N-D suite (`mise run test:nd`; expected to fail)
- [x] Run focused native Fennel tests and compare direct native API behavior
- [x] Run explicit no-native fallback tests (`mise run test:fallback`)
- [x] Add and run a portable benchmark command (no fixed speed threshold)
- [x] Perform a correctness-only review against this document; reject scope expansion
- [x] Update this document and `ROADMAP.md`
- [x] Create a `jj` bookmark, push it, and open pull request #2 with `gh`

## Acceptance criteria

1. Tensor values use GC-owned native `f32` storage whenever the module is available.
2. Existing indexing, mutation, iteration, graph, and autograd behavior remains intact within the pre-existing supported scope.
3. Scalar and SIMD implementations agree within documented f32 tolerances, including tails and aliasing.
4. Complex or unsupported operations continue through Fennel rather than growing v1 abstractions.
5. No manual memory management is exposed to Lua.
6. The default native and explicit fallback suites pass; the opt-in deferred N-D suite continues to expose the documented nine deferred errors without being treated as v1 regressions.
