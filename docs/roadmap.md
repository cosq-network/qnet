# qnet Implementation Roadmap

> Project status as of July 2026 — building toward production AI model capability.

---

## ✅ Phase 1 — Tensor Infrastructure

**Status:** Complete

- [x] Contiguous N-dimensional storage (`std::vector<float>`)
- [x] Stride-based index calculation
- [x] Row-major stride computation
- [x] Deep copy (clone) vs. zero-copy view semantics (slice, view)
- [x] Gradient buffer allocation (lazy)
- [x] Bounds-checked element access via `at(indices)`
- [x] `shape_string()` and `print_data()` utilities

**Files:** `include/qnet/tensor.hpp`, `src/tensor.cpp`

---

## ✅ Phase 2 — OpenBLAS Mapping Layer

**Status:** Complete

- [x] `cblas_sgemm` C++ wrapper in `cosq::qnet::blas` namespace
- [x] Row-major GEMM with transpose support
- [x] `Ops::matmul` uses BLAS sgemm
- [x] `Ops::matmul_backward` uses BLAS sgemm
- [x] Batched GEMM (`sgemm_batch`) with loop fallback
- [x] F32 precision throughout

**Files:** `include/qnet/blas.hpp`, `src/blas.cpp`

---

## ✅ Phase 3 — Autograd DAG Engine

**Status:** Complete

- [x] Node graph with `OpType` enum (INPUT, PARAMETER, MATMUL, ADD, MUL, RELU, SIGMOID, SOFTMAX, CONV2D, EMBEDDING)
- [x] Topological sort forward pass
- [x] Reverse topological sort backward pass
- [x] Gradient accumulation across multiple uses
- [x] `Graph::variable()` / `Graph::parameter()`
- [x] `Graph::forward(output_node)` / `Graph::backward(output_node)`
- [x] `Graph::zero_grad()`
- [x] Thread-pool parallel forward execution (depth-grouped nodes)

**Files:** `include/qnet/node.hpp`, `include/qnet/graph.hpp`, `src/graph.cpp`

---

## ✅ Phase 4 — Ops & DL Blocks

**Status:** Complete

- [x] `Ops::relu` — element-wise ReLU
- [x] `Ops::sigmoid` — element-wise sigmoid
- [x] `Ops::softmax` — batch softmax (stable via max subtraction)
- [x] `Ops::conv2d` — im2col + GEMM convolution
- [x] `Ops::embedding` — lookup table
- [x] `Ops::add` / `Ops::mul` — element-wise arithmetic
- [x] `Ops::matmul` / `Ops::matmul_backward`
- [x] SIMD-accelerated paths: relu, add, mul, sigmoid (NEON/AVX/SSE2)

**Files:** `include/qnet/ops.hpp`, `src/ops.cpp`, `include/qnet/simd.hpp`

---

## ✅ Phase 5 — Safetensors I/O

**Status:** Complete

- [x] SafeTensors binary format parser (JSON header + raw floats)
- [x] Multi-tensor file support
- [x] Named tensor lookup
- [x] SafeTensorsWriter for weight export
- [x] Graph weight export/import via safetensors

**Files:** `include/qnet/safetensors.hpp`, `src/safetensors.cpp`, `include/qnet/serializer.hpp`, `src/serializer.cpp`

---

## ✅ Phase 6 — Graph Serialization

**Status:** Complete

- [x] Binary `.qnet` format (magic + version + per-node topology + weights + grad)
- [x] `GraphSerializer::save(graph, path)` / `GraphSerializer::load(path)`
- [x] `GraphSerializer::export_safetensors()` / `import_safetensors()`

**Files:** `include/qnet/serializer.hpp`, `src/serializer.cpp`

---

## ✅ Phase 7 — Portability & Build

**Status:** Complete

- [x] pybind11 module (`src/bindings/python.cpp`)
- [x] Emscripten WASM target (`cmake/emscripten.cmake`)
- [x] CMake install rules (lib + headers)
- [x] `find_package(qnet)` via `qnetConfig.cmake` + version file
- [x] CPack platform packages: DEB, RPM, DMG, NSIS, TGZ, ZIP
- [x] CI pipeline: macOS/Linux/Windows, vcpkg, test, package
- [x] `.gitignore` + MIT LICENSE

**Files:** `CMakeLists.txt`, `cmake/qnetConfig.cmake.in`, `.github/workflows/ci.yml`, `.gitignore`, `LICENSE`

---

## ✅ Phase 8 — Complete Autograd (Backward Pass)

**Status:** Complete

- [x] `Ops::relu_backward(dy, x)` — mask `dy` where `x > 0`
- [x] `Ops::add_backward(dy, a, b)` — equal-shape gradient split
- [x] `Ops::mul_backward(dy, a, b)` — `da = dy * b`, `db = dy * a`
- [x] `Ops::sigmoid_backward(dy, out)` — `dy * out * (1 - out)`
- [x] `Ops::softmax_backward(dy, out)` — Jacobian: `out * (dy - sum(out * dy, dim))`
- [x] `Ops::conv2d_backward(dy, input, kernel)` — im2col + GEMM weight/input gradients
- [x] `Ops::embedding_backward(dy, indices)` — gradient scatter to lookup rows
- [x] Wire all backward ops into `Graph::backward()` via `OpType` dispatch
- [x] Forward + backward for `MUL`, `CONV2D`, `EMBEDDING` in `execute_node`
- [x] `Graph::mul()`, `Graph::conv2d()`, `Graph::embedding()` builder methods
- [x] Numerical gradient check for every op

**Files:** `include/qnet/ops.hpp`, `src/ops.cpp`, `include/qnet/node.hpp`, `include/qnet/graph.hpp`, `src/graph.cpp`, `tests/test_backward.cpp`

---

## ✅ Phase 9 — Loss Functions

**Status:** Complete

- [x] `Ops::cross_entropy_loss(logits, targets)` — softmax + NLL combined (stable via log-sum-exp)
- [x] `Ops::cross_entropy_backward(logits, targets)` — `(softmax - one_hot) / N`
- [x] `Ops::mse_loss(pred, target)` — mean squared error
- [x] `Ops::mse_backward(pred, target)` — `2 * (pred - target) / N`
- [x] `Ops::binary_cross_entropy(pred, target)` — BCE with logits (numerically stable)
- [x] `Ops::bce_backward(pred, target)` — gradient
- [x] Unit tests with numerical gradient checks, error paths (out-of-range, shape mismatch, non-unit grad_output)
- [x] Graph-level loss backward (`cross_entropy_loss`, `mse_loss`, `binary_cross_entropy`)

**Files:** `include/qnet/ops.hpp`, `src/ops.cpp`, `tests/test_loss.cpp`

---

## ✅ Phase 10 — Optimizers

**Status:** Mostly Complete

- [x] `Optimizer` base class — `step()`, `zero_grad()`, parameter list, validation
- [x] `SGD` — weight decay, momentum, Nesterov
- [x] `Adam` — bias-corrected first/second moment estimates (coupled weight decay)
- [x] `AdamW` — decoupled weight decay
- [x] Unit tests: multi-step, zero-gradient, shape mismatch errors, weight decay
- [ ] Parameter group support (per-layer learning rates)
- [ ] Gradient clipping (norm & value)
- [ ] Learning rate schedulers (step, cosine, linear warmup)

**Files:** `include/qnet/optimizer.hpp`, `src/optimizer.cpp`, `tests/test_optimizer.cpp`

---

## ✅ Phase 11 — Layer Abstractions

**Status:** Complete

- [x] `Layer` base class — `forward()`, `parameters()`, `name`, `train()`/`eval()`
- [x] `Linear(in_features, out_features, bias)` — weight `[out, in]`, bias `[out]`
- [x] `Conv2d(in_c, out_c, kernel, stride, pad)` — wraps im2col conv
- [x] `Embedding(vocab_size, dim)` — lookup table wrapper
- [x] `LayerNorm(normalized_shape, eps, elementwise_affine)` — fused op with backward
- [x] `Dropout(rate)` — training-time masking + scaling (fused op with mask storage)
- [x] `Sequential` — ordered container, forward chains layers
- [x] `Model` — parameter registration, train/eval mode
- [x] Layer unit tests: forward, backward, parameter count, train/eval mode
- [ ] ~~`BatchNorm1d` / `BatchNorm2d`~~ — deferred (requires running statistics)

---

## 🔧 Phase 12 — Transformer Building Blocks

**Priority: MEDIUM**

- [ ] `MultiHeadAttention(d_model, n_heads, causal)` — scaled dot-product
- [ ] `RoPE` (Rotary Position Embedding) — frequency-based rotation
- [ ] `GELU` activation (tanh approximation)
- [ ] `SiLU` / `Swish` activation
- [ ] `FeedForward(d_model, d_ff, activation)` — up/gate/down projections
- [ ] `TransformerBlock(d_model, n_heads, d_ff, dropout)` — attention + FFN
- [ ] `CausalMask` — upper-triangular attention mask
- [ ] KV-cache for incremental decoding
- [ ] Unit tests: attention output shape, causal mask, RoPE rotation

---

## 🔧 Phase 13 — Training Loop Infrastructure

**Priority: MEDIUM**

- [ ] `DataLoader` — batching, shuffling, iteration
- [ ] `Dataset` base class + `TensorDataset`
- [ ] `Trainer` — epoch loop, batch step, metric logging
- [ ] Checkpoint save/load (optimizer state + model weights)
- [ ] Early stopping
- [ ] Metrics: accuracy, perplexity
- [ ] Mixed precision training scaffolding (F32 fallback initially)

---

## 🔧 Phase 14 — Quantization & Reduced Precision

**Priority: LOW**

- [ ] FP16 tensor storage + conversion helpers
- [ ] INT8 quantization — per-tensor and per-channel
- [ ] Quantized matmul (INT8 x INT8 → INT32 → F32)
- [ ] Quantization-aware training (QAT) stubs
- [ ] Dynamic quantization for inference

---

## 🔧 Phase 15 — GPU / Accelerator Backend

**Priority: LOW**

- [ ] Metal Performance Shaders (MPS) backend for Apple Silicon
- [ ] CUDA backend stub with device memory management
- [ ] Kernel: matmul, relu, add, mul on GPU
- [ ] Automatic fallback: GPU if available, else CPU
- [ ] Host-device transfer utilities

---

## 🔧 Phase 16 — Model Zoo & Examples

**Priority: LOW**

- [ ] MLP on MNIST (training + inference)
- [ ] ConvNet on CIFAR-10
- [ ] GPT-style small language model
- [ ] Transformer encoder for text classification
- [ ] ImageNet-style classifier (ResNet-18)
- [ ] Inference-only example with pretrained safetensors weights
- [ ] Python notebook examples (via pybind11)

---

## 🔧 Phase 17 — ONNX / Interoperability

**Priority: LOW**

- [ ] ONNX export — graph → ONNX protobuf
- [ ] ONNX import — ONNX → qnet graph
- [ ] Supported ops: MatMul, Add, Relu, Sigmoid, Softmax, Conv, Gemm
- [ ] Shape inference on import

---

## 🔧 Phase 18 — Performance & Production Hardening

**Priority: LOW**

- [ ] Benchmark suite: matmul GFLOPs, conv throughput, end-to-end latency
- [ ] Memory profiling: tensor allocation tracking
- [ ] Fused kernels: `Linear+ReLU`, `Conv+BN+ReLU`
- [ ] Custom allocator (arena / slab)
- [ ] Thread pool affinity / pinning for NUMA
- [ ] CI benchmark tracking (prevent regressions)

---

## Summary

| Phase | Area | Priority | Status |
|---|---|---|---|---|
| 1–7 | Core, portability, build, CI | — | ✅ Done |
| 8 | Complete backward pass | HIGH | ✅ Done |
| 9 | Loss functions | HIGH | ✅ Done |
| 10 | Optimizers | HIGH | ✅ Mostly Done |
| 11 | Layer abstractions | HIGH | ✅ Done |
| 12 | Transformer blocks | MEDIUM | 🔧 Not started |
| 13 | Training loop | MEDIUM | 🔧 Not started |
| 14 | Quantization | LOW | 🔧 Not started |
| 15 | GPU backend | LOW | 🔧 Not started |
| 16 | Model zoo & examples | LOW | 🔧 Not started |
| 17 | ONNX | LOW | 🔧 Not started |
| 18 | Performance hardening | LOW | 🔧 Not started |

**Next concrete step:** Build Transformer blocks (Multi-Head Attention, RoPE, GELU, FeedForward) to enable language model implementations.
