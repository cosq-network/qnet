# qnet Roadmap

> Status snapshot: July 2026
>
> This roadmap is based on the current source tree, public headers, unit tests, and build system. It separates what is already implemented from what still needs engineering work to make qnet broadly usable as a training and inference runtime.

---

## Current State

`qnet` already has a functional CPU-first autograd core:

- Tensor storage with views, slicing, and lazy gradient buffers
- Static graph execution with forward and backward passes
- Core ops for matmul, add, mul, relu, sigmoid, softmax, conv2d, embedding
- Losses: cross-entropy, MSE, binary cross-entropy
- Optimizers: SGD, Adam, AdamW
- Higher-level layers: `Linear`, `Conv2d`, `Embedding`, `Dropout`, `LayerNorm`, `Sequential`, `Model`
- Serialization through `.qnet` and safetensors
- CMake packaging, install/export rules, and CI
- Unit tests covering tensors, graph execution, backward passes, losses, optimizers, serialization, safetensors, convolution, and layers

The project is no longer at the "can this work?" stage. The immediate work is around broadening operator coverage, improving ergonomics, and tightening non-core surfaces such as bindings and portability.

---

## Implemented Milestones

### 1. Tensor Core

**Status:** Complete

- [x] Contiguous `float` tensor storage
- [x] Row-major shape/stride bookkeeping
- [x] Bounds-checked indexed access
- [x] `slice()` and `view()` APIs
- [x] Deep copy via `clone()`
- [x] Lazy gradient allocation on `Tensor`
- [x] Basic introspection utilities (`shape_string()`, `print_data()`)

Primary files:
- `include/qnet/tensor.hpp`
- `src/tensor.cpp`

### 2. CPU Math Backend

**Status:** Complete for current operator set

- [x] OpenBLAS-backed GEMM wrapper
- [x] `Ops::matmul` and `Ops::matmul_backward`
- [x] SIMD paths for selected elementwise ops
- [x] CPU-only execution model across the library

Primary files:
- `include/qnet/blas.hpp`
- `include/qnet/simd.hpp`
- `src/blas.cpp`
- `src/ops.cpp`

### 3. Graph and Autograd Engine

**Status:** Complete for current op set

- [x] `Node` graph with explicit `OpType`
- [x] Graph builders for variables and parameters
- [x] Forward execution through topological traversal
- [x] Reverse-mode autograd with gradient accumulation
- [x] `Graph::zero_grad()`
- [x] Parameter enumeration via `Graph::parameters()`
- [x] Thread-pool-backed parallel forward execution by graph depth

Primary files:
- `include/qnet/node.hpp`
- `include/qnet/graph.hpp`
- `include/qnet/thread_pool.hpp`
- `src/graph.cpp`
- `src/thread_pool.cpp`

### 4. Primitive Ops

**Status:** Complete for the currently exposed API

- [x] Elementwise ops: `add`, `mul`, `relu`, `sigmoid`
- [x] Distribution op: `softmax`
- [x] Convolution: `conv2d`
- [x] Lookup op: `embedding`
- [x] Regularization op: `dropout`
- [x] Normalization op: `layer_norm`
- [x] Backward implementations for all of the above

Primary files:
- `include/qnet/ops.hpp`
- `src/ops.cpp`

### 5. Losses and Optimization

**Status:** Strong baseline, not feature-complete

- [x] Cross-entropy loss + backward
- [x] Mean squared error + backward
- [x] Binary cross-entropy + backward
- [x] Optimizer base class
- [x] SGD with momentum, Nesterov, and weight decay
- [x] Adam
- [x] AdamW

Still missing in this area:
- [ ] Parameter groups
- [ ] Gradient clipping
- [ ] Learning-rate schedulers
- [ ] Optimizer state checkpointing

Primary files:
- `include/qnet/optimizer.hpp`
- `src/optimizer.cpp`

### 6. Layer Abstractions

**Status:** Complete baseline

- [x] Abstract `Layer`
- [x] `Linear`
- [x] `Conv2d`
- [x] `Embedding`
- [x] `Dropout`
- [x] `LayerNorm`
- [x] `Sequential`
- [x] `Model`
- [x] Train/eval propagation

Primary files:
- `include/qnet/layer.hpp`
- `src/layer.cpp`

### 7. Serialization and Interchange

**Status:** Complete baseline

- [x] Safetensors reader
- [x] Safetensors writer
- [x] Graph weight export/import through safetensors
- [x] Binary `.qnet` graph save/load

Primary files:
- `include/qnet/safetensors.hpp`
- `include/qnet/serializer.hpp`
- `src/safetensors.cpp`
- `src/serializer.cpp`

### 8. Build, Packaging, and CI

**Status:** Present, with some surface-area gaps

- [x] CMake build for shared library and tests
- [x] Install/export rules for `find_package(qnet)`
- [x] CPack packaging setup
- [x] GitHub Actions CI across Linux, macOS, and Windows via vcpkg
- [x] Optional pybind11 module build
- [x] Emscripten helper CMake file

What is implemented but still narrow:
- [ ] Python bindings expose only part of the C++ API
- [ ] WASM support is not integrated into the main build and is not test-covered

Primary files:
- `CMakeLists.txt`
- `cmake/qnetConfig.cmake.in`
- `cmake/emscripten.cmake`
- `.github/workflows/ci.yml`
- `src/bindings/python.cpp`

---

## What The Tests Validate Today

The existing unit suite gives good confidence in the current CPU core:

- Tensor shape/indexing/view behavior
- Graph forward and backward execution
- Backward correctness for current primitive ops
- Loss behavior and error paths
- Optimizer update rules and validation
- Conv2d behavior
- Safetensors and graph serialization
- Layer wiring, train/eval behavior, and a simple end-to-end training loop

Gaps in coverage:

- [ ] Python bindings
- [ ] Packaging/install smoke tests
- [ ] WASM/Emscripten builds
- [ ] Cross-platform numerical parity checks beyond CI build/test pass
- [ ] Performance/regression benchmarks

---

## Near-Term Priorities

### Phase A. Make The Training Stack Practical

**Priority:** High

- [ ] `Dataset` abstraction
- [ ] `TensorDataset`
- [ ] `DataLoader` with batching and shuffling
- [ ] `Trainer` loop with metrics and hooks
- [ ] Checkpointing of model and optimizer state
- [ ] Basic metrics: accuracy, perplexity, loss tracking

Why this matters:
- The math core exists, but users still have to hand-wire training loops around the graph.

### Phase B. Fill Operator Gaps For Modern Models

**Priority:** High

- [ ] `GELU`
- [ ] `SiLU` / `Swish`
- [ ] `tanh`
- [ ] `MultiHeadAttention`
- [ ] Causal masking
- [ ] RoPE
- [ ] Feed-forward transformer block
- [ ] KV-cache support for decoding

Why this matters:
- Current layers are enough for MLP/CNN-style experiments, but not yet enough for serious transformer work.

### Phase C. Strengthen Optimizer Ergonomics

**Priority:** Medium

- [ ] Parameter groups
- [ ] Gradient clipping by value and norm
- [ ] LR schedulers: step, cosine, warmup
- [ ] Better state serialization for resume/restart flows

Why this matters:
- The optimizer math is there; the missing pieces are the controls expected in real training code.

### Phase D. Expand Python Bindings

**Priority:** Medium

- [ ] Bind `mul`, `conv2d`, `embedding`, `dropout`, `layer_norm`
- [ ] Bind layer abstractions and serialization helpers
- [ ] Improve tensor ergonomics for Python use
- [ ] Add Python-level tests

Why this matters:
- The current binding surface is enough for simple demos, but it lags the C++ API materially.

---

## Medium-Term Roadmap

### 1. Portability and Runtime Targets

**Priority:** Medium

- [ ] Integrate WASM builds into the primary CMake flow
- [ ] Add WASM-specific tests/examples
- [ ] Clarify supported feature subset for browser/runtime targets

### 2. Reduced Precision and Quantization

**Priority:** Medium

- [ ] FP16 tensor support
- [ ] INT8 quantization primitives
- [ ] Quantized matmul path
- [ ] Dynamic quantization for inference

### 3. Accelerator Backends

**Priority:** Low

- [ ] Apple Metal / MPS backend
- [ ] CUDA backend
- [ ] Backend abstraction for device selection and fallback

### 4. Model Import/Export

**Priority:** Low

- [ ] ONNX export
- [ ] ONNX import
- [ ] Shape inference and op compatibility layer

### 5. Examples and Model Zoo

**Priority:** Low

- [ ] MLP example
- [ ] ConvNet example
- [ ] Small GPT-style decoder example
- [ ] Inference examples with safetensors checkpoints

---

## Project Risks and Constraints

- The entire runtime is `float`-only today.
- Broadcasting semantics are limited; many ops still expect tightly matching shapes.
- The graph is static and explicit; dynamic-control-flow model authoring is not a goal right now.
- Python and WASM support exist as side surfaces, not first-class validated platforms yet.
- There is no benchmark harness yet, so performance claims are still qualitative.

---

## Recommended Next Sequence

If the goal is to move qnet from a solid kernel into a usable small-framework, the next order should be:

1. Build `Dataset` / `DataLoader` / `Trainer`.
2. Add transformer-critical ops: `GELU`, `SiLU`, attention, masks, RoPE.
3. Finish optimizer ergonomics: parameter groups, clipping, schedulers.
4. Expand and test Python bindings.
5. Add examples and checkpoints that exercise the whole stack.

That sequence compounds well: each step makes the next one easier to validate.
