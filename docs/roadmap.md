# qnet Implementation Roadmap

> Project status as of July 2026

---

## Phase 1 — Tensor Infrastructure & Striding Logic

**Status:** ✅ Complete

- [x] Contiguous N-dimensional storage (`std::vector<float>`)
- [x] Stride-based index calculation: `index = offset + Σ(coord_i × stride_i)`
- [x] Row-major stride computation
- [x] Deep copy (clone) vs. zero-copy view semantics (slice, view)
- [x] Gradient buffer allocation (lazy)
- [x] Bounds-checked element access via `at(indices)`
- [x] `shape_string()` and `print_data()` utilities

**Files:**
- `include/qnet/tensor.hpp`
- `src/tensor.cpp`

---

## Phase 2 — OpenBLAS Mapping Layer

**Status:** ✅ Complete

- [x] `cblas_sgemm` C++ wrapper in `cosq::qnet::blas` namespace
- [x] Row-major GEMM with transpose support
- [x] `Ops::matmul` uses BLAS sgemm instead of Eigen
- [x] `Ops::matmul_backward` uses BLAS sgemm for gradient propagation
- [x] F32 precision throughout

**Files:**
- `include/qnet/blas.hpp`
- `src/blas.cpp`

---

## Phase 3 — Autograd DAG Engine

**Status:** ✅ Complete

- [x] Node graph with `OpType` enum (INPUT, PARAMETER, MATMUL, ADD, RELU, SIGMOID, SOFTMAX)
- [x] Topological sort forward pass execution
- [x] Reverse topological sort backward pass
- [x] Gradient accumulation across multiple uses
- [x] `Graph::variable()` / `Graph::parameter()` for inputs
- [x] `Graph::forward(output_node)` / `Graph::backward(output_node)`
- [x] `Graph::zero_grad()` to reset gradients

**Files:**
- `include/qnet/node.hpp`
- `include/qnet/graph.hpp`
- `src/graph.cpp`

---

## Phase 4 — Deep Learning Block Expansion

**Status:** ✅ Complete

- [x] `Ops::relu` — element-wise ReLU
- [x] `Ops::sigmoid` — element-wise sigmoid
- [x] `Ops::softmax` — batch softmax (stable via max subtraction)
- [x] `Ops::conv2d` — NCHW convolution with padding/stride
- [x] `Ops::embedding` — lookup table embedding
- [x] `Ops::add` / `Ops::mul` — element-wise arithmetic

**Files:**
- `include/qnet/ops.hpp`
- `src/ops.cpp`

---

## Phase 5 — Safetensors / Weight Ingestion

**Status:** ✅ Complete

- [x] SafeTensors binary format parser (header: JSON, body: raw floats)
- [x] 8-byte header size + JSON header parsing
- [x] F32 dtype support
- [x] Multi-tensor file support
- [x] Named tensor lookup with `load_tensor(name)`
- [x] Bounds checking and error reporting

**Format supported:**
```
[8 bytes: header_size (LE u64)]
[header_size bytes: JSON header]
[remaining: raw float data]
```

**Files:**
- `include/qnet/safetensors.hpp`
- `src/safetensors.cpp`

---

## Phase 6 — Portability Layers

**Status:** 🔧 Implemented (requires external SDKs to build)

- [x] **pybind11 module** (`src/bindings/python.cpp`):
  - Tensor construction, access, cloning
  - Graph building (variable, parameter, matmul, add, relu, sigmoid, softmax)
  - Forward and backward pass
  - SafeTensors reader
- [x] **Emscripten WASM target** (`cmake/emscripten.cmake`):
  - Static library build for WebAssembly
  - Memory growth enabled
  - LTO + O3 optimization flags

**Build notes:**
- Python bindings: `cmake -DBUILD_PYTHON_BINDINGS=ON` (requires pybind11)
- WASM: `emcmake cmake -B build_wasm -DCMAKE_TOOLCHAIN_FILE=<emsdk>/cmake/Modules/Platform/Emscripten.cmake`

**Files:**
- `src/bindings/python.cpp`
- `cmake/emscripten.cmake`

---

## Test Coverage

| Test suite | File | Status |
|---|---|---|
| Tensor construction, access, slicing, views, clone, grad | `tests/test_tensor.cpp` | ✅ 7 tests |
| Graph forward + backward pass | `tests/test_graph.cpp` | ✅ 2 tests |
| SafeTensors read + multi-tensor | `tests/test_safetensors.cpp` | ✅ 2 tests |

**Total:** 11 tests, all passing.

---

## Project Structure

```
qnet/
├── CMakeLists.txt              # Build system
├── vcpkg.json                  # Manifest: openblas, eigen3
├── cmake/
│   └── emscripten.cmake        # WASM toolchain helper
├── include/qnet/
│   ├── tensor.hpp              # ND strided tensor
│   ├── blas.hpp                # OpenBLAS sgemm wrapper
│   ├── node.hpp                # Graph node definition
│   ├── graph.hpp               # Computation graph
│   ├── ops.hpp                 # Math operations
│   └── safetensors.hpp         # SafeTensors reader
├── src/
│   ├── tensor.cpp
│   ├── blas.cpp
│   ├── graph.cpp
│   ├── ops.cpp
│   ├── safetensors.cpp
│   └── bindings/
│       └── python.cpp          # pybind11 module
├── tests/
│   ├── CMakeLists.txt
│   ├── test_tensor.cpp
│   ├── test_graph.cpp
│   └── test_safetensors.cpp
└── docs/
    └── roadmap.md              # This file
```
