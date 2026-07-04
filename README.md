# `qnet` — Lightweight C++ Neural Network Graph Framework

A minimal, high-performance, ahead-of-time (AOT) computation graph library. Provides core mathematical foundations for inference and training across diverse workloads — image classification, regression, text embeddings, and transformers.

`qnet` uses a static computation graph paradigm (similar to ONNX Runtime / GGML), optimized for embedding into edge applications, Python extensions (`pybind11`), or WebAssembly (`Emscripten`).

---

## Key Features

- **Contiguous N-Dimensional Tensors** — Stride-based indexing with zero-copy slicing and views
- **Static Computation Graph** — AOT-compiled forward + reverse-mode autograd
- **SIMD-Accelerated Ops** — NEON (Apple Silicon), AVX, SSE2 for element-wise operations
- **OpenBLAS GEMM** — Matrix multiplication via `cblas_sgemm` with batched-GEMM support
- **im2col + GEMM Conv2D** — Efficient convolution via patch extraction + BLAS matmul
- **Thread-Pool Parallelism** — Multi-threaded graph execution (independent nodes run in parallel)
- **Safetensors** — Read/write HuggingFace-compatible weight format
- **Graph Serialization** — Save/load full graph structure + weights to `.qnet` binary format
- **Zero-Bloat Core** — Minimal dependencies (OpenBLAS + Eigen3)

---

## Architecture

```
              [ Graph Builder ]
                     │  (Topological Sort)
                     ▼
           [ Ahead-of-Time Compiled Graph ]
                     │
    ┌────────────────┴────────────────┐
    ▼                                 ▼
[ Forward Pass Engine ]      [ Autograd Engine (Backward) ]
    │                                 │
    └────────────────┬────────────────┘
                     ▼
        [ Contiguous Tensor Storage ]
             │  (Hardware Mapping)
             ▼
     [ Accelerated Hardware (BLAS/SIMD) ]
```

1. **Storage Layer (`Tensor`)** — Raw `std::vector<float>` with stride-based indexing and lazy gradient allocation
2. **Graph Compiler (`Graph`)** — DAG execution via topological sort, parallel node dispatch
3. **Hardware Layer (`blas`, `simd`)** — OpenBLAS GEMM + platform-specific SIMD intrinsics

---

## Building

### Dependencies

- C++17 compiler
- [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS)
- [Eigen3](https://eigen.tuxfamily.org)
- CMake ≥ 3.15

### macOS (Homebrew)

```bash
brew install openblas eigen
```

### Build

```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenBLAS_DIR="$(brew --prefix openblas)/lib/cmake/openblas" \
  -DEigen3_DIR="$(brew --prefix eigen)/share/eigen3/cmake"

cmake --build build -j$(nproc)

# Run tests
./build/tests/test_tensor
./build/tests/test_graph
./build/tests/test_safetensors
./build/tests/test_conv
./build/tests/test_serializer
```

### vcpkg

```bash
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

---

## Usage

### Basic Graph

```cpp
#include <qnet/tensor.hpp>
#include <qnet/graph.hpp>
#include <iostream>

using namespace cosq::qnet;

int main() {
    Tensor X({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor W({4, 2}, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f});

    Graph graph(4);  // 4 worker threads

    auto input  = graph.variable(X);
    auto weight = graph.parameter(W);
    auto output = graph.matmul(input, weight);

    graph.forward(output);

    Tensor result = output->output;
    std::cout << "Output: " << result.shape_string() << "\n";
    result.print_data();

    return 0;
}
```

### Save / Load

```cpp
// Save graph structure + weights
GraphSerializer::save(graph, "model.qnet");

// Load back
Graph loaded = GraphSerializer::load("model.qnet");

// Export just weights as safetensors
GraphSerializer::export_safetensors(graph, "weights.safetensors");

// Load weights from safetensors
GraphSerializer::import_safetensors(graph, "weights.safetensors");
```

### Conv2D

```cpp
Tensor input({1, 3, 32, 32});  // NCHW
Tensor kernel({16, 3, 3, 3});  // K x C x KH x KW

Tensor result = Ops::conv2d(input, kernel, /*stride=*/1, 1, /*pad=*/0, 0);
// Uses im2col + sgemm internally
```

---

## Project Structure

```
include/qnet/
├── tensor.hpp        # ND strided tensor
├── blas.hpp          # OpenBLAS GEMM wrapper (sgemm, batched sgemm)
├── simd.hpp          # SIMD ops: relu, add, mul, sigmoid (NEON/AVX/SSE)
├── node.hpp          # Graph node definition
├── graph.hpp         # Computation graph + thread pool
├── ops.hpp           # Math operations
├── safetensors.hpp   # SafeTensors reader
├── serializer.hpp    # Graph save/load + SafeTensors writer
└── thread_pool.hpp   # Multi-threaded task pool

src/
├── tensor.cpp
├── blas.cpp
├── graph.cpp
├── ops.cpp
├── safetensors.cpp
├── serializer.cpp
├── thread_pool.cpp
└── bindings/
    └── python.cpp    # pybind11 module (requires BUILD_PYTHON_BINDINGS)

tests/
├── test_tensor.cpp
├── test_graph.cpp
├── test_safetensors.cpp
├── test_conv.cpp
└── test_serializer.cpp

cmake/
└── emscripten.cmake  # WASM build helper
```

---

## API Reference

### Tensor

| Method | Description |
|---|---|
| `Tensor(shape)` | Allocate tensor with given shape |
| `Tensor(shape, data)` | Initialize from vector |
| `at({i, j, ...})` | Bounds-checked element access |
| `slice(dim, start, end)` | Zero-copy view |
| `view(new_shape)` | Reshape (zero-copy, same data) |
| `clone()` | Deep copy |
| `grad()` / `zero_grad()` | Gradient buffer access |

### Operations (`Ops::`)

| Op | Description |
|---|---|
| `matmul(a, b)` | Matrix multiply (OpenBLAS sgemm) |
| `add(a, b)` / `mul(a, b)` | Element-wise (SIMD) |
| `relu(x)` / `sigmoid(x)` | Activation functions (SIMD) |
| `softmax(x)` | Batch softmax |
| `conv2d(x, k, ...)` | 2D convolution (im2col + GEMM) |
| `embedding(w, idx)` | Lookup table |

### Graph

| Method | Description |
|---|---|
| `Graph(nthreads)` | Create graph with thread pool |
| `variable(tensor)` | Input node |
| `parameter(tensor)` | Trainable weight node |
| `matmul(a, b)` / `add(a, b)` / etc. | Add operation node |
| `forward(output)` | Execute forward pass (parallel) |
| `backward(output)` | Execute backward pass |
| `zero_grad()` | Reset all gradients |
| `add_node(node)` | Insert existing node |

### Serialization (`GraphSerializer::`)

| Method | File format | Description |
|---|---|---|
| `save(graph, path)` | `.qnet` | Binary graph + weights |
| `load(path)` | `.qnet` | Full reconstruction |
| `export_safetensors(graph, path)` | `.safetensors` | Weights only (HF-compatible) |
| `import_safetensors(graph, path)` | `.safetensors` | Load weights by node name |

---

## Test Suite (16 tests)

```
test_tensor      — construction, access, slice, view, clone, grad, ops
test_graph       — forward pass, backward pass (autograd)
test_safetensors — single/multi-tensor read
test_conv        — im2col conv2d, multi-channel, thread pool integration
test_serializer  — save/load roundtrip, safetensors export
```

---

## Roadmap

See [`docs/roadmap.md`](docs/roadmap.md) for detailed implementation status.

- Phase 1: Tensor Infrastructure — ✅ Complete
- Phase 2: OpenBLAS Mapping — ✅ Complete
- Phase 3: Autograd DAG Engine — ✅ Complete
- Phase 4: Deep Learning Blocks — ✅ Complete
- Phase 5: Safetensors — ✅ Complete
- Phase 6: Portability (pybind11, WASM) — 🔧 Implemented (requires SDKs)

---

## License

MIT License — see [LICENSE](LICENSE) file. Developed by **COSQ Network Private Limited**.
