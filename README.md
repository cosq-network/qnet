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
- **Loss Functions** — Cross-entropy, MSE, Binary cross-entropy with numerically stable backward
- **Optimizers** — SGD (momentum, Nesterov, weight decay), Adam, AdamW
- **Thread-Pool Parallelism** — Multi-threaded graph execution (independent nodes run in parallel)
- **Safetensors** — Read/write HuggingFace-compatible weight format
- **Graph Serialization** — Save/load full graph structure + weights to `.qnet` binary format
- **Python Bindings** — pybind11 module exposing graph, ops, tensors, and optimizers
- **Zero-Bloat Core** — Minimal dependencies (OpenBLAS + Eigen3)

---

## Architecture

```
              [ Graph Builder ]
                     |  (Topological Sort)
                     ▼
           [ Ahead-of-Time Compiled Graph ]
                     |
    ┌────────────────┴────────────────┐
    ▼                                 ▼
[ Forward Pass Engine ]      [ Autograd Engine (Backward) ]
    |                                 |
    ├── Matmul, Add, Mul              ├── Gradient accumulation
    ├── ReLU, Sigmoid, Softmax        ├── Per-op backward dispatch
    ├── Conv2D (im2col+GEMM)          └── Loss backward (CE, MSE, BCE)
    ├── Embedding lookup
    └── Loss (CE, MSE, BCE)
                     |
                     ▼
         [ Contiguous Tensor Storage ]
              |  (Hardware Mapping)
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

# Run all tests
ctest --test-dir build
```

### vcpkg

```bash
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

### Python Bindings

```bash
cmake -B build -S . -DBUILD_PYTHON_BINDINGS=ON -DPython3_EXECUTABLE=$(which python3)
cmake --build build
```

---

## Usage

### Basic Graph (Forward + Backward)

```cpp
#include <qnet/graph.hpp>
#include <iostream>

using namespace cosq::qnet;

int main() {
    Graph graph;

    auto x = graph.variable(Tensor({1, 3}, {1.0f, 2.0f, 3.0f}));
    auto w = graph.parameter(Tensor({3, 1}, {0.5f, 1.0f, 1.5f}));
    auto y = graph.matmul(x, w);

    graph.forward(y);
    std::cout << "Output: " << y->output.data()[0] << "\n";  // 7.0

    graph.backward(y);
    std::cout << "dL/dw: " << w->gradient.data()[0] << "\n";  // 1.0

    return 0;
}
```

### Training Loop (with Loss + Optimizer)

```cpp
#include <qnet/graph.hpp>
#include <qnet/optimizer.hpp>

using namespace cosq::qnet;

int main() {
    Graph graph;

    // Model: x → matmul(w) → output
    auto x = graph.variable(Tensor({1, 2}, {1.0f, 2.0f}));
    auto w = graph.parameter(Tensor({2, 1}, {0.1f, 0.2f}));
    auto pred = graph.matmul(x, w);

    // Target
    auto t = graph.variable(Tensor({1}, {1.0f}));

    // Loss: MSE(pred, target)
    auto loss = graph.mse_loss(pred, t);

    // Optimizer
    SGD optimizer(graph.parameters(), /*lr=*/0.01f);

    for (int epoch = 0; epoch < 100; ++epoch) {
        graph.forward(loss);
        graph.backward(loss);
        optimizer.step();
        optimizer.zero_grad();

        if (epoch % 10 == 0)
            std::cout << "Epoch " << epoch << " loss=" << loss->output.data()[0] << "\n";
    }

    return 0;
}
```

### Loss Functions

```cpp
// Cross-entropy (classification)
auto logits = graph.variable(Tensor({2, 3}, /*...*/));
auto targets = graph.variable(Tensor({2}, {0.0f, 2.0f}));
auto ce = graph.cross_entropy_loss(logits, targets);

// MSE (regression)
auto mse = graph.mse_loss(prediction, target);

// Binary cross-entropy
auto bce = graph.binary_cross_entropy(prediction, target);
```

### Optimizers

```cpp
auto params = graph.parameters();

// SGD with momentum + weight_decay + Nesterov
SGD sgd(params, 0.01f, /*momentum=*/0.9f, /*weight_decay=*/1e-4f, /*nesterov=*/true);

// Adam
Adam adam(params, 1e-3f, 0.9f, 0.999f, 1e-8f, /*weight_decay=*/0.0f);

// AdamW (decoupled weight decay)
AdamW adamw(params, 1e-3f, 0.9f, 0.999f, 1e-8f, /*weight_decay=*/0.01f);
```

Shape mismatch or empty gradients are handled explicitly:
- Empty gradient (after `zero_grad`) → parameter is skipped (no-op)
- Non-empty gradient with mismatched shape → `std::logic_error` thrown

### Gradient Accumulation

```cpp
graph.forward(loss);
graph.backward(loss);  // grad += dL/dw
graph.backward(loss);  // grad += dL/dw (accumulated)
optimizer.step();      // update with 2x gradient
optimizer.zero_grad(); // reset
```

### Conv2D

```cpp
#include <qnet/ops.hpp>

// NCHW format
Tensor input({1, 3, 32, 32});
Tensor kernel({16, 3, 3, 3});  // K × C × KH × KW

// Standalone op
Tensor result = Ops::conv2d(input, kernel, /*stride=*/1, 1, /*pad=*/0, 0);

// Via graph (with autograd)
auto conv = graph.conv2d(x, k, 1, 1, 0, 0);
graph.forward(conv);
graph.backward(conv);
```

### Embedding

```cpp
Tensor weight({1000, 256});   // vocab_size × embedding_dim
Tensor indices({4}, {5, 42, 7, 99});

Tensor result = Ops::embedding(weight, indices);  // shape {4, 256}

// Via graph
auto emb = graph.embedding(weight_node, indices_node);
```

### Save / Load

```cpp
// Save graph structure + weights
GraphSerializer::save(graph, "model.qnet");

// Load back
Graph loaded = GraphSerializer::load("model.qnet");

// Export just weights as safetensors (HuggingFace-compatible)
GraphSerializer::export_safetensors(graph, "weights.safetensors");

// Load weights from safetensors into an existing graph
GraphSerializer::import_safetensors(graph, "weights.safetensors");

// Read safetensors directly
SafeTensorsReader reader("weights.safetensors");
Tensor w1 = reader.load_tensor("layer1.weight");
```

---

## Project Structure

```
include/qnet/
├── tensor.hpp        # ND strided tensor with gradient buffer
├── blas.hpp          # OpenBLAS GEMM wrapper (sgemm, batched sgemm)
├── simd.hpp          # SIMD ops: relu, add, mul, sigmoid (NEON/AVX/SSE)
├── node.hpp          # Graph node definition (OpType enum, input/output/gradient)
├── graph.hpp         # Computation graph + thread pool + autograd
├── ops.hpp           # Math operations: forward + backward for all ops
├── layer.hpp         # Layer abstractions (Linear, Conv2d, Embedding, Dropout, LayerNorm, Sequential, Model)
├── optimizer.hpp     # SGD, Adam, AdamW optimizers
├── safetensors.hpp   # SafeTensors reader/writer
├── serializer.hpp    # Graph save/load + SafeTensors writer
└── thread_pool.hpp   # Multi-threaded task pool

src/
├── tensor.cpp
├── blas.cpp
├── graph.cpp
├── ops.cpp
├── layer.cpp
├── optimizer.cpp
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
├── test_loss.cpp
├── test_backward.cpp
├── test_optimizer.cpp
├── test_serializer.cpp
└── test_layer.cpp

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
| `Tensor()` | Default empty tensor |
| `ndim()` / `shape()` / `size()` / `strides()` | Shape queries |
| `at({i, j, ...})` | Bounds-checked element access (read/write) |
| `data()` | Raw pointer to underlying buffer |
| `slice(dim, start, end)` | Zero-copy strided view |
| `view(new_shape)` | Reshape (zero-copy, same data, size must match) |
| `clone()` | Deep copy (always contiguous) |
| `is_view()` | Whether tensor shares data with another |
| `bytes()` | Total bytes (`size() * sizeof(float)`) |
| `has_grad()` / `grad()` / `zero_grad()` | Gradient buffer access |

### Operations (`Ops::`)

| Op | Forward | Backward |
|---|---|---|
| `matmul(a, b)` | Matrix multiply (sgemm) | `matmul_backward` |
| `add(a, b)` | Element-wise (SIMD) | `add_backward` |
| `mul(a, b)` | Element-wise (SIMD) | `mul_backward` |
| `relu(x)` | ReLU activation (SIMD) | `relu_backward` |
| `sigmoid(x)` | Sigmoid activation (SIMD) | `sigmoid_backward` |
| `softmax(x)` | Batch softmax | `softmax_backward` |
| `cross_entropy_loss(l, t)` | Cross-entropy (log-sum-exp) | `cross_entropy_backward` |
| `mse_loss(p, t)` | Mean squared error | `mse_backward` |
| `binary_cross_entropy(p, t)` | Binary cross-entropy | `bce_backward` |
| `conv2d(x, k, ...)` | 2D conv (im2col+GEMM) | `conv2d_backward` |
| `embedding(w, idx)` | Lookup table | `embedding_backward` |
| `dropout(x, rate, mask)` | Dropout (training-time masking) | `dropout_backward` |
| `layer_norm(x, g, b, eps)` | Layer normalization | `layer_norm_backward` |

### Graph Builder

| Method | Description |
|---|---|
| `Graph(nthreads)` | Create graph (0 = single-threaded) |
| `variable(tensor)` | Input node (no gradient computed) |
| `parameter(tensor)` | Trainable weight node |
| `matmul(a, b)` | Matrix multiply |
| `add(a, b)` / `mul(a, b)` | Element-wise |
| `relu(x)` / `sigmoid(x)` / `softmax(x)` | Activations |
| `cross_entropy_loss(l, t)` / `mse_loss(p, t)` / `binary_cross_entropy(p, t)` | Loss |
| `conv2d(x, k, stride, pad)` | 2D convolution |
| `embedding(w, idx)` | Embedding lookup |
| `dropout(x, rate)` | Dropout (training-time masking with scaling) |
| `layer_norm(x, gamma, beta, eps)` | Layer normalization (over last dim) |
| `forward(node)` | Execute forward pass (topological order, parallel) |
| `backward(node)` | Execute backward pass (reverse topological order) |
| `zero_grad()` | Reset all parameter gradients to empty |
| `parameters()` | Get all PARAMETER nodes (for optimizers) |
| `nodes()` | Get all nodes in the graph |
| `add_node(node)` | Insert an externally-created node |

### Optimizers

| Class | Algorithm | Key Parameters |
|---|---|---|
| `SGD` | SGD + momentum + Nesterov + weight decay | `lr`, `momentum`, `weight_decay`, `nesterov` |
| `Adam` | Adam (coupled weight decay) | `lr`, `beta1`, `beta2`, `eps`, `weight_decay` |
| `AdamW` | AdamW (decoupled weight decay) | `lr`, `beta1`, `beta2`, `eps`, `weight_decay` |

All optimizers share:
| Method | Description |
|---|---|
| `step()` | Update all parameters from their gradients |
| `zero_grad()` | Clear all parameter gradients |
| `parameters()` | View the parameter list |

### Serialization (`GraphSerializer::`)

| Method | File format | Description |
|---|---|---|
| `save(graph, path)` | `.qnet` | Binary graph + weights |
| `load(path)` | `.qnet` | Full graph reconstruction |
| `export_safetensors(graph, path)` | `.safetensors` | Weights only (HF-compatible) |
| `import_safetensors(graph, path)` | `.safetensors` | Load weights by node name |

---

## Test Suite

```
test_tensor      — construction, access, slice, view, clone, grad, ops
test_graph       — forward/backward, activations, add, mul, multi-output graphs
test_safetensors — single/multi-tensor safe tensors read
test_conv        — im2col conv2d, multi-channel, stride/pad, graph backward
test_loss        — CE/MSE/BCE forward, backward (analytical + numerical), error paths
test_backward    — per-op backward: matmul, relu, add, mul, sigmoid, softmax,
                   conv2d, embedding + numerical gradient checks + gradient accumulation
test_optimizer   — SGD (momentum, Nesterov, weight decay), Adam (multi-step),
                   AdamW (multi-step, decoupled WD), shape mismatch errors
test_serializer  — save/load roundtrip, safetensors export
test_layer       — Linear/Conv2d/Embedding forward/backward, Dropout train/eval,
                   LayerNorm forward/backward, Sequential, Model, end-to-end training
```

---

## Roadmap

See [`docs/roadmap.md`](docs/roadmap.md) for detailed implementation status.

- Phase 1: Tensor Infrastructure — ✅ Complete
- Phase 2: OpenBLAS Mapping — ✅ Complete
- Phase 3: Autograd DAG Engine — ✅ Complete
- Phase 4: Deep Learning Blocks — ✅ Complete
- Phase 5: Safetensors — ✅ Complete
- Phase 6: Graph Serialization — ✅ Complete
- Phase 7: Portability & Build — ✅ Complete
- Phase 8: Complete Autograd Backward — ✅ Complete
- Phase 9: Loss Functions — ✅ Complete
- Phase 10: Optimizers — ✅ Mostly Complete
- Phase 11: Layer Abstractions — ✅ Complete

---

## License

MIT License — see [LICENSE](LICENSE) file. Developed by **COSQ Network Private Limited**.
