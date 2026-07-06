# `qnet`

Lightweight C++17 neural network graph library for CPU training and inference.

`qnet` is built around a static computation graph with reverse-mode autograd, OpenBLAS-backed matrix math, and a small training stack for minibatch iteration, validation, checkpoints, and simple metrics.

---

## Current Feature Set

### Core runtime

- Strided `Tensor` storage with slicing, views, cloning, and lazy gradients
- Static `Graph` execution with forward and backward passes
- Thread-pool-backed parallel execution across graph levels
- `float`-only CPU runtime

### Ops

- `matmul`
- `add` with same-rank broadcasting
- `mul`
- `relu`
- `sigmoid`
- `softmax`
- `conv2d` via im2col + GEMM
- `embedding`
- `dropout`
- `layer_norm`

All of the above have backward implementations in the graph runtime.

### Losses

- Cross-entropy
- Mean squared error
- Binary cross-entropy

### Layers

- `Linear`
- `Conv2d`
- `Embedding`
- `Dropout`
- `LayerNorm`
- `Sequential`
- `Model`

### Optimizers

- `SGD`
  - momentum
  - Nesterov
  - weight decay
- `Adam`
- `AdamW`

### Training utilities

- `Dataset` interface
- `TensorDataset`
- `DataLoader`
- `Trainer`
  - minibatch training
  - validation passes
  - epoch callbacks
  - early stopping
  - best-checkpoint tracking and restore
  - classification accuracy metric for cross-entropy workflows

### Serialization

- Full graph save/load via `.qnet`
- Parameter-only save/load
- Safetensors import/export
- Optimizer state save/load

### Build and packaging

- CMake build
- install/export rules for `find_package(qnet)`
- CPack packaging
- GitHub Actions CI for Linux, macOS, and Windows

### Python bindings

There is an optional `pybind11` module, but it currently exposes only a subset of the C++ API:

- `Tensor`
- `Graph`
- `Node`
- `SGD`, `Adam`, `AdamW`
- `SafeTensorsReader`

It does not yet expose the newer layer, data, trainer, or checkpointing surfaces.

---

## Architecture

```text
[ Tensor ]
    |
    v
[ Static Graph + Nodes ]
    |
    +--> forward execution
    +--> reverse-mode autograd
    |
    v
[ CPU math backend ]
    +--> OpenBLAS GEMM
    +--> SIMD elementwise ops
    |
    v
[ Layers / Optimizers / Trainer ]
```

The graph is explicit and static. You build a DAG of nodes once, then reuse input and target nodes across minibatches during training.

---

## Dependencies

- C++17 compiler
- CMake 3.15+
- OpenBLAS
- Eigen3

Optional:

- `pybind11` for Python bindings

---

## Build

### macOS (Homebrew)

```bash
brew install openblas eigen

cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenBLAS_DIR="$(brew --prefix openblas)/lib/cmake/openblas" \
  -DEigen3_DIR="$(brew --prefix eigen)/share/eigen3/cmake"

cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

### vcpkg

```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

cmake --build build --config Release --parallel
ctest --test-dir build --output-on-failure -C Release
```

### Python bindings

```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPython3_EXECUTABLE="$(which python3)"

cmake --build build --parallel
```

---

## Usage

### Basic graph

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
    std::cout << y->output.data()[0] << "\n";

    graph.backward(y);
    std::cout << w->gradient.data()[0] << "\n";
}
```

### Layer-based model

```cpp
#include <qnet/graph.hpp>
#include <qnet/layer.hpp>

using namespace cosq::qnet;

int main() {
    Graph graph;

    Model model(graph);
    model.add(std::make_unique<Linear>(graph, 2, 8, true));
    model.add(std::make_unique<Linear>(graph, 8, 1, true));

    auto input = graph.variable(Tensor({1, 2}, {0.0f, 0.0f}));
    auto pred = model.forward(input);

    graph.forward(pred);
}
```

### Minibatch training with `Trainer`

```cpp
#include <qnet/data.hpp>
#include <qnet/graph.hpp>
#include <qnet/layer.hpp>
#include <qnet/optimizer.hpp>
#include <qnet/trainer.hpp>

using namespace cosq::qnet;

int main() {
    Graph graph;

    Model model(graph);
    model.add(std::make_unique<Linear>(graph, 2, 8, true));
    model.add(std::make_unique<Linear>(graph, 8, 1, true));

    auto input = graph.variable(Tensor({1, 2}, {0.0f, 0.0f}));
    auto target = graph.variable(Tensor({1, 1}, {0.0f}));
    auto pred = model.forward(input);
    auto loss = graph.mse_loss(pred, target);

    SGD optimizer(model.parameters(), 0.01f, 0.9f);
    Trainer trainer(graph, optimizer, input, target, loss);

    Tensor inputs({4, 2}, {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    });

    Tensor targets({4, 1}, {
        0.0f,
        1.0f,
        1.0f,
        2.0f
    });

    TensorDataset dataset(inputs, targets);
    DataLoader loader(dataset, 2, true, 1234u);

    auto history = trainer.fit(loader, 20);
    return history.empty() ? 1 : 0;
}
```

### Validation and early stopping

```cpp
FitOptions options;
options.validation_loader = &val_loader;
options.early_stopping_patience = 5;
options.early_stopping_min_delta = 1e-4f;
options.best_parameter_path = "best.params";
options.best_optimizer_state_path = "best.opt";
options.restore_best_checkpoint = true;

auto history = trainer.fit(train_loader, 100, options);
```

### Classification accuracy metric

Accuracy reporting is currently supported for trainers whose loss node is `cross_entropy_loss`.

```cpp
FitOptions options;
options.validation_loader = &val_loader;
options.metric = MetricType::CLASSIFICATION_ACCURACY;

auto history = trainer.fit(train_loader, 10, options);
float accuracy = trainer.evaluate_accuracy(val_loader);
```

### Optimizers

```cpp
auto params = model.parameters();

SGD sgd(params, 0.01f, 0.9f, 1e-4f, true);
Adam adam(params, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f);
AdamW adamw(params, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f);
```

### Serialization and checkpoints

```cpp
#include <qnet/serializer.hpp>

// Full graph
GraphSerializer::save(graph, "model.qnet");
Graph loaded = GraphSerializer::load("model.qnet");

// Parameters only
GraphSerializer::save_parameters(graph, "model.params");
GraphSerializer::load_parameters(graph, "model.params");

// Safetensors
GraphSerializer::export_safetensors(graph, "weights.safetensors");
GraphSerializer::import_safetensors(graph, "weights.safetensors");

// Trainer checkpoint
trainer.save_checkpoint("best.params", "best.opt");
trainer.load_checkpoint("best.params", "best.opt");
```

---

## Project Structure

```text
include/qnet/
├── tensor.hpp
├── blas.hpp
├── simd.hpp
├── node.hpp
├── graph.hpp
├── ops.hpp
├── layer.hpp
├── optimizer.hpp
├── data.hpp
├── trainer.hpp
├── safetensors.hpp
├── serializer.hpp
└── thread_pool.hpp

src/
├── tensor.cpp
├── blas.cpp
├── graph.cpp
├── ops.cpp
├── layer.cpp
├── optimizer.cpp
├── data.cpp
├── trainer.cpp
├── safetensors.cpp
├── serializer.cpp
├── thread_pool.cpp
└── bindings/python.cpp

tests/
├── test_tensor.cpp
├── test_graph.cpp
├── test_safetensors.cpp
├── test_conv.cpp
├── test_loss.cpp
├── test_backward.cpp
├── test_optimizer.cpp
├── test_serializer.cpp
├── test_layer.cpp
└── test_trainer.cpp
```

---

## Test Coverage

The unit suite currently covers:

- tensor construction, indexing, slicing, views, cloning, and gradients
- graph forward/backward execution
- broadcasted `add` forward and backward
- convolution, embedding, dropout, and layer norm behavior
- loss forward/backward paths
- optimizer update rules and optimizer state persistence
- graph serialization and safetensors
- dataloaders, trainer fit/evaluate flows, validation, callbacks, early stopping, checkpoints, and classification accuracy

Run everything with:

```bash
ctest --test-dir build --output-on-failure
```

---

## Limitations

- CPU-only runtime
- `float`-only tensors
- limited broadcasting support; `add` currently supports same-rank broadcasting
- Python bindings expose only part of the C++ API
- WASM support exists as a helper build path, not a first-class tested target

---

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the current implementation roadmap and next priorities.

---

## License

MIT. See [LICENSE](LICENSE).
