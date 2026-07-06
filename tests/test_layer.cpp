#include <qnet/layer.hpp>
#include <qnet/optimizer.hpp>

#include <cassert>
#include <cmath>
#include <iostream>

using namespace cosq::qnet;

static bool approx(float a, float b, float eps = 1e-4f) {
    return std::abs(a - b) < eps;
}

// ── Linear ──────────────────────────────────────────────────────────

static void test_linear_forward() {
    Graph graph;
    Linear linear(graph, 3, 2, true);

    auto x = graph.variable(Tensor({1, 3}, {1.0f, 2.0f, 3.0f}));
    auto y = linear.forward(x);

    graph.forward(y);
    assert(y->output.shape().size() == 2);
    assert(y->output.shape()[0] == 1);
    assert(y->output.shape()[1] == 2);

    auto params = linear.parameters();
    assert(params.size() == 2); // weight + bias

    std::cout << "  [PASS] test_linear_forward\n";
}

static void test_linear_no_bias() {
    Graph graph;
    Linear linear(graph, 3, 2, false);

    auto params = linear.parameters();
    assert(params.size() == 1);

    std::cout << "  [PASS] test_linear_no_bias\n";
}

static void test_linear_backward() {
    Graph graph;
    Linear linear(graph, 2, 1, true);

    auto x = graph.variable(Tensor({1, 2}, {1.0f, 2.0f}));
    auto target = graph.variable(Tensor({1, 1}, {1.0f}));
    auto y = linear.forward(x);
    auto loss = graph.mse_loss(y, target);

    graph.forward(loss);
    graph.backward(loss);

    for (auto& p : linear.parameters()) {
        assert(p->gradient.size() > 0);
    }

    std::cout << "  [PASS] test_linear_backward\n";
}

// ── Conv2d ──────────────────────────────────────────────────────────

static void test_conv2d_forward() {
    Graph graph;
    Conv2d conv(graph, 1, 2, 3, 1, 0, false);

    auto x = graph.variable(Tensor({1, 1, 5, 5}));
    auto y = conv.forward(x);

    graph.forward(y);
    assert(y->output.shape()[0] == 1);
    assert(y->output.shape()[1] == 2);
    assert(y->output.shape()[2] == 3);
    assert(y->output.shape()[3] == 3);

    auto params = conv.parameters();
    assert(params.size() == 1); // weight only

    std::cout << "  [PASS] test_conv2d_forward\n";
}

// ── Embedding ───────────────────────────────────────────────────────

static void test_embedding_forward() {
    Graph graph;
    Embedding emb(graph, 100, 16);

    auto x = graph.variable(Tensor({3}, {0.0f, 5.0f, 42.0f}));
    auto y = emb.forward(x);

    graph.forward(y);
    assert(y->output.shape()[0] == 3);
    assert(y->output.shape()[1] == 16);

    auto params = emb.parameters();
    assert(params.size() == 1);

    std::cout << "  [PASS] test_embedding_forward\n";
}

// ── Dropout ─────────────────────────────────────────────────────────

static void test_dropout_train_vs_eval() {
    Graph graph;
    Dropout drop(graph, 0.5f);

    auto x = graph.variable(Tensor({1, 100}, std::vector<float>(100, 2.0f)));
    // In eval mode, dropout is a no-op
    drop.train(false);
    auto y = drop.forward(x);
    graph.forward(y);

    // All values should pass through unchanged
    for (size_t i = 0; i < y->output.size(); ++i) {
        assert(approx(y->output.data()[i], 2.0f));
    }

    std::cout << "  [PASS] test_dropout_train_vs_eval\n";
}

static void test_dropout_rate_zero() {
    Graph graph;
    Dropout drop(graph, 0.0f);

    auto x = graph.variable(Tensor({1, 50}, std::vector<float>(50, 1.5f)));
    auto y = drop.forward(x);
    graph.forward(y);

    for (size_t i = 0; i < y->output.size(); ++i) {
        assert(approx(y->output.data()[i], 1.5f));
    }

    std::cout << "  [PASS] test_dropout_rate_zero\n";
}

// ── LayerNorm ───────────────────────────────────────────────────────

static void test_layer_norm_forward() {
    Graph graph;
    LayerNorm ln(graph, 4, 1e-5f, true);

    Tensor data({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto x = graph.variable(data);
    auto y = ln.forward(x);

    graph.forward(y);

    // With gamma=1, beta=0, output should be standardized
    float mean = 0.0f;
    for (size_t i = 0; i < 4; ++i) mean += y->output.data()[i];
    mean /= 4.0f;
    assert(approx(mean, 0.0f, 1e-5f));

    float var = 0.0f;
    for (size_t i = 0; i < 4; ++i) {
        float diff = y->output.data()[i] - mean;
        var += diff * diff;
    }
    var /= 4.0f;
    assert(approx(var, 1.0f, 1e-4f));

    auto params = ln.parameters();
    assert(params.size() == 2);

    std::cout << "  [PASS] test_layer_norm_forward\n";
}

static void test_layer_norm_no_affine() {
    Graph graph;
    LayerNorm ln(graph, 3, 1e-5f, false);

    auto params = ln.parameters();
    assert(params.empty());

    std::cout << "  [PASS] test_layer_norm_no_affine\n";
}

static void test_layer_norm_backward() {
    Graph graph;
    LayerNorm ln(graph, 3, 1e-5f, true);

    auto x = graph.variable(Tensor({1, 3}, {1.0f, 2.0f, 3.0f}));
    auto target = graph.variable(Tensor({1, 3}, {0.0f, 0.0f, 0.0f}));
    auto y = ln.forward(x);
    auto loss = graph.mse_loss(y, target);

    graph.forward(loss);
    graph.backward(loss);

    assert(x->gradient.size() > 0);
    for (auto& p : ln.parameters()) {
        assert(p->gradient.size() > 0);
    }

    std::cout << "  [PASS] test_layer_norm_backward\n";
}

// ── Sequential ──────────────────────────────────────────────────────

static void test_sequential() {
    Graph graph;

    Sequential seq(graph);
    seq.add(std::make_unique<Linear>(graph, 4, 8, true));
    seq.add(std::make_unique<Linear>(graph, 8, 2, true));

    auto x = graph.variable(Tensor({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f}));
    auto y = seq.forward(x);

    graph.forward(y);
    assert(y->output.shape()[0] == 1);
    assert(y->output.shape()[1] == 2);
    assert(seq.size() == 2);

    auto params = seq.parameters();
    assert(params.size() == 4);

    std::cout << "  [PASS] test_sequential\n";
}

// ── Model ───────────────────────────────────────────────────────────

static void test_model_train_eval() {
    Graph graph;

    Model model(graph);
    model.add(std::make_unique<Dropout>(graph, 0.5f));
    model.add(std::make_unique<Linear>(graph, 4, 2));

    assert(model.is_training());

    model.eval();
    assert(!model.is_training());

    auto x = graph.variable(Tensor({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f}));
    auto y = model.forward(x);
    graph.forward(y);

    assert(y->output.shape()[0] == 1);
    assert(y->output.shape()[1] == 2);

    std::cout << "  [PASS] test_model_train_eval\n";
}

static void test_end_to_end_training() {
    Graph graph;

    // Model: Linear(2 → 8) → Linear(8 → 1)
    Model model(graph);
    model.add(std::make_unique<Linear>(graph, 2, 8, true));
    model.add(std::make_unique<Linear>(graph, 8, 1, true));

    auto x = graph.variable(Tensor({1, 2}, {0.5f, -0.5f}));
    auto target = graph.variable(Tensor({1, 1}, {1.0f}));

    auto pred = model.forward(x);
    auto loss = graph.mse_loss(pred, target);

    SGD optimizer(model.parameters(), 0.01f);

    float initial_loss = 0.0f;
    graph.forward(loss);
    initial_loss = loss->output.data()[0];

    for (int i = 0; i < 50; ++i) {
        graph.forward(loss);
        graph.backward(loss);
        optimizer.step();
        optimizer.zero_grad();
    }

    graph.forward(loss);
    assert(loss->output.data()[0] < initial_loss);

    std::cout << "  [PASS] test_end_to_end_training\n";
}

// ── Main ────────────────────────────────────────────────────────────

int main() {
    std::cout << "Layer tests:\n";
    test_linear_forward();
    test_linear_no_bias();
    test_linear_backward();
    test_conv2d_forward();
    test_embedding_forward();
    test_dropout_train_vs_eval();
    test_dropout_rate_zero();
    test_layer_norm_forward();
    test_layer_norm_no_affine();
    test_layer_norm_backward();
    test_sequential();
    test_model_train_eval();
    test_end_to_end_training();
    std::cout << "All layer tests passed!\n";
    return 0;
}
