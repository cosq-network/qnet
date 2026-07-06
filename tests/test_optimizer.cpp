#include <qnet/graph.hpp>
#include <qnet/optimizer.hpp>

#include <cassert>
#include <cmath>
#include <iostream>

using namespace cosq::qnet;

static bool approx(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

static void test_graph_parameters() {
    Graph graph;
    auto input = graph.variable(Tensor({1}, {1.0f}));
    auto p1 = graph.parameter(Tensor({1}, {2.0f}));
    auto p2 = graph.parameter(Tensor({1}, {3.0f}));
    auto out = graph.add(p1, p2);
    (void)input;
    (void)out;

    auto params = graph.parameters();
    assert(params.size() == 2);
    assert(params[0] == p1);
    assert(params[1] == p2);

    std::cout << "  [PASS] test_graph_parameters\n";
}

static void test_sgd_step() {
    auto parameter = std::make_shared<Node>(OpType::PARAMETER);
    parameter->output = Tensor({2}, {1.0f, -2.0f});
    parameter->gradient = Tensor({2}, {0.5f, -1.0f});

    SGD optimizer({parameter}, 0.1f);
    optimizer.step();

    assert(approx(parameter->output.data()[0], 0.95f));
    assert(approx(parameter->output.data()[1], -1.9f));

    std::cout << "  [PASS] test_sgd_step\n";
}

static void test_sgd_momentum_nesterov() {
    auto parameter = std::make_shared<Node>(OpType::PARAMETER);
    parameter->output = Tensor({1}, {1.0f});

    SGD optimizer({parameter}, 0.1f, 0.9f, 0.0f, true);

    parameter->gradient = Tensor({1}, {0.5f});
    optimizer.step();
    assert(approx(parameter->output.data()[0], 0.905f));

    parameter->gradient = Tensor({1}, {0.5f});
    optimizer.step();
    assert(approx(parameter->output.data()[0], 0.7695f));

    std::cout << "  [PASS] test_sgd_momentum_nesterov\n";
}

static void test_adam_multi_step() {
    auto parameter = std::make_shared<Node>(OpType::PARAMETER);
    parameter->output = Tensor({1}, {1.0f});
    parameter->gradient = Tensor({1}, {0.5f});

    Adam optimizer({parameter}, 0.1f, 0.9f, 0.999f, 1e-8f, 0.0f);

    // With constant g, bias corrections cancel: each step updates by lr=0.1
    optimizer.step();
    assert(approx(parameter->output.data()[0], 0.9f, 1e-4f));

    optimizer.step();
    assert(approx(parameter->output.data()[0], 0.8f, 1e-4f));

    optimizer.step();
    assert(approx(parameter->output.data()[0], 0.7f, 1e-4f));

    std::cout << "  [PASS] test_adam_multi_step\n";
}

static void test_adamw_multi_step() {
    auto parameter = std::make_shared<Node>(OpType::PARAMETER);
    parameter->output = Tensor({1}, {1.0f});
    parameter->gradient = Tensor({1}, {0.5f});

    AdamW optimizer({parameter}, 0.1f, 0.9f, 0.999f, 1e-8f, 0.1f);

    optimizer.step();
    assert(approx(parameter->output.data()[0], 0.89f, 1e-4f));

    optimizer.step();
    assert(approx(parameter->output.data()[0], 0.7811f, 1e-4f));

    optimizer.step();
    assert(approx(parameter->output.data()[0], 0.673289f, 1e-4f));

    std::cout << "  [PASS] test_adamw_multi_step\n";
}

static void test_optimizer_zero_grad() {
    auto parameter = std::make_shared<Node>(OpType::PARAMETER);
    parameter->output = Tensor({1}, {1.0f});
    parameter->gradient = Tensor({1}, {0.5f});

    SGD optimizer({parameter}, 0.1f);
    optimizer.zero_grad();

    assert(parameter->gradient.size() == 0);

    // step() on empty gradient should be a no-op (not crash)
    optimizer.step();

    std::cout << "  [PASS] test_optimizer_zero_grad\n";
}

static void test_sgd_weight_decay() {
    auto parameter = std::make_shared<Node>(OpType::PARAMETER);
    parameter->output = Tensor({2}, {1.0f, 2.0f});
    parameter->gradient = Tensor({2}, {0.5f, 1.0f});

    SGD optimizer({parameter}, 0.1f, 0.0f, 0.1f);
    optimizer.step();

    // g_wd = grad + wd * param = {0.5+0.1, 1.0+0.2} = {0.6, 1.2}
    // param -= lr * g_wd = {1-0.06, 2-0.12} = {0.94, 1.88}
    assert(approx(parameter->output.data()[0], 0.94f));
    assert(approx(parameter->output.data()[1], 1.88f));

    std::cout << "  [PASS] test_sgd_weight_decay\n";
}

static void test_adam_zero_gradient() {
    auto parameter = std::make_shared<Node>(OpType::PARAMETER);
    parameter->output = Tensor({2}, {1.0f, 2.0f});
    parameter->gradient = Tensor({2}, {0.0f, 0.0f});

    Adam optimizer({parameter}, 0.1f);
    optimizer.step();

    // With zero gradient: m=0, v=0 → update=0 → params unchanged
    assert(approx(parameter->output.data()[0], 1.0f));
    assert(approx(parameter->output.data()[1], 2.0f));

    std::cout << "  [PASS] test_adam_zero_gradient\n";
}

static void test_gradient_shape_mismatch_throws() {
    auto parameter = std::make_shared<Node>(OpType::PARAMETER);
    parameter->output = Tensor({2}, {1.0f, 2.0f});
    parameter->gradient = Tensor({1}, {0.5f}); // wrong shape

    SGD optimizer({parameter}, 0.1f);

    bool threw = false;
    try {
        optimizer.step();
    } catch (const std::logic_error&) {
        threw = true;
    }
    assert(threw);

    std::cout << "  [PASS] test_gradient_shape_mismatch_throws\n";
}

static void test_adamw_shape_mismatch_throws() {
    auto parameter = std::make_shared<Node>(OpType::PARAMETER);
    parameter->output = Tensor({2}, {1.0f, 2.0f});
    parameter->gradient = Tensor({1}, {0.5f});

    AdamW optimizer({parameter}, 0.1f);

    bool threw = false;
    try {
        optimizer.step();
    } catch (const std::logic_error&) {
        threw = true;
    }
    assert(threw);

    std::cout << "  [PASS] test_adamw_shape_mismatch_throws\n";
}

int main() {
    std::cout << "Optimizer tests:\n";
    test_graph_parameters();
    test_sgd_step();
    test_sgd_momentum_nesterov();
    test_adam_multi_step();
    test_adamw_multi_step();
    test_sgd_weight_decay();
    test_adam_zero_gradient();
    test_optimizer_zero_grad();
    test_gradient_shape_mismatch_throws();
    test_adamw_shape_mismatch_throws();
    std::cout << "All optimizer tests passed!\n";
    return 0;
}
