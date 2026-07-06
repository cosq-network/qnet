#include <qnet/graph.hpp>
#include <qnet/ops.hpp>

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>

using namespace cosq::qnet;

static bool approx(float a, float b, float eps = 1e-4f) {
    return std::abs(a - b) < eps;
}

static float numerical_grad_scalar(Tensor& input, size_t index,
                                   const std::function<float()>& forward,
                                   float eps = 1e-4f) {
    float original = input.data()[index];

    input.data()[index] = original + eps;
    float pos = forward();

    input.data()[index] = original - eps;
    float neg = forward();

    input.data()[index] = original;
    return (pos - neg) / (2.0f * eps);
}

static float stable_cross_entropy_sample(const Tensor& logits, size_t batch_index,
                                         size_t target_index) {
    float max_logit = logits.at({batch_index, 0});
    for (size_t c = 1; c < logits.shape()[1]; ++c) {
        max_logit = std::max(max_logit, logits.at({batch_index, c}));
    }

    float exp_sum = 0.0f;
    for (size_t c = 0; c < logits.shape()[1]; ++c) {
        exp_sum += std::exp(logits.at({batch_index, c}) - max_logit);
    }

    return (max_logit + std::log(exp_sum)) - logits.at({batch_index, target_index});
}

static void test_cross_entropy_loss_value() {
    Tensor logits({2, 3}, {2.0f, 0.5f, -1.0f,
                            0.1f, 1.2f, 2.4f});
    Tensor targets({2}, {0.0f, 2.0f});

    Tensor loss = Ops::cross_entropy_loss(logits, targets);
    float expected =
        (stable_cross_entropy_sample(logits, 0, 0) +
         stable_cross_entropy_sample(logits, 1, 2)) / 2.0f;

    assert(loss.shape() == std::vector<size_t>({1}));
    assert(approx(loss.data()[0], expected, 1e-5f));
    std::cout << "  [PASS] test_cross_entropy_loss_value\n";
}

static void test_cross_entropy_backward() {
    Tensor logits({2, 3}, {1.5f, -0.5f, 0.25f,
                            -1.0f, 0.75f, 2.0f});
    Tensor targets({2}, {2.0f, 1.0f});
    Tensor grad_output({1}, {1.0f});
    Tensor grad_logits(logits.shape());

    Ops::cross_entropy_backward(logits, targets, grad_output, grad_logits);

    for (size_t i = 0; i < logits.size(); ++i) {
        float numerical = numerical_grad_scalar(
            logits, i, [&]() { return Ops::cross_entropy_loss(logits, targets).data()[0]; });
        assert(approx(grad_logits.data()[i], numerical, 1e-3f));
    }

    std::cout << "  [PASS] test_cross_entropy_backward\n";
}

static void test_mse_loss_value() {
    Tensor pred({2, 2}, {1.0f, 3.0f, 2.0f, 4.0f});
    Tensor target({2, 2}, {0.0f, 1.0f, 2.0f, 5.0f});

    Tensor loss = Ops::mse_loss(pred, target);
    float expected = (1.0f + 4.0f + 0.0f + 1.0f) / 4.0f;

    assert(approx(loss.data()[0], expected, 1e-6f));
    std::cout << "  [PASS] test_mse_loss_value\n";
}

static void test_mse_backward() {
    Tensor pred({2, 2}, {1.0f, -2.0f, 0.5f, 3.5f});
    Tensor target({2, 2}, {0.0f, -1.0f, 1.5f, 2.0f});
    Tensor grad_output({1}, {1.0f});
    Tensor grad_pred(pred.shape());
    Tensor grad_target(target.shape());

    Ops::mse_backward(pred, target, grad_output, grad_pred, grad_target);

    for (size_t i = 0; i < pred.size(); ++i) {
        float numerical = numerical_grad_scalar(
            pred, i, [&]() { return Ops::mse_loss(pred, target).data()[0]; });
        assert(approx(grad_pred.data()[i], numerical, 1e-3f));
        assert(approx(grad_target.data()[i], -grad_pred.data()[i], 1e-6f));
    }

    std::cout << "  [PASS] test_mse_backward\n";
}

static void test_binary_cross_entropy_value() {
    Tensor pred({4}, {-2.0f, 0.0f, 1.5f, -0.25f});
    Tensor target({4}, {0.0f, 1.0f, 1.0f, 0.0f});

    Tensor loss = Ops::binary_cross_entropy(pred, target);

    float expected = 0.0f;
    for (size_t i = 0; i < pred.size(); ++i) {
        float x = pred.data()[i];
        float y = target.data()[i];
        expected += std::max(x, 0.0f) - x * y + std::log1p(std::exp(-std::fabs(x)));
    }
    expected /= static_cast<float>(pred.size());

    assert(approx(loss.data()[0], expected, 1e-6f));
    std::cout << "  [PASS] test_binary_cross_entropy_value\n";
}

static void test_binary_cross_entropy_backward() {
    Tensor pred({4}, {-1.25f, 0.75f, 2.5f, -0.5f});
    Tensor target({4}, {0.0f, 1.0f, 1.0f, 0.0f});
    Tensor grad_output({1}, {1.0f});
    Tensor grad_pred(pred.shape());
    Tensor grad_target(target.shape());

    Ops::bce_backward(pred, target, grad_output, grad_pred, grad_target);

    for (size_t i = 0; i < pred.size(); ++i) {
        float numerical = numerical_grad_scalar(
            pred, i, [&]() { return Ops::binary_cross_entropy(pred, target).data()[0]; });
        assert(approx(grad_pred.data()[i], numerical, 1e-3f));
        float expected_target_grad = -pred.data()[i] / static_cast<float>(pred.size());
        assert(approx(grad_target.data()[i], expected_target_grad, 1e-6f));
    }

    std::cout << "  [PASS] test_binary_cross_entropy_backward\n";
}

static void test_cross_entropy_target_out_of_range() {
    Tensor logits({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor targets_bad({2}, {3.0f, 1.0f}); // 3 is out of range for 3 classes

    bool threw = false;
    try {
        Ops::cross_entropy_loss(logits, targets_bad);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);

    Tensor targets_neg({2}, {-1.0f, 1.0f}); // -1 is negative
    threw = false;
    try {
        Ops::cross_entropy_loss(logits, targets_neg);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);

    std::cout << "  [PASS] test_cross_entropy_target_out_of_range\n";
}

static void test_cross_entropy_shape_mismatch() {
    Tensor logits({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor targets_bad({3}, {0.0f, 1.0f, 2.0f}); // batch size mismatch

    bool threw = false;
    try {
        Ops::cross_entropy_loss(logits, targets_bad);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    // 1D logits should also throw
    Tensor logits_1d({3}, {1.0f, 2.0f, 3.0f});
    Tensor targets_ok({3}, {0.0f, 1.0f, 2.0f});
    threw = false;
    try {
        Ops::cross_entropy_loss(logits_1d, targets_ok);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    std::cout << "  [PASS] test_cross_entropy_shape_mismatch\n";
}

static void test_loss_non_unit_grad_output() {
    Tensor logits({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor targets({2}, {0.0f, 2.0f});
    Tensor grad_output({1}, {2.0f}); // non-unit scaling
    Tensor grad_logits(logits.shape());

    Ops::cross_entropy_backward(logits, targets, grad_output, grad_logits);

    // With grad_output=2.0, gradients should be 2x compared to unit
    Tensor grad_logits_unit(logits.shape());
    Ops::cross_entropy_backward(logits, targets, Tensor({1}, {1.0f}), grad_logits_unit);

    for (size_t i = 0; i < grad_logits.size(); ++i) {
        assert(approx(grad_logits.data()[i], 2.0f * grad_logits_unit.data()[i]));
    }

    std::cout << "  [PASS] test_loss_non_unit_grad_output\n";

    // Same test for MSE
    Tensor pred({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor target({2, 2}, {0.0f, 1.0f, 2.0f, 5.0f});
    Tensor grad_pred(pred.shape()), grad_pred_unit(pred.shape());
    Tensor grad_target(target.shape()), grad_target_unit(target.shape());

    Ops::mse_backward(pred, target, grad_output, grad_pred, grad_target);
    Ops::mse_backward(pred, target, Tensor({1}, {1.0f}), grad_pred_unit, grad_target_unit);

    for (size_t i = 0; i < grad_pred.size(); ++i) {
        assert(approx(grad_pred.data()[i], 2.0f * grad_pred_unit.data()[i]));
        assert(approx(grad_target.data()[i], 2.0f * grad_target_unit.data()[i]));
    }

    std::cout << "  [PASS] test_mse_non_unit_grad_output\n";
}

static void test_empty_loss_inputs_throw() {
    bool threw = false;
    try {
        Ops::mse_loss(Tensor({0}), Tensor({0}));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        Ops::binary_cross_entropy(Tensor({0}), Tensor({0}));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        Ops::cross_entropy_loss(Tensor({0, 3}), Tensor({0}));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        Ops::cross_entropy_loss(Tensor({2, 0}), Tensor({2}, {0.0f, 0.0f}));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    std::cout << "  [PASS] test_empty_loss_inputs_throw\n";
}

static void test_graph_cross_entropy_backward() {
    Graph graph;
    auto logits = graph.variable(Tensor({2, 3}, {1.2f, 0.1f, -0.4f,
                                                  -1.0f, 0.2f, 2.3f}));
    auto targets = graph.variable(Tensor({2}, {0.0f, 2.0f}));
    auto loss = graph.cross_entropy_loss(logits, targets);

    graph.forward(loss);
    graph.backward(loss);

    Tensor expected(logits->output.shape());
    Ops::cross_entropy_backward(logits->output, targets->output, Tensor({1}, {1.0f}), expected);

    assert(loss->output.shape() == std::vector<size_t>({1}));
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(approx(logits->gradient.data()[i], expected.data()[i], 1e-5f));
    }
    assert(targets->gradient.size() == 0);

    std::cout << "  [PASS] test_graph_cross_entropy_backward\n";
}

static void test_graph_mse_backward() {
    Graph graph;
    auto pred = graph.variable(Tensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}));
    auto target = graph.variable(Tensor({2, 2}, {0.0f, 2.5f, 2.0f, 5.0f}));
    auto loss = graph.mse_loss(pred, target);

    graph.forward(loss);
    graph.backward(loss);

    Tensor expected(pred->output.shape());
    Tensor expected_target(target->output.shape());
    Ops::mse_backward(pred->output, target->output, Tensor({1}, {1.0f}), expected, expected_target);

    for (size_t i = 0; i < expected.size(); ++i) {
        assert(approx(pred->gradient.data()[i], expected.data()[i], 1e-5f));
        assert(approx(target->gradient.data()[i], expected_target.data()[i], 1e-5f));
    }

    std::cout << "  [PASS] test_graph_mse_backward\n";
}

static void test_graph_binary_cross_entropy_backward() {
    Graph graph;
    auto pred = graph.variable(Tensor({4}, {-1.0f, 0.5f, 2.0f, -0.75f}));
    auto target = graph.variable(Tensor({4}, {0.0f, 1.0f, 1.0f, 0.0f}));
    auto loss = graph.binary_cross_entropy(pred, target);

    graph.forward(loss);
    graph.backward(loss);

    Tensor expected(pred->output.shape());
    Tensor expected_target(target->output.shape());
    Ops::bce_backward(pred->output, target->output, Tensor({1}, {1.0f}), expected, expected_target);

    for (size_t i = 0; i < expected.size(); ++i) {
        assert(approx(pred->gradient.data()[i], expected.data()[i], 1e-5f));
        assert(approx(target->gradient.data()[i], expected_target.data()[i], 1e-5f));
    }

    std::cout << "  [PASS] test_graph_binary_cross_entropy_backward\n";
}

int main() {
    std::cout << "Loss tests:\n";
    test_cross_entropy_loss_value();
    test_cross_entropy_backward();
    test_cross_entropy_target_out_of_range();
    test_cross_entropy_shape_mismatch();
    test_mse_loss_value();
    test_mse_backward();
    test_binary_cross_entropy_value();
    test_binary_cross_entropy_backward();
    test_loss_non_unit_grad_output();
    test_empty_loss_inputs_throw();
    test_graph_cross_entropy_backward();
    test_graph_mse_backward();
    test_graph_binary_cross_entropy_backward();
    std::cout << "All loss tests passed!\n";
    return 0;
}
