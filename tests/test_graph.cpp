#include <qnet/graph.hpp>
#include <qnet/ops.hpp>

#include <cassert>
#include <cmath>
#include <iostream>

using namespace cosq::qnet;

static bool approx(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

static void test_forward() {
    Graph graph;

    Tensor X({1, 4}, {1, 2, 3, 4});
    Tensor W({4, 2}, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f});

    auto input_node = graph.variable(X);
    auto weight_node = graph.parameter(W);
    auto output_node = graph.matmul(input_node, weight_node);

    graph.forward(output_node);

    Tensor result = output_node->output;
    assert(result.shape()[0] == 1);
    assert(result.shape()[1] == 2);

    float expected_0 = 1.0f * 0.1f + 2.0f * 0.3f + 3.0f * 0.5f + 4.0f * 0.7f;
    float expected_1 = 1.0f * 0.2f + 2.0f * 0.4f + 3.0f * 0.6f + 4.0f * 0.8f;
    assert(approx(result.data()[0], expected_0));
    assert(approx(result.data()[1], expected_1));

    std::cout << "  [PASS] test_forward\n";
}

static void test_forward_activations() {
    Graph graph;
    auto x = graph.variable(Tensor({1, 4}, {-2.0f, -1.0f, 0.0f, 1.0f}));

    auto r = graph.relu(x);
    graph.forward(r);
    assert(approx(r->output.data()[0], 0.0f));
    assert(approx(r->output.data()[1], 0.0f));
    assert(approx(r->output.data()[2], 0.0f));
    assert(approx(r->output.data()[3], 1.0f));

    auto s = graph.sigmoid(x);
    graph.forward(s);
    for (size_t i = 0; i < 4; ++i) {
        float expected = 1.0f / (1.0f + std::exp(-x->output.data()[i]));
        assert(approx(s->output.data()[i], expected));
    }

    Tensor logits_data({1, 3}, {1.0f, 2.0f, 3.0f});
    auto logits = graph.variable(logits_data);
    auto sm = graph.softmax(logits);
    graph.forward(sm);
    float sum = 0.0f;
    for (size_t i = 0; i < 3; ++i) sum += sm->output.data()[i];
    assert(approx(sum, 1.0f));

    std::cout << "  [PASS] test_forward_activations\n";
}

static void test_forward_add_mul() {
    Graph graph;
    auto a = graph.variable(Tensor({1, 3}, {1, 2, 3}));
    auto b = graph.variable(Tensor({1, 3}, {4, 5, 6}));

    auto sum_node = graph.add(a, b);
    graph.forward(sum_node);
    assert(approx(sum_node->output.data()[0], 5.0f));
    assert(approx(sum_node->output.data()[1], 7.0f));
    assert(approx(sum_node->output.data()[2], 9.0f));

    auto mul_node = graph.mul(a, b);
    graph.forward(mul_node);
    assert(approx(mul_node->output.data()[0], 4.0f));
    assert(approx(mul_node->output.data()[1], 10.0f));
    assert(approx(mul_node->output.data()[2], 18.0f));

    std::cout << "  [PASS] test_forward_add_mul\n";
}

static void test_forward_add_broadcast() {
    Graph graph;
    auto x = graph.variable(Tensor({2, 3}, {1, 2, 3,
                                            4, 5, 6}));
    auto bias = graph.parameter(Tensor({1, 3}, {10, 20, 30}));

    auto y = graph.add(x, bias);
    graph.forward(y);

    assert(approx(y->output.at({0, 0}), 11.0f));
    assert(approx(y->output.at({0, 2}), 33.0f));
    assert(approx(y->output.at({1, 0}), 14.0f));
    assert(approx(y->output.at({1, 2}), 36.0f));

    graph.backward(y);
    assert(approx(x->gradient.at({0, 0}), 1.0f));
    assert(approx(x->gradient.at({1, 2}), 1.0f));
    assert(approx(bias->gradient.at({0, 0}), 2.0f));
    assert(approx(bias->gradient.at({0, 1}), 2.0f));
    assert(approx(bias->gradient.at({0, 2}), 2.0f));

    std::cout << "  [PASS] test_forward_add_broadcast\n";
}

static void test_graph_multi_output() {
    Graph graph;
    auto x = graph.variable(Tensor({1, 2}, {1, 2}));
    auto w1 = graph.parameter(Tensor({2, 1}, {0.5f, 1.0f}));
    auto w2 = graph.parameter(Tensor({2, 1}, {2.0f, 3.0f}));

    auto y1 = graph.matmul(x, w1);
    auto y2 = graph.matmul(x, w2);

    graph.forward(y1);
    graph.forward(y2);
    graph.backward(y1);

    assert(x->gradient.size() > 0);
    float grad_x0 = x->gradient.data()[0];

    // Accumulate gradient from y2
    graph.backward(y2);
    assert(approx(x->gradient.data()[0], grad_x0 + 2.0f, 1e-5f));

    std::cout << "  [PASS] test_graph_multi_output\n";
}

static void test_backward() {
    Graph graph;

    Tensor X({1, 3}, {1, 2, 3});
    Tensor W({3, 1}, {0.5f, 1.0f, 1.5f});

    auto input_node = graph.variable(X);
    auto weight_node = graph.parameter(W);
    auto output_node = graph.matmul(input_node, weight_node);

    graph.forward(output_node);
    graph.backward(output_node);

    assert(input_node->gradient.size() > 0);
    assert(weight_node->gradient.size() > 0);

    float expected_grad_x0 = 0.5f;
    float expected_grad_x1 = 1.0f;
    float expected_grad_x2 = 1.5f;
    assert(approx(input_node->gradient.data()[0], expected_grad_x0));
    assert(approx(input_node->gradient.data()[1], expected_grad_x1));
    assert(approx(input_node->gradient.data()[2], expected_grad_x2));

    std::cout << "  [PASS] test_backward\n";
}

int main() {
    std::cout << "Graph tests:\n";
    test_forward();
    test_forward_activations();
    test_forward_add_mul();
    test_forward_add_broadcast();
    test_backward();
    test_graph_multi_output();
    std::cout << "All graph tests passed!\n";
    return 0;
}
