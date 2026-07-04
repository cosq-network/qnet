#include <qnet/graph.hpp>
#include <qnet/ops.hpp>

#include <cassert>
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
    test_backward();
    std::cout << "All graph tests passed!\n";
    return 0;
}
