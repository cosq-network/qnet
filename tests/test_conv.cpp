#include <qnet/graph.hpp>
#include <qnet/ops.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>

using namespace cosq::qnet;

static bool approx(float a, float b, float eps = 1e-4f) {
    return std::abs(a - b) < eps;
}

static void test_im2col_conv2d() {
    Tensor input({1, 1, 4, 4});
    for (size_t i = 0; i < input.size(); ++i) input.data()[i] = static_cast<float>(i);

    Tensor kernel({1, 1, 3, 3});
    float k_data[] = {1, 0, -1, 1, 0, -1, 1, 0, -1};
    for (size_t i = 0; i < 9; ++i) kernel.data()[i] = k_data[i];

    Tensor result = Ops::conv2d(input, kernel, 1, 1, 0, 0);

    assert(result.shape()[0] == 1);
    assert(result.shape()[1] == 1);
    assert(result.shape()[2] == 2);
    assert(result.shape()[3] == 2);

    float expected[4] = {
        -8.0f, -8.0f,
        -8.0f, -8.0f
    };

    assert(approx(result.at({0, 0, 0, 0}), expected[0]));
    assert(approx(result.at({0, 0, 0, 1}), expected[1]));
    assert(approx(result.at({0, 0, 1, 0}), expected[2]));
    assert(approx(result.at({0, 0, 1, 1}), expected[3]));

    std::cout << "  [PASS] test_im2col_conv2d\n";
}

static void test_conv2d_multi_channel() {
    Tensor input({1, 2, 3, 3});
    for (size_t i = 0; i < input.size(); ++i) input.data()[i] = static_cast<float>(i);

    Tensor kernel({2, 2, 2, 2});
    for (size_t i = 0; i < kernel.size(); ++i) kernel.data()[i] = 1.0f;

    Tensor result = Ops::conv2d(input, kernel, 1, 1, 0, 0);

    assert(result.shape()[0] == 1);
    assert(result.shape()[1] == 2);
    assert(result.shape()[2] == 2);
    assert(result.shape()[3] == 2);

    assert(approx(result.at({0, 0, 0, 0}), 60.0f));

    std::cout << "  [PASS] test_conv2d_multi_channel\n";
}

static void test_conv2d_stride_pad() {
    Tensor input({1, 1, 5, 5});
    for (size_t i = 0; i < input.size(); ++i) input.data()[i] = static_cast<float>(i);

    Tensor kernel({1, 1, 3, 3});
    float k_data[] = {1, 0, -1, 1, 0, -1, 1, 0, -1};
    for (size_t i = 0; i < 9; ++i) kernel.data()[i] = k_data[i];

    // stride=2, pad=1
    Tensor result = Ops::conv2d(input, kernel, 2, 2, 1, 1);
    assert(result.shape()[0] == 1);
    assert(result.shape()[1] == 1);
    assert(result.shape()[2] == 3);
    assert(result.shape()[3] == 3);

    assert(approx(result.at({0, 0, 0, 0}), -4.0f));
    assert(approx(result.at({0, 0, 1, 1}), -40.0f));
    assert(approx(result.at({0, 0, 2, 2}), -4.0f));

    std::cout << "  [PASS] test_conv2d_stride_pad\n";
}

static void test_conv2d_graph_backward() {
    Graph graph;
    Tensor X({1, 1, 4, 4});
    for (size_t i = 0; i < X.size(); ++i) X.data()[i] = static_cast<float>(i);
    Tensor K({1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, -1.0f});

    auto x = graph.variable(X);
    auto k = graph.parameter(K);
    auto y = graph.conv2d(x, k, 1, 1, 0, 0);

    graph.forward(y);
    graph.backward(y);

    assert(x->gradient.size() > 0);
    assert(k->gradient.size() > 0);
    assert(approx(k->gradient.data()[0], 10.0f, 1e-2f));
    assert(approx(k->gradient.data()[3], -25.0f, 1e-2f));

    std::cout << "  [PASS] test_conv2d_graph_backward\n";
}

static void test_thread_pool() {
    Graph graph(4);
    Tensor X({1, 4}, {1, 2, 3, 4});
    Tensor W1({4, 2}, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f});
    Tensor W2({2, 3}, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f});

    auto x = graph.variable(X);
    auto w1 = graph.parameter(W1);
    auto w2 = graph.parameter(W2);
    auto h1 = graph.matmul(x, w1);
    auto h2 = graph.relu(h1);
    auto out = graph.matmul(h2, w2);

    graph.forward(out);

    assert(out->output.shape()[0] == 1);
    assert(out->output.shape()[1] == 3);

    std::cout << "  [PASS] test_thread_pool\n";
}

int main() {
    std::cout << "Conv & thread pool tests:\n";
    test_im2col_conv2d();
    test_conv2d_multi_channel();
    test_conv2d_stride_pad();
    test_conv2d_graph_backward();
    test_thread_pool();
    std::cout << "All conv & thread pool tests passed!\n";
    return 0;
}
