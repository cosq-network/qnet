#include <qnet/blas.hpp>
#include <qnet/graph.hpp>
#include <qnet/ops.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>

using namespace cosq::qnet;

static bool approx(float a, float b, float eps = 1e-4f) {
    return std::abs(a - b) < eps;
}

static Tensor make_rand(const std::vector<size_t>& shape) {
    static std::mt19937 rng(42);
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Tensor t(shape);
    for (size_t i = 0; i < t.size(); ++i)
        t.data()[i] = dist(rng);
    return t;
}

static float numerical_grad(const Tensor& input, size_t index,
                            const std::function<Tensor()>& forward,
                            float eps = 1e-4f) {
    float orig = const_cast<Tensor&>(input).data()[index];

    const_cast<Tensor&>(input).data()[index] = orig + eps;
    auto pos = forward();
    const_cast<Tensor&>(input).data()[index] = orig - eps;
    auto neg = forward();
    const_cast<Tensor&>(input).data()[index] = orig;

    float pos_sum = 0.0f, neg_sum = 0.0f;
    for (size_t i = 0; i < pos.size(); ++i) pos_sum += pos.data()[i];
    for (size_t i = 0; i < neg.size(); ++i) neg_sum += neg.data()[i];

    return (pos_sum - neg_sum) / (2.0f * eps);
}

// ----------------------------------------------------------------
// Test individual backward ops
// ----------------------------------------------------------------

static void test_relu_backward() {
    Tensor x({3}, {-1.0f, 0.5f, 2.0f});
    Tensor dy({3}, {1.0f, 1.0f, 1.0f});
    Tensor grad_x(x.shape());

    Ops::relu_backward(dy, x, grad_x);

    assert(approx(grad_x.data()[0], 0.0f));  // x < 0 → 0
    assert(approx(grad_x.data()[1], 1.0f));  // x > 0 → 1
    assert(approx(grad_x.data()[2], 1.0f));

    static const auto forward_fn = [&]() { return Ops::relu(x); };
    float ng = numerical_grad(x, 1, forward_fn);
    assert(approx(ng, 1.0f, 1e-3f));

    float ng_zero = numerical_grad(x, 0, forward_fn);
    assert(approx(ng_zero, 0.0f, 1e-3f));

    std::cout << "  [PASS] test_relu_backward\n";
}

static void test_add_backward() {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({2, 3}, {6, 5, 4, 3, 2, 1});
    Tensor dy({2, 3}, {1, 1, 1, 1, 1, 1});
    Tensor grad_a(a.shape()), grad_b(b.shape());

    Ops::add_backward(dy, a, b, grad_a, grad_b);

    for (size_t i = 0; i < dy.size(); ++i) {
        assert(approx(grad_a.data()[i], 1.0f));
        assert(approx(grad_b.data()[i], 1.0f));
    }

    static const auto forward_fn = [&]() { return Ops::add(a, b); };
    float ng = numerical_grad(a, 2, forward_fn);
    assert(approx(ng, 1.0f, 1e-3f));

    std::cout << "  [PASS] test_add_backward\n";
}

static void test_mul_backward() {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({2, 3}, {6, 5, 4, 3, 2, 1});
    Tensor dy({2, 3}, {1.0f, 0.5f, 2.0f, 1.0f, 1.0f, 1.0f});
    Tensor grad_a(a.shape()), grad_b(b.shape());

    Ops::mul_backward(dy, a, b, grad_a, grad_b);

    for (size_t i = 0; i < dy.size(); ++i) {
        assert(approx(grad_a.data()[i], dy.data()[i] * b.data()[i]));
        assert(approx(grad_b.data()[i], dy.data()[i] * a.data()[i]));
    }

    static const auto forward_fn = [&]() { return Ops::mul(a, b); };
    float ng = numerical_grad(a, 0, forward_fn);
    float expected_b0 = b.data()[0];
    assert(approx(ng, expected_b0, 1e-3f));

    std::cout << "  [PASS] test_mul_backward\n";
}

static void test_sigmoid_backward() {
    Tensor x({4}, {-2.0f, -1.0f, 0.0f, 1.0f});
    auto out = Ops::sigmoid(x);
    Tensor dy({4}, {1.0f, 1.0f, 1.0f, 1.0f});
    Tensor grad_x(x.shape());

    Ops::sigmoid_backward(dy, out, grad_x);

    for (size_t i = 0; i < x.size(); ++i) {
        float s = out.data()[i];
        float expected = s * (1.0f - s);
        assert(approx(grad_x.data()[i], expected));
    }

    static const auto forward_fn = [&]() { return Ops::sigmoid(x); };
    float ng = numerical_grad(x, 2, forward_fn);
    float s2 = 1.0f / (1.0f + std::exp(0.0f));
    assert(approx(ng, s2 * (1.0f - s2), 1e-3f));

    std::cout << "  [PASS] test_sigmoid_backward\n";
}

static void test_matmul_backward() {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor dy({2, 2}, {1, 1, 1, 1});
    Tensor grad_a(a.shape()), grad_b(b.shape());

    Ops::matmul_backward(a, b, dy, grad_a, grad_b);

    // grad_a = dy @ b^T
    assert(approx(grad_a.at({0, 0}), 7.0f + 8.0f));
    assert(approx(grad_a.at({0, 1}), 9.0f + 10.0f));
    assert(approx(grad_a.at({1, 2}), 11.0f + 12.0f));

    // grad_b = a^T @ dy
    assert(approx(grad_b.at({0, 0}), 1.0f + 4.0f));
    assert(approx(grad_b.at({1, 1}), 2.0f + 5.0f));
    assert(approx(grad_b.at({2, 0}), 3.0f + 6.0f));

    static const auto forward_fn = [&]() { return Ops::matmul(a, b); };
    float ng_a = numerical_grad(a, 0, forward_fn);
    assert(approx(ng_a, grad_a.data()[0], 1e-3f));

    float ng_b = numerical_grad(b, 0, forward_fn);
    assert(approx(ng_b, grad_b.data()[0], 1e-3f));

    std::cout << "  [PASS] test_matmul_backward\n";
}

static void test_softmax_backward() {
    Tensor x({1, 3}, {1.0f, 2.0f, 3.0f});
    auto out = Ops::softmax(x);
    Tensor dy({1, 3}, {1.0f, 0.0f, 0.0f});  // only first class gets gradient
    Tensor grad_x(x.shape());

    Ops::softmax_backward(dy, out, grad_x);

    float s0 = out.data()[0], s1 = out.data()[1], s2 = out.data()[2];
    float sum_s_dy = s0 * 1.0f + s1 * 0.0f + s2 * 0.0f;
    assert(approx(grad_x.data()[0], s0 * (1.0f - sum_s_dy)));
    assert(approx(grad_x.data()[1], s1 * (0.0f - sum_s_dy)));
    assert(approx(grad_x.data()[2], s2 * (0.0f - sum_s_dy)));

    static const auto forward_fn = [&]() { return Ops::softmax(x); };
    float ng = numerical_grad(x, 1, forward_fn);
    assert(approx(ng, grad_x.data()[1], 1e-3f));

    std::cout << "  [PASS] test_softmax_backward\n";
}

static void test_conv2d_backward() {
    size_t N = 1, C = 1, H = 4, W = 4;
    size_t K = 1, KH = 2, KW = 2;
    size_t stride = 1, pad = 0;

    std::vector<float> inp_data = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    std::vector<float> ker_data = {1, 0, 0, -1};

    Tensor input({N, C, H, W}, inp_data);
    Tensor kernel({K, C, KH, KW}, ker_data);

    auto out = Ops::conv2d(input, kernel, stride, stride, pad, pad);

    size_t OH = (H + 2 * pad - KH) / stride + 1;
    size_t OW = (W + 2 * pad - KW) / stride + 1;
    Tensor dy({N, K, OH, OW}, {1.0f, 1.0f, 1.0f,
                               1.0f, 1.0f, 1.0f,
                               1.0f, 1.0f, 1.0f});

    Tensor grad_input(input.shape());
    Tensor grad_kernel(kernel.shape());

    Ops::conv2d_backward(dy, input, kernel,
                         stride, stride, pad, pad,
                         grad_input, grad_kernel);

    assert(approx(grad_kernel.data()[0], 54.0f, 1e-2f));
    assert(approx(grad_kernel.data()[1], 63.0f, 1e-2f));
    assert(approx(grad_kernel.data()[2], 90.0f, 1e-2f));
    assert(approx(grad_kernel.data()[3], 99.0f, 1e-2f));

    // grad_input verification via im2col backward
    assert(approx(grad_input.at({0, 0, 0, 0}), -1.0f, 1e-2f));
    assert(approx(grad_input.at({0, 0, 1, 1}), 0.0f, 1e-2f));
    assert(approx(grad_input.at({0, 0, 3, 3}), 1.0f, 1e-2f));

    // Numerical gradient check on input
    static const auto fwd_input = [&]() { return Ops::conv2d(input, kernel, stride, stride, pad, pad); };
    float ng_in = numerical_grad(input, 5, fwd_input);
    assert(approx(ng_in, grad_input.data()[5], 0.5f));

    // Numerical gradient check on kernel
    static const auto fwd_kernel = [&]() { return Ops::conv2d(input, kernel, stride, stride, pad, pad); };
    float ng_ker = numerical_grad(kernel, 0, fwd_kernel);
    assert(approx(ng_ker, grad_kernel.data()[0], 0.5f));

    std::cout << "  [PASS] test_conv2d_backward\n";
}

static void test_embedding_backward_numerical() {
    Tensor weight({3, 2}, {1, 2, 3, 4, 5, 6});
    Tensor indices({3}, {0, 1, 2});
    Tensor dy({3, 2}, {1, 0, 0, 1, 1, 1});
    Tensor grad_weight(weight.shape());

    Ops::embedding_backward(dy, indices, grad_weight);

    assert(approx(grad_weight.at({0, 0}), 1.0f));
    assert(approx(grad_weight.at({0, 1}), 0.0f));
    assert(approx(grad_weight.at({1, 0}), 0.0f));
    assert(approx(grad_weight.at({1, 1}), 1.0f));
    assert(approx(grad_weight.at({2, 0}), 1.0f));
    assert(approx(grad_weight.at({2, 1}), 1.0f));

    static const auto forward_fn = [&]() { return Ops::embedding(weight, indices); };
    float ng = numerical_grad(weight, 0, forward_fn);
    assert(approx(ng, grad_weight.data()[0], 1e-3f));

    std::cout << "  [PASS] test_embedding_backward_numerical\n";
}

static void test_conv2d_stride_pad_backward() {
    size_t N = 1, C = 1, H = 5, W = 5;
    size_t K = 1, KH = 3, KW = 3;
    size_t stride = 2, pad = 1;

    std::vector<float> inp_data = {
        1,  2,  3,  4,  5,
        6,  7,  8,  9,  10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };
    std::vector<float> ker_data = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };

    Tensor input({N, C, H, W}, inp_data);
    Tensor kernel({K, C, KH, KW}, ker_data);

    auto out = Ops::conv2d(input, kernel, stride, stride, pad, pad);

    size_t OH = (H + 2 * pad - KH) / stride + 1;
    size_t OW = (W + 2 * pad - KW) / stride + 1;
    assert(OH == 3 && OW == 3);

    Tensor dy({N, K, OH, OW}, {1.0f, 1.0f, 1.0f,
                               1.0f, 1.0f, 1.0f,
                               1.0f, 1.0f, 1.0f});
    Tensor grad_input(input.shape());
    Tensor grad_kernel(kernel.shape());

    Ops::conv2d_backward(dy, input, kernel,
                         stride, stride, pad, pad,
                         grad_input, grad_kernel);

    assert(approx(grad_kernel.data()[0], 68.0f, 1e-2f));
    assert(approx(grad_kernel.data()[4], 80.0f, 1e-2f));
    assert(approx(grad_kernel.data()[8], 68.0f, 1e-2f));

    static const auto fwd = [&]() { return Ops::conv2d(input, kernel, stride, stride, pad, pad); };
    float ng_in = numerical_grad(input, 6, fwd);
    assert(approx(ng_in, grad_input.data()[6], 0.5f));

    std::cout << "  [PASS] test_conv2d_stride_pad_backward\n";
}

static void test_embedding_backward() {
    Tensor weight({4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Tensor indices({2}, {1, 3});
    Tensor out = Ops::embedding(weight, indices);

    Tensor dy({2, 3}, {1, 1, 1, 2, 2, 2});
    Tensor grad_weight(weight.shape());

    Ops::embedding_backward(dy, indices, grad_weight);

    for (size_t d = 0; d < 3; ++d) {
        assert(approx(grad_weight.at({0, d}), 0.0f));
        assert(approx(grad_weight.at({1, d}), 1.0f));
        assert(approx(grad_weight.at({2, d}), 0.0f));
        assert(approx(grad_weight.at({3, d}), 2.0f));
    }

    std::cout << "  [PASS] test_embedding_backward\n";
}

// ----------------------------------------------------------------
// End-to-end graph backward test with mixed ops
// ----------------------------------------------------------------

static void test_graph_backward_chain() {
    Graph graph;

    auto x = graph.variable(Tensor({1, 4}, {1, 2, 3, 4}));
    auto w1 = graph.parameter(Tensor({4, 3}, {0.1f, 0.2f, 0.3f,
                                                0.4f, 0.5f, 0.6f,
                                                0.7f, 0.8f, 0.9f,
                                                1.0f, 1.1f, 1.2f}));
    auto w2 = graph.parameter(Tensor({3, 1}, {0.5f, 1.0f, 1.5f}));

    auto mm1 = graph.matmul(x, w1);
    auto act = graph.relu(mm1);
    auto mm2 = graph.matmul(act, w2);

    graph.forward(mm2);
    graph.backward(mm2);

    assert(x->gradient.size() > 0);
    assert(w1->gradient.size() > 0);
    assert(w2->gradient.size() > 0);

    std::cout << "  [PASS] test_graph_backward_chain\n";
}

static void test_graph_zero_grad() {
    Graph graph;
    auto x = graph.variable(Tensor({1, 2}, {1, 2}));
    auto w = graph.parameter(Tensor({2, 1}, {0.5f, 1.0f}));
    auto y = graph.matmul(x, w);

    graph.forward(y);
    graph.backward(y);
    assert(w->gradient.size() > 0);

    graph.zero_grad();
    assert(w->gradient.size() == 0);

    std::cout << "  [PASS] test_graph_zero_grad\n";
}

static void test_graph_relu_backward() {
    // x -> relu -> sum(L = sum(output))
    // dL/dx = 1 if x > 0 else 0
    Graph graph;
    auto x = graph.variable(Tensor({1, 4}, {-1.0f, 0.0f, 2.0f, 3.0f}));
    auto r = graph.relu(x);

    graph.forward(r);
    graph.backward(r);

    assert(approx(x->gradient.data()[0], 0.0f));
    assert(approx(x->gradient.data()[1], 0.0f));
    assert(approx(x->gradient.data()[2], 1.0f));
    assert(approx(x->gradient.data()[3], 1.0f));

    std::cout << "  [PASS] test_graph_relu_backward\n";
}

static void test_graph_add_backward() {
    Graph graph;
    auto a = graph.variable(Tensor({1, 3}, {1, 2, 3}));
    auto b = graph.variable(Tensor({1, 3}, {4, 5, 6}));
    auto sum_node = graph.add(a, b);

    graph.forward(sum_node);
    graph.backward(sum_node);

    for (size_t i = 0; i < 3; ++i) {
        assert(approx(a->gradient.data()[i], 1.0f));
        assert(approx(b->gradient.data()[i], 1.0f));
    }

    std::cout << "  [PASS] test_graph_add_backward\n";
}

static void test_graph_mul_backward() {
    Graph graph;
    auto a = graph.variable(Tensor({1, 3}, {2, 3, 4}));
    auto b = graph.variable(Tensor({1, 3}, {5, 6, 7}));
    auto mul_node = graph.mul(a, b);

    graph.forward(mul_node);
    graph.backward(mul_node);

    assert(approx(a->gradient.data()[0], 5.0f));
    assert(approx(a->gradient.data()[1], 6.0f));
    assert(approx(a->gradient.data()[2], 7.0f));
    assert(approx(b->gradient.data()[0], 2.0f));
    assert(approx(b->gradient.data()[1], 3.0f));
    assert(approx(b->gradient.data()[2], 4.0f));

    std::cout << "  [PASS] test_graph_mul_backward\n";
}

static void test_graph_sigmoid_backward() {
    Graph graph;
    auto x = graph.variable(Tensor({1, 3}, {0.0f, 1.0f, -1.0f}));
    auto s = graph.sigmoid(x);

    graph.forward(s);
    graph.backward(s);

    for (size_t i = 0; i < 3; ++i) {
        float v = 1.0f / (1.0f + std::exp(-x->output.data()[i]));
        float expected = v * (1.0f - v);
        assert(approx(x->gradient.data()[i], expected, 1e-3f));
    }

    std::cout << "  [PASS] test_graph_sigmoid_backward\n";
}

static void test_gradient_accumulation() {
    Graph graph;
    auto x = graph.variable(Tensor({1, 3}, {1, 2, 3}));
    auto w = graph.parameter(Tensor({3, 1}, {0.5f, 1.0f, 1.5f}));
    auto y = graph.matmul(x, w);

    graph.forward(y);
    graph.backward(y);
    float grad_w0_first = w->gradient.data()[0];

    // Second backward should accumulate
    graph.forward(y);
    graph.backward(y);
    assert(approx(w->gradient.data()[0], 2.0f * grad_w0_first));

    // zero_grad then backward should reset
    graph.zero_grad();
    graph.forward(y);
    graph.backward(y);
    assert(approx(w->gradient.data()[0], grad_w0_first));

    std::cout << "  [PASS] test_gradient_accumulation\n";
}

int main() {
    std::cout << "Backward op tests:\n";
    test_relu_backward();
    test_add_backward();
    test_mul_backward();
    test_sigmoid_backward();
    test_softmax_backward();
    test_matmul_backward();
    test_conv2d_backward();
    test_conv2d_stride_pad_backward();
    test_embedding_backward();
    test_embedding_backward_numerical();
    std::cout << "Graph backward tests:\n";
    test_graph_backward_chain();
    test_graph_zero_grad();
    test_graph_relu_backward();
    test_graph_add_backward();
    test_graph_mul_backward();
    test_graph_sigmoid_backward();
    test_gradient_accumulation();
    std::cout << "All backward tests passed!\n";
    return 0;
}
