#pragma once

#include <qnet/tensor.hpp>

#include <memory>
#include <vector>

namespace cosq::qnet {

struct Node;

struct Ops {
    static Tensor matmul(const Tensor& a, const Tensor& b);

    static Tensor add(const Tensor& a, const Tensor& b);

    static Tensor mul(const Tensor& a, const Tensor& b);

    static Tensor relu(const Tensor& x);

    static Tensor sigmoid(const Tensor& x);

    static Tensor softmax(const Tensor& x);

    static Tensor conv2d(const Tensor& input, const Tensor& kernel,
                         size_t stride_h = 1, size_t stride_w = 1,
                         size_t pad_h = 0, size_t pad_w = 0);

    static Tensor embedding(const Tensor& weight, const Tensor& indices);

    static void matmul_backward(const Tensor& a, const Tensor& b,
                                const Tensor& grad_output,
                                Tensor& grad_a, Tensor& grad_b);

    static void relu_backward(const Tensor& dy, const Tensor& x,
                              Tensor& grad_x);

    static void add_backward(const Tensor& dy, const Tensor& a, const Tensor& b,
                             Tensor& grad_a, Tensor& grad_b);

    static void mul_backward(const Tensor& dy, const Tensor& a, const Tensor& b,
                             Tensor& grad_a, Tensor& grad_b);

    static void sigmoid_backward(const Tensor& dy, const Tensor& out,
                                 Tensor& grad_x);

    static void softmax_backward(const Tensor& dy, const Tensor& out,
                                 Tensor& grad_x);

    static void conv2d_backward(const Tensor& dy, const Tensor& input,
                                const Tensor& kernel,
                                size_t stride_h, size_t stride_w,
                                size_t pad_h, size_t pad_w,
                                Tensor& grad_input, Tensor& grad_kernel);

    static void embedding_backward(const Tensor& dy, const Tensor& indices,
                                   Tensor& grad_weight);
};

} // namespace cosq::qnet
