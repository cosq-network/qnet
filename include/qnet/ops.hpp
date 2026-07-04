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
};

} // namespace cosq::qnet
