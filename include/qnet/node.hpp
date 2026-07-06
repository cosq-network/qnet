#pragma once

#include <qnet/tensor.hpp>

#include <memory>
#include <string>
#include <vector>

namespace cosq::qnet {

enum class OpType {
    INPUT = 0,
    PARAMETER = 1,
    MATMUL = 2,
    ADD = 3,
    MUL = 4,
    RELU = 5,
    SIGMOID = 6,
    SOFTMAX = 7,
    CONV2D = 8,
    EMBEDDING = 9,
    NONE = 10,
    CROSS_ENTROPY_LOSS = 11,
    MSE_LOSS = 12,
    BINARY_CROSS_ENTROPY_LOSS = 13,
    DROPOUT = 14,
    LAYER_NORM = 15
};

struct Node {
    OpType op_type = OpType::NONE;
    std::vector<std::shared_ptr<Node>> inputs;
    Tensor output;
    Tensor gradient;
    std::string name;

    // Conv2d attributes
    size_t conv_stride_h = 1;
    size_t conv_stride_w = 1;
    size_t conv_pad_h = 0;
    size_t conv_pad_w = 0;

    // Dropout attributes
    float dropout_rate = 0.0f;
    Tensor dropout_mask;

    // LayerNorm attributes
    size_t layer_norm_size = 0;
    float layer_norm_eps = 1e-5f;

    Node() = default;
    explicit Node(OpType type) : op_type(type) {}
};

} // namespace cosq::qnet
