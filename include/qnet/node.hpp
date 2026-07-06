#pragma once

#include <qnet/export.hpp>
#include <qnet/tensor.hpp>

#include <memory>
#include <string>
#include <vector>

namespace cosq::qnet {

enum class OpType {
    INPUT,
    PARAMETER,
    MATMUL,
    ADD,
    MUL,
    RELU,
    SIGMOID,
    SOFTMAX,
    CONV2D,
    EMBEDDING,
    NONE
};

struct QNET_API Node {
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

    Node() = default;
    explicit Node(OpType type) : op_type(type) {}
};

} // namespace cosq::qnet
