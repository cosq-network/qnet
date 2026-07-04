#pragma once

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

struct Node {
    OpType op_type = OpType::NONE;
    std::vector<std::shared_ptr<Node>> inputs;
    Tensor output;
    Tensor gradient;
    std::string name;

    Node() = default;
    explicit Node(OpType type) : op_type(type) {}
};

} // namespace cosq::qnet
