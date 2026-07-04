#include <qnet/graph.hpp>
#include <qnet/ops.hpp>

#include <algorithm>
#include <deque>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cosq::qnet {

Graph::Graph(size_t num_threads) : pool_(std::make_unique<ThreadPool>(num_threads)) {}

void Graph::set_num_threads(size_t n) {
    pool_ = std::make_unique<ThreadPool>(n);
}

void Graph::add_node(std::shared_ptr<Node> node) {
    nodes_.push_back(std::move(node));
}

std::shared_ptr<Node> Graph::create_node(OpType type,
                                          const std::vector<std::shared_ptr<Node>>& inputs) {
    auto node = std::make_shared<Node>(type);
    node->inputs = inputs;
    nodes_.push_back(node);
    return node;
}

std::shared_ptr<Node> Graph::variable(const Tensor& data) {
    auto node = create_node(OpType::INPUT, {});
    node->output = data;
    return node;
}

std::shared_ptr<Node> Graph::parameter(const Tensor& data) {
    auto node = create_node(OpType::PARAMETER, {});
    node->output = data;
    return node;
}

std::shared_ptr<Node> Graph::matmul(const std::shared_ptr<Node>& a,
                                     const std::shared_ptr<Node>& b) {
    return create_node(OpType::MATMUL, {a, b});
}

std::shared_ptr<Node> Graph::add(const std::shared_ptr<Node>& a,
                                  const std::shared_ptr<Node>& b) {
    return create_node(OpType::ADD, {a, b});
}

std::shared_ptr<Node> Graph::relu(const std::shared_ptr<Node>& x) {
    return create_node(OpType::RELU, {x});
}

std::shared_ptr<Node> Graph::sigmoid(const std::shared_ptr<Node>& x) {
    return create_node(OpType::SIGMOID, {x});
}

std::shared_ptr<Node> Graph::softmax(const std::shared_ptr<Node>& x) {
    return create_node(OpType::SOFTMAX, {x});
}

void Graph::execute_node(Node* node) {
    switch (node->op_type) {
    case OpType::INPUT:
    case OpType::PARAMETER:
        break;

    case OpType::MATMUL: {
        auto& a = node->inputs[0]->output;
        auto& b = node->inputs[1]->output;
        node->output = Ops::matmul(a, b);
        break;
    }

    case OpType::ADD: {
        auto& a = node->inputs[0]->output;
        auto& b = node->inputs[1]->output;
        node->output = Ops::add(a, b);
        break;
    }

    case OpType::RELU: {
        auto& x = node->inputs[0]->output;
        node->output = Ops::relu(x);
        break;
    }

    case OpType::SIGMOID: {
        auto& x = node->inputs[0]->output;
        node->output = Ops::sigmoid(x);
        break;
    }

    case OpType::SOFTMAX: {
        auto& x = node->inputs[0]->output;
        node->output = Ops::softmax(x);
        break;
    }

    default:
        break;
    }
}

struct LevelNode {
    std::shared_ptr<Node> node;
    size_t depth;
};

static void topo_sort_leveled(const std::shared_ptr<Node>& node,
                              std::vector<LevelNode>& sorted,
                              std::unordered_set<Node*>& visited,
                              std::unordered_map<Node*, size_t>& depths) {
    if (visited.count(node.get())) return;
    visited.insert(node.get());

    size_t max_depth = 0;
    for (auto& input : node->inputs) {
        topo_sort_leveled(input, sorted, visited, depths);
        max_depth = std::max(max_depth, depths[input.get()]);
    }

    size_t depth = max_depth + 1;
    depths[node.get()] = depth;
    sorted.push_back({node, depth});
}

static void reverse_topo(const std::shared_ptr<Node>& node,
                         std::vector<std::shared_ptr<Node>>& sorted,
                         std::unordered_set<Node*>& visited) {
    if (visited.count(node.get())) return;
    visited.insert(node.get());

    sorted.push_back(node);
    for (auto& input : node->inputs) {
        reverse_topo(input, sorted, visited);
    }
}

void Graph::forward(const std::shared_ptr<Node>& output_node) {
    std::vector<LevelNode> order;
    std::unordered_set<Node*> visited;
    std::unordered_map<Node*, size_t> depths;
    topo_sort_leveled(output_node, order, visited, depths);

    if (order.empty()) return;

    size_t max_depth = 0;
    for (auto& ln : order) {
        max_depth = std::max(max_depth, ln.depth);
    }

    for (size_t d = 1; d <= max_depth; ++d) {
        std::vector<size_t> level_indices;
        for (size_t i = 0; i < order.size(); ++i) {
            if (order[i].depth == d) {
                level_indices.push_back(i);
            }
        }

        if (level_indices.empty()) continue;

        if (pool_->num_threads() <= 1 || level_indices.size() <= 1) {
            for (auto idx : level_indices) {
                execute_node(order[idx].node.get());
            }
        } else {
            pool_->parallel_for(level_indices.size(), [&](size_t i) {
                execute_node(order[level_indices[i]].node.get());
            });
        }
    }
}

void Graph::backward(const std::shared_ptr<Node>& output_node) {
    std::vector<std::shared_ptr<Node>> order;
    std::unordered_set<Node*> visited;
    reverse_topo(output_node, order, visited);

    for (auto& node : order) {
        if (node->gradient.size() == 0) {
            node->gradient = Tensor(node->output.shape());
            if (node == output_node) {
                node->gradient.data()[0] = 1.0f;
            }
        }
    }

    for (auto& node : order) {
        if (node->inputs.empty()) continue;

        switch (node->op_type) {
        case OpType::MATMUL: {
            auto& a = node->inputs[0];
            auto& b = node->inputs[1];
            Tensor grad_a(a->output.shape());
            Tensor grad_b(b->output.shape());
            Ops::matmul_backward(a->output, b->output, node->gradient,
                                 grad_a, grad_b);

            if (a->gradient.size() == 0) {
                a->gradient = grad_a;
            } else {
                Tensor tmp = Ops::add(a->gradient, grad_a);
                a->gradient = std::move(tmp);
            }

            if (b->gradient.size() == 0) {
                b->gradient = grad_b;
            } else {
                Tensor tmp = Ops::add(b->gradient, grad_b);
                b->gradient = std::move(tmp);
            }
            break;
        }

        default:
            break;
        }
    }
}

void Graph::zero_grad() {
    for (auto& node : nodes_) {
        node->gradient = Tensor();
    }
}

} // namespace cosq::qnet
