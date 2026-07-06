#pragma once

#include <qnet/export.hpp>
#include <qnet/node.hpp>
#include <qnet/tensor.hpp>
#include <qnet/thread_pool.hpp>

#include <memory>
#include <vector>

namespace cosq::qnet {

class QNET_API Graph {
public:
    explicit Graph(size_t num_threads = 0);
    ~Graph() = default;

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&&) = default;
    Graph& operator=(Graph&&) = default;

    std::shared_ptr<Node> variable(const Tensor& data);

    std::shared_ptr<Node> parameter(const Tensor& data);

    std::shared_ptr<Node> matmul(const std::shared_ptr<Node>& a,
                                 const std::shared_ptr<Node>& b);

    std::shared_ptr<Node> add(const std::shared_ptr<Node>& a,
                              const std::shared_ptr<Node>& b);

    std::shared_ptr<Node> relu(const std::shared_ptr<Node>& x);

    std::shared_ptr<Node> sigmoid(const std::shared_ptr<Node>& x);

    std::shared_ptr<Node> softmax(const std::shared_ptr<Node>& x);

    std::shared_ptr<Node> mul(const std::shared_ptr<Node>& a,
                              const std::shared_ptr<Node>& b);

    std::shared_ptr<Node> conv2d(const std::shared_ptr<Node>& input,
                                 const std::shared_ptr<Node>& kernel,
                                 size_t stride_h = 1, size_t stride_w = 1,
                                 size_t pad_h = 0, size_t pad_w = 0);

    std::shared_ptr<Node> embedding(const std::shared_ptr<Node>& weight,
                                   const std::shared_ptr<Node>& indices);

    void forward(const std::shared_ptr<Node>& output_node);

    void backward(const std::shared_ptr<Node>& output_node);

    void zero_grad();

    const std::vector<std::shared_ptr<Node>>& nodes() const { return nodes_; }

    void set_num_threads(size_t n);

    void add_node(std::shared_ptr<Node> node);

private:
    std::vector<std::shared_ptr<Node>> nodes_;
    std::unique_ptr<ThreadPool> pool_;

    std::shared_ptr<Node> create_node(OpType type,
                                      const std::vector<std::shared_ptr<Node>>& inputs);

    static void execute_node(Node* node);

    static void accumulate_grad(const std::shared_ptr<Node>& node,
                                const Tensor& grad);
};

} // namespace cosq::qnet
