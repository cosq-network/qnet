#pragma once

#include <qnet/graph.hpp>
#include <qnet/ops.hpp>

#include <memory>
#include <string>
#include <vector>

namespace cosq::qnet {

class Layer {
public:
    explicit Layer(Graph& graph, std::string name = {});
    virtual ~Layer() = default;

    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
    Layer(Layer&&) = delete;
    Layer& operator=(Layer&&) = delete;

    virtual std::shared_ptr<Node> forward(std::shared_ptr<Node> input) = 0;

    virtual std::vector<std::shared_ptr<Node>> parameters() { return {}; }

    const std::string& name() const { return name_; }
    void set_name(const std::string& name) { name_ = name; }

    virtual void train(bool mode = true) { training_ = mode; }
    bool is_training() const { return training_; }

protected:
    Graph& graph_;
    std::string name_;
    bool training_ = true;
};

class Linear : public Layer {
public:
    Linear(Graph& graph, size_t in_features, size_t out_features, bool bias = true,
           std::string name = {});

    std::shared_ptr<Node> forward(std::shared_ptr<Node> input) override;

    std::vector<std::shared_ptr<Node>> parameters() override;

    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }

    const std::shared_ptr<Node>& weight() const { return weight_; }
    const std::shared_ptr<Node>& bias() const { return bias_; }

private:
    size_t in_features_;
    size_t out_features_;
    bool has_bias_;
    std::shared_ptr<Node> weight_;
    std::shared_ptr<Node> bias_;
};

class Conv2d : public Layer {
public:
    Conv2d(Graph& graph, size_t in_channels, size_t out_channels,
           size_t kernel_size, size_t stride = 1, size_t pad = 0,
           bool bias = true, std::string name = {});

    std::shared_ptr<Node> forward(std::shared_ptr<Node> input) override;

    std::vector<std::shared_ptr<Node>> parameters() override;

    size_t in_channels() const { return in_channels_; }
    size_t out_channels() const { return out_channels_; }

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t stride_;
    size_t pad_;
    bool has_bias_;
    std::shared_ptr<Node> weight_;
    std::shared_ptr<Node> bias_;
};

class Embedding : public Layer {
public:
    Embedding(Graph& graph, size_t vocab_size, size_t embedding_dim,
              std::string name = {});

    std::shared_ptr<Node> forward(std::shared_ptr<Node> input) override;

    std::vector<std::shared_ptr<Node>> parameters() override;

    size_t vocab_size() const { return vocab_size_; }
    size_t embedding_dim() const { return embedding_dim_; }

private:
    size_t vocab_size_;
    size_t embedding_dim_;
    std::shared_ptr<Node> weight_;
};

class Dropout : public Layer {
public:
    Dropout(Graph& graph, float rate = 0.5f, std::string name = {});

    std::shared_ptr<Node> forward(std::shared_ptr<Node> input) override;

    float rate() const { return rate_; }

private:
    float rate_;
    std::shared_ptr<Node> drop_node_;
};

class LayerNorm : public Layer {
public:
    LayerNorm(Graph& graph, size_t normalized_shape, float eps = 1e-5f,
              bool elementwise_affine = true, std::string name = {});

    std::shared_ptr<Node> forward(std::shared_ptr<Node> input) override;

    std::vector<std::shared_ptr<Node>> parameters() override;

    size_t normalized_shape() const { return normalized_shape_; }

private:
    size_t normalized_shape_;
    float eps_;
    bool elementwise_affine_;
    std::shared_ptr<Node> gamma_;
    std::shared_ptr<Node> beta_;
};

class Sequential : public Layer {
public:
    Sequential(Graph& graph, std::string name = {});

    void add(std::unique_ptr<Layer> layer);

    std::shared_ptr<Node> forward(std::shared_ptr<Node> input) override;

    std::vector<std::shared_ptr<Node>> parameters() override;

    Layer& operator[](size_t index) { return *layers_[index]; }
    const Layer& operator[](size_t index) const { return *layers_[index]; }
    size_t size() const { return layers_.size(); }

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};

class Model : public Layer {
public:
    Model(Graph& graph, std::string name = {});

    void add(std::unique_ptr<Layer> layer);

    std::shared_ptr<Node> forward(std::shared_ptr<Node> input) override;

    std::vector<std::shared_ptr<Node>> parameters() override;

    void train(bool mode) override;
    void eval() { train(false); }

    Layer& operator[](size_t index) { return *layers_[index]; }
    const Layer& operator[](size_t index) const { return *layers_[index]; }
    size_t size() const { return layers_.size(); }

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};

} // namespace cosq::qnet
