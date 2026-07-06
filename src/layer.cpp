#include <qnet/layer.hpp>

#include <cmath>
#include <random>
#include <stdexcept>

namespace cosq::qnet {

// ── Layer base ──────────────────────────────────────────────────────

Layer::Layer(Graph& graph, std::string name)
    : graph_(graph), name_(std::move(name)) {}

// ── Linear ──────────────────────────────────────────────────────────

Linear::Linear(Graph& graph, size_t in_features, size_t out_features,
               bool bias, std::string name)
    : Layer(graph, std::move(name)),
      in_features_(in_features),
      out_features_(out_features),
      has_bias_(bias) {
    if (in_features_ == 0 || out_features_ == 0) {
        throw std::invalid_argument("Linear: features must be > 0");
    }

    float bound = std::sqrt(6.0f / static_cast<float>(in_features_));
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-bound, bound);

    Tensor w({in_features_, out_features_});
    for (size_t i = 0; i < w.size(); ++i) w.data()[i] = dist(rng);
    weight_ = graph.parameter(w);

    if (has_bias_) {
        Tensor b({1, out_features_});
        bias_ = graph.parameter(b);
    }
}

std::shared_ptr<Node> Linear::forward(std::shared_ptr<Node> input) {
    auto y = graph_.matmul(input, weight_);
    if (has_bias_) {
        y = graph_.add(y, bias_);
    }
    return y;
}

std::vector<std::shared_ptr<Node>> Linear::parameters() {
    auto params = std::vector<std::shared_ptr<Node>>{weight_};
    if (has_bias_) params.push_back(bias_);
    return params;
}

// ── Conv2d ──────────────────────────────────────────────────────────

Conv2d::Conv2d(Graph& graph, size_t in_channels, size_t out_channels,
               size_t kernel_size, size_t stride, size_t pad,
               bool bias, std::string name)
    : Layer(graph, std::move(name)),
      in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      pad_(pad),
      has_bias_(bias) {
    if (in_channels_ == 0 || out_channels_ == 0 || kernel_size_ == 0) {
        throw std::invalid_argument("Conv2d: dimensions must be > 0");
    }

    float bound = std::sqrt(6.0f / static_cast<float>(in_channels_ * kernel_size_ * kernel_size_));
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-bound, bound);

    Tensor w({out_channels_, in_channels_, kernel_size_, kernel_size_});
    for (size_t i = 0; i < w.size(); ++i) w.data()[i] = dist(rng);
    weight_ = graph.parameter(w);

    if (has_bias_) {
        Tensor b({1, out_channels_, 1, 1});
        bias_ = graph.parameter(b);
    }
}

std::shared_ptr<Node> Conv2d::forward(std::shared_ptr<Node> input) {
    auto y = graph_.conv2d(input, weight_, stride_, stride_, pad_, pad_);
    if (has_bias_) {
        y = graph_.add(y, bias_);
    }
    return y;
}

std::vector<std::shared_ptr<Node>> Conv2d::parameters() {
    auto params = std::vector<std::shared_ptr<Node>>{weight_};
    if (has_bias_) params.push_back(bias_);
    return params;
}

// ── Embedding ───────────────────────────────────────────────────────

Embedding::Embedding(Graph& graph, size_t vocab_size, size_t embedding_dim,
                     std::string name)
    : Layer(graph, std::move(name)),
      vocab_size_(vocab_size),
      embedding_dim_(embedding_dim) {
    if (vocab_size_ == 0 || embedding_dim_ == 0) {
        throw std::invalid_argument("Embedding: dimensions must be > 0");
    }

    float bound = 1.0f / std::sqrt(static_cast<float>(embedding_dim_));
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-bound, bound);

    Tensor w({vocab_size_, embedding_dim_});
    for (size_t i = 0; i < w.size(); ++i) w.data()[i] = dist(rng);
    weight_ = graph.parameter(w);
}

std::shared_ptr<Node> Embedding::forward(std::shared_ptr<Node> input) {
    return graph_.embedding(weight_, input);
}

std::vector<std::shared_ptr<Node>> Embedding::parameters() {
    return {weight_};
}

// ── Dropout ─────────────────────────────────────────────────────────

Dropout::Dropout(Graph& graph, float rate, std::string name)
    : Layer(graph, std::move(name)), rate_(rate) {
    if (rate < 0.0f || rate >= 1.0f) {
        throw std::invalid_argument("Dropout: rate must be in [0, 1)");
    }
}

std::shared_ptr<Node> Dropout::forward(std::shared_ptr<Node> input) {
    if (!is_training() || rate_ == 0.0f) {
        return input;
    }
    return graph_.dropout(input, rate_);
}

// ── LayerNorm ───────────────────────────────────────────────────────

LayerNorm::LayerNorm(Graph& graph, size_t normalized_shape, float eps,
                     bool elementwise_affine, std::string name)
    : Layer(graph, std::move(name)),
      normalized_shape_(normalized_shape),
      eps_(eps),
      elementwise_affine_(elementwise_affine) {
    if (normalized_shape_ == 0) {
        throw std::invalid_argument("LayerNorm: normalized_shape must be > 0");
    }

    if (elementwise_affine_) {
        Tensor g(std::vector<size_t>{normalized_shape_});
        for (size_t i = 0; i < g.size(); ++i) g.data()[i] = 1.0f;
        Tensor b(std::vector<size_t>{normalized_shape_});
        gamma_ = graph.parameter(g);
        beta_ = graph.parameter(b);
    }
}

std::shared_ptr<Node> LayerNorm::forward(std::shared_ptr<Node> input) {
    if (elementwise_affine_) {
        return graph_.layer_norm(input, gamma_, beta_, eps_);
    }
    return graph_.layer_norm(input, nullptr, nullptr, eps_);
}

std::vector<std::shared_ptr<Node>> LayerNorm::parameters() {
    if (!elementwise_affine_) return {};
    return {gamma_, beta_};
}

// ── Sequential ──────────────────────────────────────────────────────

Sequential::Sequential(Graph& graph, std::string name)
    : Layer(graph, std::move(name)) {}

void Sequential::add(std::unique_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
}

std::shared_ptr<Node> Sequential::forward(std::shared_ptr<Node> input) {
    for (auto& layer : layers_) {
        layer->train(is_training());
        input = layer->forward(input);
    }
    return input;
}

std::vector<std::shared_ptr<Node>> Sequential::parameters() {
    std::vector<std::shared_ptr<Node>> params;
    for (auto& layer : layers_) {
        auto p = layer->parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}

// ── Model ───────────────────────────────────────────────────────────

Model::Model(Graph& graph, std::string name)
    : Layer(graph, std::move(name)) {}

void Model::add(std::unique_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
}

std::shared_ptr<Node> Model::forward(std::shared_ptr<Node> input) {
    for (auto& layer : layers_) {
        layer->train(is_training());
        input = layer->forward(input);
    }
    return input;
}

std::vector<std::shared_ptr<Node>> Model::parameters() {
    std::vector<std::shared_ptr<Node>> params;
    for (auto& layer : layers_) {
        auto p = layer->parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}

void Model::train(bool mode) {
    Layer::train(mode);
    for (auto& layer : layers_) {
        layer->train(mode);
    }
}

} // namespace cosq::qnet
