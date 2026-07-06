#pragma once

#include <qnet/data.hpp>
#include <qnet/graph.hpp>
#include <qnet/optimizer.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cosq::qnet {

enum class MetricType {
    NONE = 0,
    CLASSIFICATION_ACCURACY = 1,
};

struct QNET_API EpochMetrics {
    size_t epoch = 0;
    size_t batch_count = 0;
    float average_loss = 0.0f;
    float average_accuracy = 0.0f;
    bool has_accuracy = false;
    size_t validation_batch_count = 0;
    float validation_loss = 0.0f;
    float validation_accuracy = 0.0f;
    bool has_validation = false;
    bool is_best = false;
};

struct QNET_API FitOptions {
    DataLoader* validation_loader = nullptr;
    std::function<void(const EpochMetrics&)> on_epoch_end;
    MetricType metric = MetricType::NONE;
    size_t early_stopping_patience = 0;
    float early_stopping_min_delta = 0.0f;
    std::string best_parameter_path;
    std::string best_optimizer_state_path;
    bool restore_best_checkpoint = false;
};

class QNET_API Trainer {
public:
    Trainer(Graph& graph,
            Optimizer& optimizer,
            std::shared_ptr<Node> input_node,
            std::shared_ptr<Node> target_node,
            std::shared_ptr<Node> loss_node);

    float train_batch(const Tensor& inputs, const Tensor& targets);
    float evaluate_batch(const Tensor& inputs, const Tensor& targets);
    float evaluate(DataLoader& loader);
    float evaluate_accuracy(DataLoader& loader);

    std::vector<EpochMetrics> fit(DataLoader& loader, size_t epochs,
                                  const FitOptions& options = {});

    void save_checkpoint(const std::string& parameter_path,
                         const std::string& optimizer_state_path) const;

    void load_checkpoint(const std::string& parameter_path,
                         const std::string& optimizer_state_path);

private:
    Graph& graph_;
    Optimizer& optimizer_;
    std::shared_ptr<Node> input_node_;
    std::shared_ptr<Node> target_node_;
    std::shared_ptr<Node> loss_node_;
};

} // namespace cosq::qnet
