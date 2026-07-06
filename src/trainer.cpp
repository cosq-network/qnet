#include <qnet/serializer.hpp>
#include <qnet/trainer.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace cosq::qnet {

namespace {

bool supports_metric(MetricType metric, OpType loss_type) {
    switch (metric) {
    case MetricType::NONE:
        return true;
    case MetricType::CLASSIFICATION_ACCURACY:
        return loss_type == OpType::CROSS_ENTROPY_LOSS;
    default:
        return false;
    }
}

float batch_classification_accuracy(const Tensor& logits, const Tensor& targets) {
    if (logits.ndim() != 2) {
        throw std::invalid_argument("classification accuracy requires 2D logits");
    }
    if (targets.ndim() != 1) {
        throw std::invalid_argument("classification accuracy requires 1D targets");
    }
    if (logits.shape()[0] != targets.shape()[0]) {
        throw std::invalid_argument("classification accuracy batch size mismatch");
    }

    size_t batch = logits.shape()[0];
    size_t classes = logits.shape()[1];
    if (batch == 0 || classes == 0) {
        return 0.0f;
    }

    size_t correct = 0;
    for (size_t b = 0; b < batch; ++b) {
        size_t best_index = 0;
        float best_value = logits.at({b, 0});
        for (size_t c = 1; c < classes; ++c) {
            float value = logits.at({b, c});
            if (value > best_value) {
                best_value = value;
                best_index = c;
            }
        }

        long target_index = static_cast<long>(std::lround(targets.at({b})));
        if (target_index < 0 || static_cast<size_t>(target_index) >= classes) {
            throw std::out_of_range("classification accuracy target index out of bounds");
        }
        if (best_index == static_cast<size_t>(target_index)) {
            ++correct;
        }
    }

    return static_cast<float>(correct) / static_cast<float>(batch);
}

void validate_fit_options(const FitOptions& options) {
    if (options.early_stopping_min_delta < 0.0f) {
        throw std::invalid_argument("FitOptions early_stopping_min_delta must be non-negative");
    }

    bool uses_validation_only_features =
        options.early_stopping_patience > 0 ||
        options.restore_best_checkpoint ||
        !options.best_parameter_path.empty() ||
        !options.best_optimizer_state_path.empty();

    if (uses_validation_only_features && !options.validation_loader) {
        throw std::invalid_argument(
            "FitOptions validation_loader is required for early stopping and best-checkpoint tracking");
    }

    bool has_best_parameter_path = !options.best_parameter_path.empty();
    bool has_best_optimizer_path = !options.best_optimizer_state_path.empty();
    if (has_best_parameter_path != has_best_optimizer_path) {
        throw std::invalid_argument(
            "FitOptions best_parameter_path and best_optimizer_state_path must both be set or both be empty");
    }

    if (options.restore_best_checkpoint && !has_best_parameter_path) {
        throw std::invalid_argument(
            "FitOptions restore_best_checkpoint requires best checkpoint paths");
    }
}

} // namespace

Trainer::Trainer(Graph& graph,
                 Optimizer& optimizer,
                 std::shared_ptr<Node> input_node,
                 std::shared_ptr<Node> target_node,
                 std::shared_ptr<Node> loss_node)
    : graph_(graph),
      optimizer_(optimizer),
      input_node_(std::move(input_node)),
      target_node_(std::move(target_node)),
      loss_node_(std::move(loss_node)) {
    if (!input_node_ || !target_node_ || !loss_node_) {
        throw std::invalid_argument("Trainer requires non-null graph nodes");
    }
}

float Trainer::train_batch(const Tensor& inputs, const Tensor& targets) {
    input_node_->output = inputs;
    target_node_->output = targets;

    graph_.zero_grad();
    optimizer_.zero_grad();

    graph_.forward(loss_node_);
    graph_.backward(loss_node_);

    float loss = loss_node_->output.data()[0];

    optimizer_.step();
    optimizer_.zero_grad();
    graph_.zero_grad();

    return loss;
}

float Trainer::evaluate_batch(const Tensor& inputs, const Tensor& targets) {
    input_node_->output = inputs;
    target_node_->output = targets;

    graph_.zero_grad();
    optimizer_.zero_grad();
    graph_.forward(loss_node_);
    float loss = loss_node_->output.data()[0];
    graph_.zero_grad();

    return loss;
}

float Trainer::evaluate(DataLoader& loader) {
    loader.reset();

    float loss_sum = 0.0f;
    size_t batch_count = 0;
    while (loader.has_next()) {
        Batch batch = loader.next();
        loss_sum += evaluate_batch(batch.inputs, batch.targets);
        ++batch_count;
    }

    if (batch_count == 0) {
        return 0.0f;
    }

    return loss_sum / static_cast<float>(batch_count);
}

float Trainer::evaluate_accuracy(DataLoader& loader) {
    if (loss_node_->op_type != OpType::CROSS_ENTROPY_LOSS) {
        throw std::logic_error("evaluate_accuracy requires a cross_entropy_loss trainer graph");
    }

    loader.reset();

    float accuracy_sum = 0.0f;
    size_t batch_count = 0;
    while (loader.has_next()) {
        Batch batch = loader.next();
        input_node_->output = batch.inputs;
        target_node_->output = batch.targets;

        graph_.zero_grad();
        optimizer_.zero_grad();
        graph_.forward(loss_node_);
        accuracy_sum += batch_classification_accuracy(loss_node_->inputs[0]->output, target_node_->output);
        graph_.zero_grad();
        ++batch_count;
    }

    if (batch_count == 0) {
        return 0.0f;
    }

    return accuracy_sum / static_cast<float>(batch_count);
}

std::vector<EpochMetrics> Trainer::fit(DataLoader& loader, size_t epochs,
                                       const FitOptions& options) {
    validate_fit_options(options);
    if (!supports_metric(options.metric, loss_node_->op_type)) {
        throw std::invalid_argument("requested trainer metric is incompatible with the current loss node");
    }

    std::vector<EpochMetrics> history;
    history.reserve(epochs);

    float best_validation_loss = std::numeric_limits<float>::infinity();
    size_t epochs_without_improvement = 0;
    bool have_best_checkpoint = false;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        loader.reset();

        float loss_sum = 0.0f;
        float accuracy_sum = 0.0f;
        size_t batch_count = 0;
        while (loader.has_next()) {
            Batch batch = loader.next();
            loss_sum += train_batch(batch.inputs, batch.targets);
            if (options.metric == MetricType::CLASSIFICATION_ACCURACY) {
                graph_.forward(loss_node_);
                accuracy_sum += batch_classification_accuracy(loss_node_->inputs[0]->output, target_node_->output);
            }
            ++batch_count;
        }

        EpochMetrics metrics{
            epoch + 1,
            batch_count,
            batch_count == 0 ? 0.0f : loss_sum / static_cast<float>(batch_count),
            options.metric == MetricType::CLASSIFICATION_ACCURACY && batch_count > 0
                ? accuracy_sum / static_cast<float>(batch_count)
                : 0.0f,
            options.metric == MetricType::CLASSIFICATION_ACCURACY,
            0,
            0.0f,
            0.0f,
            false,
            false,
        };

        if (options.validation_loader) {
            options.validation_loader->reset();
            float validation_loss_sum = 0.0f;
            float validation_accuracy_sum = 0.0f;
            size_t validation_batch_count = 0;
            while (options.validation_loader->has_next()) {
                Batch batch = options.validation_loader->next();
                validation_loss_sum += evaluate_batch(batch.inputs, batch.targets);
                if (options.metric == MetricType::CLASSIFICATION_ACCURACY) {
                    graph_.forward(loss_node_);
                    validation_accuracy_sum += batch_classification_accuracy(loss_node_->inputs[0]->output,
                                                                             target_node_->output);
                }
                ++validation_batch_count;
            }

            metrics.validation_batch_count = validation_batch_count;
            metrics.validation_loss = validation_batch_count == 0
                ? 0.0f
                : validation_loss_sum / static_cast<float>(validation_batch_count);
            metrics.validation_accuracy =
                options.metric == MetricType::CLASSIFICATION_ACCURACY && validation_batch_count > 0
                    ? validation_accuracy_sum / static_cast<float>(validation_batch_count)
                    : 0.0f;
            metrics.has_validation = true;

            if (validation_batch_count > 0) {
                bool improved =
                    metrics.validation_loss < (best_validation_loss - options.early_stopping_min_delta);

                if (improved || !std::isfinite(best_validation_loss)) {
                    best_validation_loss = metrics.validation_loss;
                    epochs_without_improvement = 0;
                    metrics.is_best = true;

                    if (!options.best_parameter_path.empty()) {
                        save_checkpoint(options.best_parameter_path, options.best_optimizer_state_path);
                        have_best_checkpoint = true;
                    }
                } else {
                    ++epochs_without_improvement;
                }
            }
        }

        if (options.on_epoch_end) {
            options.on_epoch_end(metrics);
        }

        history.push_back(metrics);

        if (options.validation_loader &&
            options.early_stopping_patience > 0 &&
            epochs_without_improvement >= options.early_stopping_patience) {
            break;
        }
    }

    if (options.restore_best_checkpoint && have_best_checkpoint) {
        load_checkpoint(options.best_parameter_path, options.best_optimizer_state_path);
    }

    return history;
}

void Trainer::save_checkpoint(const std::string& parameter_path,
                              const std::string& optimizer_state_path) const {
    GraphSerializer::save_parameters(graph_, parameter_path);
    optimizer_.save_state(optimizer_state_path);
}

void Trainer::load_checkpoint(const std::string& parameter_path,
                              const std::string& optimizer_state_path) {
    GraphSerializer::load_parameters(graph_, parameter_path);
    optimizer_.load_state(optimizer_state_path);
}

} // namespace cosq::qnet
