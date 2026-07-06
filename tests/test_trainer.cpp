#include <qnet/data.hpp>
#include <qnet/layer.hpp>
#include <qnet/optimizer.hpp>
#include <qnet/trainer.hpp>

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>

using namespace cosq::qnet;

static bool approx(float a, float b, float eps = 1e-4f) {
    return std::abs(a - b) < eps;
}

static void test_dataloader_batches() {
    Tensor inputs({3, 2}, {1, 2,
                           3, 4,
                           5, 6});
    Tensor targets({3, 1}, {10, 20, 30});

    TensorDataset dataset(inputs, targets);
    DataLoader loader(dataset, 2, false);

    assert(loader.num_batches() == 2);

    Batch first = loader.next();
    assert(first.inputs.shape()[0] == 2);
    assert(first.inputs.shape()[1] == 2);
    assert(first.targets.shape()[0] == 2);
    assert(first.targets.shape()[1] == 1);
    assert(approx(first.inputs.at({0, 0}), 1.0f));
    assert(approx(first.inputs.at({1, 1}), 4.0f));
    assert(approx(first.targets.at({1, 0}), 20.0f));

    Batch second = loader.next();
    assert(second.inputs.shape()[0] == 1);
    assert(second.inputs.shape()[1] == 2);
    assert(approx(second.inputs.at({0, 0}), 5.0f));
    assert(approx(second.targets.at({0, 0}), 30.0f));

    std::cout << "  [PASS] test_dataloader_batches\n";
}

static void test_trainer_fit_reduces_loss() {
    Graph graph;
    Model model(graph);
    model.add(std::make_unique<Linear>(graph, 2, 1, true));

    auto input = graph.variable(Tensor({1, 2}, {0.0f, 0.0f}));
    auto target = graph.variable(Tensor({1, 1}, {0.0f}));
    auto pred = model.forward(input);
    auto loss = graph.mse_loss(pred, target);

    SGD optimizer(model.parameters(), 0.05f, 0.9f);
    Trainer trainer(graph, optimizer, input, target, loss);

    Tensor inputs({4, 2}, {0, 0,
                           0, 1,
                           1, 0,
                           1, 1});
    Tensor targets({4, 1}, {0, 1, 1, 2});

    TensorDataset dataset(inputs, targets);
    DataLoader loader(dataset, 2, true, 1234u);

    loader.reset();
    Batch first_batch = loader.next();
    graph.zero_grad();
    input->output = first_batch.inputs;
    target->output = first_batch.targets;
    graph.forward(loss);
    float initial_loss = loss->output.data()[0];

    std::vector<EpochMetrics> history = trainer.fit(loader, 80);
    assert(!history.empty());
    assert(history.back().average_loss < initial_loss);

    loader.reset();
    Batch eval_batch = loader.next();
    graph.zero_grad();
    input->output = eval_batch.inputs;
    target->output = eval_batch.targets;
    graph.forward(loss);
    assert(loss->output.data()[0] < initial_loss);

    std::cout << "  [PASS] test_trainer_fit_reduces_loss\n";
}

static void test_trainer_evaluate_is_non_mutating() {
    Graph graph;
    Model model(graph);
    model.add(std::make_unique<Linear>(graph, 2, 1, true));

    auto input = graph.variable(Tensor({1, 2}, {0.0f, 0.0f}));
    auto target = graph.variable(Tensor({1, 1}, {0.0f}));
    auto pred = model.forward(input);
    auto loss = graph.mse_loss(pred, target);

    SGD optimizer(model.parameters(), 0.05f, 0.9f);
    Trainer trainer(graph, optimizer, input, target, loss);

    Tensor inputs({4, 2}, {0, 0,
                           0, 1,
                           1, 0,
                           1, 1});
    Tensor targets({4, 1}, {0, 1, 1, 2});

    TensorDataset dataset(inputs, targets);
    DataLoader loader(dataset, 2, false);

    auto before = model.parameters();
    std::vector<std::vector<float>> snapshot;
    snapshot.reserve(before.size());
    for (const auto& parameter : before) {
        snapshot.emplace_back(parameter->output.data(),
                              parameter->output.data() + parameter->output.size());
    }

    float eval_loss = trainer.evaluate(loader);
    assert(eval_loss >= 0.0f);

    auto after = model.parameters();
    assert(before.size() == after.size());
    for (size_t i = 0; i < after.size(); ++i) {
        for (size_t j = 0; j < after[i]->output.size(); ++j) {
            assert(approx(after[i]->output.data()[j], snapshot[i][j], 1e-6f));
        }
    }

    std::cout << "  [PASS] test_trainer_evaluate_is_non_mutating\n";
}

static void test_trainer_fit_reports_validation_and_callback() {
    Graph graph;
    Model model(graph);
    model.add(std::make_unique<Linear>(graph, 2, 1, true));

    auto input = graph.variable(Tensor({1, 2}, {0.0f, 0.0f}));
    auto target = graph.variable(Tensor({1, 1}, {0.0f}));
    auto pred = model.forward(input);
    auto loss = graph.mse_loss(pred, target);

    SGD optimizer(model.parameters(), 0.05f, 0.9f);
    Trainer trainer(graph, optimizer, input, target, loss);

    Tensor train_inputs({4, 2}, {0, 0,
                                 0, 1,
                                 1, 0,
                                 1, 1});
    Tensor train_targets({4, 1}, {0, 1, 1, 2});
    Tensor val_inputs({2, 2}, {0, 0,
                               1, 1});
    Tensor val_targets({2, 1}, {0, 2});

    TensorDataset train_dataset(train_inputs, train_targets);
    TensorDataset val_dataset(val_inputs, val_targets);
    DataLoader train_loader(train_dataset, 2, true, 99u);
    DataLoader val_loader(val_dataset, 1, false);

    size_t callback_count = 0;
    std::vector<EpochMetrics> callback_metrics;
    FitOptions options;
    options.validation_loader = &val_loader;
    options.on_epoch_end = [&](const EpochMetrics& metrics) {
        ++callback_count;
        callback_metrics.push_back(metrics);
    };

    std::vector<EpochMetrics> history = trainer.fit(train_loader, 5, options);
    assert(history.size() == 5);
    assert(callback_count == 5);
    assert(callback_metrics.size() == history.size());

    for (size_t i = 0; i < history.size(); ++i) {
        assert(history[i].has_validation);
        assert(history[i].validation_batch_count == 2);
        assert(history[i].validation_loss >= 0.0f);
        assert(history[i].epoch == i + 1);
        assert(!history[i].has_accuracy);
        assert(approx(history[i].validation_loss, callback_metrics[i].validation_loss, 1e-6f));
    }

    float eval_loss = trainer.evaluate(val_loader);
    assert(approx(eval_loss, history.back().validation_loss, 1e-6f));

    std::cout << "  [PASS] test_trainer_fit_reports_validation_and_callback\n";
}

static void test_trainer_classification_accuracy_metric() {
    Graph graph;

    auto input = graph.variable(Tensor({1, 2}, {0.0f, 0.0f}));
    auto target = graph.variable(Tensor({1}, {0.0f}));
    auto weight = graph.parameter(Tensor({2, 2}, {1.0f, -1.0f,
                                                  -1.0f, 1.0f}));
    auto logits = graph.matmul(input, weight);
    auto loss = graph.cross_entropy_loss(logits, target);

    SGD optimizer(graph.parameters(), 0.1f);
    Trainer trainer(graph, optimizer, input, target, loss);

    Tensor inputs({4, 2}, {2.0f, 0.0f,
                           0.0f, 2.0f,
                           3.0f, 1.0f,
                           1.0f, 3.0f});
    Tensor targets({4}, {0.0f, 1.0f, 0.0f, 1.0f});

    TensorDataset dataset(inputs, targets);
    DataLoader loader(dataset, 2, false);
    DataLoader validation_loader(dataset, 2, false);

    FitOptions options;
    options.validation_loader = &validation_loader;
    options.metric = MetricType::CLASSIFICATION_ACCURACY;

    std::vector<EpochMetrics> history = trainer.fit(loader, 1, options);
    assert(history.size() == 1);
    assert(history[0].has_accuracy);
    assert(approx(history[0].average_accuracy, 1.0f, 1e-6f));
    assert(approx(history[0].validation_accuracy, 1.0f, 1e-6f));

    float eval_accuracy = trainer.evaluate_accuracy(validation_loader);
    assert(approx(eval_accuracy, 1.0f, 1e-6f));

    std::cout << "  [PASS] test_trainer_classification_accuracy_metric\n";
}

static void test_trainer_metric_validation() {
    Graph graph;
    Model model(graph);
    model.add(std::make_unique<Linear>(graph, 2, 1, true));

    auto input = graph.variable(Tensor({1, 2}, {0.0f, 0.0f}));
    auto target = graph.variable(Tensor({1, 1}, {0.0f}));
    auto pred = model.forward(input);
    auto loss = graph.mse_loss(pred, target);

    SGD optimizer(model.parameters(), 0.05f, 0.9f);
    Trainer trainer(graph, optimizer, input, target, loss);

    Tensor inputs({2, 2}, {0, 0,
                           1, 1});
    Tensor targets({2, 1}, {0, 2});
    TensorDataset dataset(inputs, targets);
    DataLoader loader(dataset, 1, false);

    FitOptions options;
    options.metric = MetricType::CLASSIFICATION_ACCURACY;

    bool threw = false;
    try {
        (void)trainer.fit(loader, 1, options);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        (void)trainer.evaluate_accuracy(loader);
    } catch (const std::logic_error&) {
        threw = true;
    }
    assert(threw);

    std::cout << "  [PASS] test_trainer_metric_validation\n";
}

static void test_trainer_early_stopping_and_best_restore() {
    Graph graph;
    Model model(graph);
    model.add(std::make_unique<Linear>(graph, 2, 1, true));

    auto input = graph.variable(Tensor({1, 2}, {0.0f, 0.0f}));
    auto target = graph.variable(Tensor({1, 1}, {0.0f}));
    auto pred = model.forward(input);
    auto loss = graph.mse_loss(pred, target);

    SGD optimizer(model.parameters(), 0.05f, 0.9f);
    Trainer trainer(graph, optimizer, input, target, loss);

    Tensor train_inputs({4, 2}, {0, 0,
                                 0, 1,
                                 1, 0,
                                 1, 1});
    Tensor train_targets({4, 1}, {0, 1, 1, 2});
    Tensor val_inputs({2, 2}, {0, 0,
                               1, 1});
    Tensor val_targets({2, 1}, {0, 2});

    TensorDataset train_dataset(train_inputs, train_targets);
    TensorDataset val_dataset(val_inputs, val_targets);
    DataLoader train_loader(train_dataset, 2, true, 33u);
    DataLoader val_loader(val_dataset, 1, false);

    std::filesystem::path base = std::filesystem::temp_directory_path() / "qnet_early_stop";
    std::filesystem::path parameter_path = base;
    parameter_path += ".params";
    std::filesystem::path optimizer_path = base;
    optimizer_path += ".opt";

    FitOptions options;
    options.validation_loader = &val_loader;
    options.early_stopping_patience = 2;
    options.early_stopping_min_delta = 1e6f;
    options.best_parameter_path = parameter_path.string();
    options.best_optimizer_state_path = optimizer_path.string();
    options.restore_best_checkpoint = true;

    std::vector<EpochMetrics> history = trainer.fit(train_loader, 10, options);
    assert(history.size() == 3);
    assert(history[0].is_best);
    assert(!history[1].is_best);
    assert(!history[2].is_best);

    Graph restored_graph;
    Model restored_model(restored_graph);
    restored_model.add(std::make_unique<Linear>(restored_graph, 2, 1, true));

    auto restored_input = restored_graph.variable(Tensor({1, 2}, {0.0f, 0.0f}));
    auto restored_target = restored_graph.variable(Tensor({1, 1}, {0.0f}));
    auto restored_pred = restored_model.forward(restored_input);
    auto restored_loss = restored_graph.mse_loss(restored_pred, restored_target);
    (void)restored_loss;

    SGD restored_optimizer(restored_model.parameters(), 0.01f);
    Trainer restored_trainer(restored_graph, restored_optimizer,
                             restored_input, restored_target, restored_loss);
    restored_trainer.load_checkpoint(parameter_path.string(), optimizer_path.string());

    auto params = model.parameters();
    auto restored_params = restored_model.parameters();
    assert(params.size() == restored_params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        for (size_t j = 0; j < params[i]->output.size(); ++j) {
            assert(approx(params[i]->output.data()[j],
                          restored_params[i]->output.data()[j], 1e-6f));
        }
    }

    std::filesystem::remove(parameter_path);
    std::filesystem::remove(optimizer_path);

    std::cout << "  [PASS] test_trainer_early_stopping_and_best_restore\n";
}

static void test_trainer_fit_option_validation() {
    Graph graph;
    Model model(graph);
    model.add(std::make_unique<Linear>(graph, 2, 1, true));

    auto input = graph.variable(Tensor({1, 2}, {0.0f, 0.0f}));
    auto target = graph.variable(Tensor({1, 1}, {0.0f}));
    auto pred = model.forward(input);
    auto loss = graph.mse_loss(pred, target);

    SGD optimizer(model.parameters(), 0.05f, 0.9f);
    Trainer trainer(graph, optimizer, input, target, loss);

    Tensor inputs({2, 2}, {0, 0,
                           1, 1});
    Tensor targets({2, 1}, {0, 2});
    TensorDataset dataset(inputs, targets);
    DataLoader loader(dataset, 1, false);

    FitOptions options;
    options.early_stopping_patience = 1;

    bool threw = false;
    try {
        (void)trainer.fit(loader, 3, options);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    std::cout << "  [PASS] test_trainer_fit_option_validation\n";
}

static void test_trainer_checkpoint_roundtrip() {
    Graph graph_a;
    Model model_a(graph_a);
    model_a.add(std::make_unique<Linear>(graph_a, 2, 1, true));

    auto input_a = graph_a.variable(Tensor({1, 2}, {0.0f, 0.0f}));
    auto target_a = graph_a.variable(Tensor({1, 1}, {0.0f}));
    auto pred_a = model_a.forward(input_a);
    auto loss_a = graph_a.mse_loss(pred_a, target_a);

    SGD optimizer_a(model_a.parameters(), 0.05f, 0.9f);
    Trainer trainer_a(graph_a, optimizer_a, input_a, target_a, loss_a);

    Tensor inputs({4, 2}, {0, 0,
                           0, 1,
                           1, 0,
                           1, 1});
    Tensor targets({4, 1}, {0, 1, 1, 2});

    TensorDataset dataset(inputs, targets);
    DataLoader loader(dataset, 2, true, 7u);
    trainer_a.fit(loader, 20);

    std::filesystem::path base = std::filesystem::temp_directory_path() / "qnet_trainer_checkpoint";
    std::filesystem::path parameter_path = base;
    parameter_path += ".params";
    std::filesystem::path optimizer_path = base;
    optimizer_path += ".opt";
    trainer_a.save_checkpoint(parameter_path.string(), optimizer_path.string());

    Graph graph_b;
    Model model_b(graph_b);
    model_b.add(std::make_unique<Linear>(graph_b, 2, 1, true));

    auto input_b = graph_b.variable(Tensor({1, 2}, {0.0f, 0.0f}));
    auto target_b = graph_b.variable(Tensor({1, 1}, {0.0f}));
    auto pred_b = model_b.forward(input_b);
    auto loss_b = graph_b.mse_loss(pred_b, target_b);

    SGD optimizer_b(model_b.parameters(), 0.01f);
    Trainer trainer_b(graph_b, optimizer_b, input_b, target_b, loss_b);
    trainer_b.load_checkpoint(parameter_path.string(), optimizer_path.string());

    Batch batch = DataLoader(dataset, 2, false).next();
    float loss_after_a = trainer_a.train_batch(batch.inputs, batch.targets);
    float loss_after_b = trainer_b.train_batch(batch.inputs, batch.targets);

    assert(approx(loss_after_a, loss_after_b, 1e-5f));

    auto params_a = model_a.parameters();
    auto params_b = model_b.parameters();
    assert(params_a.size() == params_b.size());
    for (size_t i = 0; i < params_a.size(); ++i) {
        assert(params_a[i]->output.shape() == params_b[i]->output.shape());
        for (size_t j = 0; j < params_a[i]->output.size(); ++j) {
            assert(approx(params_a[i]->output.data()[j], params_b[i]->output.data()[j], 1e-5f));
        }
    }

    std::filesystem::remove(parameter_path);
    std::filesystem::remove(optimizer_path);

    std::cout << "  [PASS] test_trainer_checkpoint_roundtrip\n";
}

int main() {
    std::cout << "Trainer tests:\n";
    test_dataloader_batches();
    test_trainer_fit_reduces_loss();
    test_trainer_evaluate_is_non_mutating();
    test_trainer_fit_reports_validation_and_callback();
    test_trainer_classification_accuracy_metric();
    test_trainer_metric_validation();
    test_trainer_early_stopping_and_best_restore();
    test_trainer_fit_option_validation();
    test_trainer_checkpoint_roundtrip();
    std::cout << "All trainer tests passed!\n";
    return 0;
}
