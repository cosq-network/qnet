#pragma once

#include <qnet/export.hpp>
#include <qnet/tensor.hpp>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace cosq::qnet {

struct QNET_API Sample {
    Tensor input;
    Tensor target;
};

struct QNET_API Batch {
    Tensor inputs;
    Tensor targets;
    std::vector<size_t> indices;
};

class QNET_API Dataset {
public:
    virtual ~Dataset() = default;

    virtual size_t size() const = 0;
    virtual Sample get(size_t index) const = 0;
};

class QNET_API TensorDataset : public Dataset {
public:
    TensorDataset(const Tensor& inputs, const Tensor& targets);

    size_t size() const override;
    Sample get(size_t index) const override;

    const Tensor& inputs() const { return inputs_; }
    const Tensor& targets() const { return targets_; }

private:
    Tensor inputs_;
    Tensor targets_;
};

class QNET_API DataLoader {
public:
    DataLoader(const Dataset& dataset,
               size_t batch_size,
               bool shuffle = false,
               uint32_t seed = 5489u);

    bool has_next() const;
    Batch next();
    void reset();

    size_t batch_size() const { return batch_size_; }
    size_t num_batches() const;

private:
    const Dataset& dataset_;
    size_t batch_size_;
    bool shuffle_;
    std::mt19937 rng_;
    std::vector<size_t> order_;
    size_t cursor_ = 0;

    static Tensor stack_samples(const std::vector<Tensor>& samples);
};

} // namespace cosq::qnet
