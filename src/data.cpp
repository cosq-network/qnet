#include <qnet/data.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace cosq::qnet {

namespace {

void validate_tensor_dataset_shapes(const Tensor& inputs, const Tensor& targets) {
    if (inputs.ndim() == 0 || targets.ndim() == 0) {
        throw std::invalid_argument("TensorDataset requires tensors with at least 1 dimension");
    }
    if (inputs.shape()[0] == 0 || targets.shape()[0] == 0) {
        throw std::invalid_argument("TensorDataset requires non-empty tensors");
    }
    if (inputs.shape()[0] != targets.shape()[0]) {
        throw std::invalid_argument("TensorDataset input/target sample count mismatch");
    }
}

size_t sample_width(const Tensor& tensor) {
    return tensor.size() / tensor.shape()[0];
}

} // namespace

TensorDataset::TensorDataset(const Tensor& inputs, const Tensor& targets)
    : inputs_(inputs),
      targets_(targets) {
    validate_tensor_dataset_shapes(inputs_, targets_);
}

size_t TensorDataset::size() const {
    return inputs_.shape()[0];
}

Sample TensorDataset::get(size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("TensorDataset index out of range");
    }

    return Sample{
        inputs_.slice(0, index, index + 1).clone(),
        targets_.slice(0, index, index + 1).clone(),
    };
}

DataLoader::DataLoader(const Dataset& dataset,
                       size_t batch_size,
                       bool shuffle,
                       uint32_t seed)
    : dataset_(dataset),
      batch_size_(batch_size),
      shuffle_(shuffle),
      rng_(seed) {
    if (batch_size_ == 0) {
        throw std::invalid_argument("DataLoader batch_size must be > 0");
    }
    reset();
}

bool DataLoader::has_next() const {
    return cursor_ < order_.size();
}

Batch DataLoader::next() {
    if (!has_next()) {
        throw std::out_of_range("DataLoader has no remaining batches");
    }

    size_t end = std::min(cursor_ + batch_size_, order_.size());
    std::vector<Tensor> inputs;
    std::vector<Tensor> targets;
    std::vector<size_t> indices;
    inputs.reserve(end - cursor_);
    targets.reserve(end - cursor_);
    indices.reserve(end - cursor_);

    for (size_t i = cursor_; i < end; ++i) {
        size_t index = order_[i];
        Sample sample = dataset_.get(index);
        inputs.push_back(std::move(sample.input));
        targets.push_back(std::move(sample.target));
        indices.push_back(index);
    }

    cursor_ = end;

    return Batch{
        stack_samples(inputs),
        stack_samples(targets),
        std::move(indices),
    };
}

void DataLoader::reset() {
    order_.resize(dataset_.size());
    std::iota(order_.begin(), order_.end(), size_t{0});
    if (shuffle_) {
        std::shuffle(order_.begin(), order_.end(), rng_);
    }
    cursor_ = 0;
}

size_t DataLoader::num_batches() const {
    if (order_.empty()) {
        return 0;
    }
    return (order_.size() + batch_size_ - 1) / batch_size_;
}

Tensor DataLoader::stack_samples(const std::vector<Tensor>& samples) {
    if (samples.empty()) {
        throw std::invalid_argument("cannot stack an empty sample list");
    }

    const Tensor& first = samples.front();
    if (first.ndim() == 0 || first.shape()[0] != 1) {
        throw std::invalid_argument("DataLoader samples must have leading batch dimension 1");
    }

    std::vector<size_t> batch_shape = first.shape();
    batch_shape[0] = samples.size();

    Tensor batch(batch_shape);
    size_t row_width = sample_width(first);

    for (size_t i = 0; i < samples.size(); ++i) {
        const Tensor& sample = samples[i];
        if (sample.shape() != first.shape()) {
            throw std::invalid_argument("DataLoader requires consistent sample shapes");
        }

        std::copy_n(sample.data(), row_width, batch.data() + i * row_width);
    }

    return batch;
}

} // namespace cosq::qnet
