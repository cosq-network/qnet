#include <qnet/tensor.hpp>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace cosq::qnet {

Tensor::Tensor(const std::vector<size_t>& shape)
    : shape_(shape), offset_(0), owns_data_(true) {
    if (shape_.empty()) {
        throw std::invalid_argument("Tensor shape cannot be empty");
    }
    compute_strides();
    data_.resize(size(), DataType{});
}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<DataType>& data)
    : shape_(shape), offset_(0), owns_data_(true) {
    if (shape_.empty()) {
        throw std::invalid_argument("Tensor shape cannot be empty");
    }
    compute_strides();
    if (data.size() != size()) {
        throw std::invalid_argument(
            "Data size (" + std::to_string(data.size()) +
            ") does not match tensor capacity (" + std::to_string(size()) + ")"
        );
    }
    data_ = data;
}

Tensor::Tensor(const Tensor& other)
    : data_(other.data_),
      grad_(other.grad_),
      shape_(other.shape_),
      strides_(other.strides_),
      offset_(other.offset_),
      owns_data_(true) {}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        data_ = other.data_;
        grad_ = other.grad_;
        shape_ = other.shape_;
        strides_ = other.strides_;
        offset_ = other.offset_;
        owns_data_ = true;
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(std::move(other.data_)),
      grad_(std::move(other.grad_)),
      shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      offset_(other.offset_),
      owns_data_(other.owns_data_) {
    other.offset_ = 0;
    other.owns_data_ = true;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        grad_ = std::move(other.grad_);
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        offset_ = other.offset_;
        owns_data_ = other.owns_data_;
        other.offset_ = 0;
        other.owns_data_ = true;
    }
    return *this;
}

size_t Tensor::size() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), size_t{1},
                           std::multiplies<size_t>());
}

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;
    strides_.back() = 1;
    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

size_t Tensor::flat_index(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::out_of_range(
            "Index dimension (" + std::to_string(indices.size()) +
            ") does not match tensor dimension (" + std::to_string(shape_.size()) + ")"
        );
    }
    size_t idx = offset_;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range(
                "Index " + std::to_string(indices[i]) + " is out of bounds "
                "for dimension " + std::to_string(i) +
                " (size " + std::to_string(shape_[i]) + ")"
            );
        }
        idx += indices[i] * strides_[i];
    }
    return idx;
}

Tensor::DataType& Tensor::at(const std::vector<size_t>& indices) {
    return data_[flat_index(indices)];
}

const Tensor::DataType& Tensor::at(const std::vector<size_t>& indices) const {
    return data_[flat_index(indices)];
}

Tensor Tensor::slice(size_t dim, size_t start, size_t end) const {
    if (dim >= shape_.size()) {
        throw std::out_of_range("Slice dimension out of range");
    }
    if (start > end || end > shape_[dim]) {
        throw std::out_of_range("Invalid slice bounds");
    }

    Tensor result;
    result.shape_ = shape_;
    result.shape_[dim] = end - start;
    result.strides_ = strides_;
    result.offset_ = offset_ + start * strides_[dim];
    result.data_.resize(data_.size());
    result.owns_data_ = false;

    result.data_ = data_;

    return result;
}

Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(),
                                      size_t{1}, std::multiplies<size_t>());
    if (new_size != size()) {
        throw std::invalid_argument(
            "New shape size (" + std::to_string(new_size) +
            ") does not match current size (" + std::to_string(size()) + ")"
        );
    }

    Tensor result;
    result.shape_ = new_shape;
    result.strides_.resize(new_shape.size());
    result.strides_.back() = 1;
    for (int i = static_cast<int>(new_shape.size()) - 2; i >= 0; --i) {
        result.strides_[i] = result.strides_[i + 1] * new_shape[i + 1];
    }
    result.offset_ = offset_;
    result.data_ = data_;
    result.owns_data_ = false;
    return result;
}

Tensor Tensor::clone() const {
    Tensor other;
    other.shape_ = shape_;
    other.strides_ = strides_;
    other.offset_ = 0;
    other.owns_data_ = true;

    other.data_.resize(size());
    if (offset_ == 0 && data_.size() == size()) {
        other.data_ = data_;
    } else {
        for (size_t i = 0; i < size(); ++i) {
            other.data_[i] = data_[offset_ + i];
        }
    }

    if (!grad_.empty()) {
        other.grad_ = grad_;
    }
    return other;
}

Tensor& Tensor::grad() {
    if (grad_.empty()) {
        grad_.resize(data_.size(), DataType{});
    }
    return *this;
}

const Tensor& Tensor::grad() const {
    return *this;
}

void Tensor::zero_grad() {
    if (!grad_.empty()) {
        std::fill(grad_.begin(), grad_.end(), DataType{});
    }
}

std::string Tensor::shape_string() const {
    if (shape_.empty()) return "[]";
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape_[i];
    }
    oss << "]";
    return oss.str();
}

void Tensor::print_data() const {
    if (shape_.empty()) {
        std::cout << "[]\n";
        return;
    }

    if (shape_.size() == 1) {
        std::cout << "[";
        for (size_t i = 0; i < shape_[0]; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << at({i});
        }
        std::cout << "]\n";
        return;
    }

    if (shape_.size() == 2) {
        for (size_t r = 0; r < shape_[0]; ++r) {
            std::cout << "[";
            for (size_t c = 0; c < shape_[1]; ++c) {
                if (c > 0) std::cout << ", ";
                std::cout << at({r, c});
            }
            std::cout << "]\n";
        }
        return;
    }

    std::cout << "Tensor(shape=" << shape_string() << ")\n";
}

} // namespace cosq::qnet
