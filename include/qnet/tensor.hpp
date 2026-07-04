#pragma once

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace cosq::qnet {

class Tensor {
public:
    using DataType = float;

    Tensor() = default;

    explicit Tensor(const std::vector<size_t>& shape);

    Tensor(const std::vector<size_t>& shape, const std::vector<DataType>& data);

    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor() = default;

    size_t ndim() const { return shape_.size(); }

    const std::vector<size_t>& shape() const { return shape_; }

    const std::vector<size_t>& strides() const { return strides_; }

    size_t offset() const { return offset_; }

    size_t size() const;

    DataType* data() { return data_.data(); }
    const DataType* data() const { return data_.data(); }

    DataType& at(const std::vector<size_t>& indices);
    const DataType& at(const std::vector<size_t>& indices) const;

    Tensor slice(size_t dim, size_t start, size_t end) const;

    Tensor view(const std::vector<size_t>& new_shape) const;

    Tensor clone() const;

    bool is_view() const { return !owns_data_; }

    Tensor& grad();
    const Tensor& grad() const;
    void zero_grad();
    bool has_grad() const { return !grad_.empty(); }

    size_t bytes() const { return data_.size() * sizeof(DataType); }

    std::string shape_string() const;
    void print_data() const;

private:
    std::vector<DataType> data_;
    std::vector<DataType> grad_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t offset_ = 0;
    bool owns_data_ = true;

    void compute_strides();
    size_t flat_index(const std::vector<size_t>& indices) const;
};

} // namespace cosq::qnet
