#pragma once

#include <qnet/export.hpp>
#include <qnet/tensor.hpp>

#include <string>
#include <unordered_map>
#include <vector>

namespace cosq::qnet {

struct QNET_API SafeTensorEntry {
    std::string dtype;
    std::vector<size_t> shape;
    size_t offset_start;
    size_t offset_end;
};

class QNET_API SafeTensorsReader {
public:
    explicit SafeTensorsReader(const std::string& filepath);

    const std::unordered_map<std::string, SafeTensorEntry>& entries() const {
        return entries_;
    }

    bool contains(const std::string& name) const;

    Tensor load_tensor(const std::string& name) const;

    std::vector<std::string> tensor_names() const;

private:
    std::vector<char> data_;
    std::unordered_map<std::string, SafeTensorEntry> entries_;

    void parse_header();
};

} // namespace cosq::qnet
