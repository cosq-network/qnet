#include <qnet/safetensors.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cosq::qnet {

SafeTensorsReader::SafeTensorsReader(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    size_t file_size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    if (file_size < 8) {
        throw std::runtime_error("File too small to be safetensors");
    }

    data_.resize(file_size);
    file.read(data_.data(), static_cast<std::streamsize>(file_size));

    parse_header();
}

void SafeTensorsReader::parse_header() {
    uint64_t header_size;
    std::memcpy(&header_size, data_.data(), sizeof(header_size));

    if (header_size + 8 > data_.size()) {
        throw std::runtime_error("Header size exceeds file size");
    }

    std::string header(data_.data() + 8, data_.data() + 8 + header_size);

    std::string key, dtype_str;
    std::vector<size_t> shape;
    size_t offset_start, offset_end;

    size_t pos = 0;
    while (pos < header.size()) {
        while (pos < header.size() && header[pos] != '"') {
            ++pos;
        }
        if (pos >= header.size()) break;
        size_t key_start = ++pos;
        while (pos < header.size() && header[pos] != '"') {
            if (header[pos] == '\\') ++pos;
            ++pos;
        }
        if (pos >= header.size()) break;
        key = header.substr(key_start, pos - key_start);
        ++pos;

        if (key == "__metadata__" || key.empty()) continue;

        while (pos < header.size() && header[pos] != '{') ++pos;
        if (pos >= header.size()) break;
        ++pos;

        dtype_str.clear();
        shape.clear();
        offset_start = offset_end = 0;

        while (pos < header.size() && header[pos] != '}') {
            while (pos < header.size() && header[pos] != '"') ++pos;
            if (pos >= header.size()) break;
            size_t field_start = ++pos;
            while (pos < header.size() && header[pos] != '"') ++pos;
            std::string field = header.substr(field_start, pos - field_start);
            ++pos;
            while (pos < header.size() && header[pos] != ':') ++pos;
            ++pos;

            if (field == "dtype") {
                while (pos < header.size() && header[pos] != '"') ++pos;
                size_t v_start = ++pos;
                while (pos < header.size() && header[pos] != '"') ++pos;
                dtype_str = header.substr(v_start, pos - v_start);
                ++pos;
            } else if (field == "shape") {
                while (pos < header.size() && header[pos] != '[') ++pos;
                ++pos;
                shape.clear();
                while (pos < header.size() && header[pos] != ']') {
                    while (pos < header.size() && !(header[pos] >= '0' && header[pos] <= '9')) ++pos;
                    if (pos >= header.size() || header[pos] == ']') break;
                    char* end = nullptr;
                    shape.push_back(static_cast<size_t>(
                        std::strtoull(header.data() + pos, &end, 10)));
                    pos = static_cast<size_t>(end - header.data());
                }
                ++pos;
            } else if (field == "data_offsets") {
                while (pos < header.size() && header[pos] != '[') ++pos;
                ++pos;
                while (pos < header.size() && !(header[pos] >= '0' && header[pos] <= '9')) ++pos;
                char* end = nullptr;
                offset_start = static_cast<size_t>(
                    std::strtoull(header.data() + pos, &end, 10));
                pos = static_cast<size_t>(end - header.data());
                while (pos < header.size() && !(header[pos] >= '0' && header[pos] <= '9')) ++pos;
                offset_end = static_cast<size_t>(
                    std::strtoull(header.data() + pos, &end, 10));
                pos = static_cast<size_t>(end - header.data());
                ++pos;
            } else {
                while (pos < header.size() && header[pos] != ',' && header[pos] != '}') ++pos;
            }
            if (pos < header.size() && header[pos] == ',') ++pos;
        }

        size_t header_end_offset = 8 + header_size;
        offset_start += header_end_offset;
        offset_end += header_end_offset;

        entries_[key] = {dtype_str, shape, offset_start, offset_end};
    }
}

bool SafeTensorsReader::contains(const std::string& name) const {
    return entries_.find(name) != entries_.end();
}

std::vector<std::string> SafeTensorsReader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(entries_.size());
    for (const auto& [name, _] : entries_) {
        names.push_back(name);
    }
    return names;
}

Tensor SafeTensorsReader::load_tensor(const std::string& name) const {
    auto it = entries_.find(name);
    if (it == entries_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }

    const auto& entry = it->second;
    if (entry.dtype != "F32" && entry.dtype != "f32") {
        throw std::runtime_error("Unsupported dtype: " + entry.dtype +
                                 " (only F32/f32 supported)");
    }

    if (entry.offset_end > data_.size()) {
        throw std::runtime_error("Tensor data out of bounds: " + name);
    }

    size_t num_elements = 1;
    for (auto s : entry.shape) {
        num_elements *= s;
    }

    size_t expected_bytes = num_elements * sizeof(float);
    size_t actual_bytes = entry.offset_end - entry.offset_start;
    if (actual_bytes < expected_bytes) {
        throw std::runtime_error("Tensor data truncated: " + name);
    }

    Tensor tensor(entry.shape);
    std::memcpy(tensor.data(), data_.data() + entry.offset_start, expected_bytes);

    return tensor;
}

} // namespace cosq::qnet
