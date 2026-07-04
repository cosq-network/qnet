#include <qnet/safetensors.hpp>
#include <qnet/serializer.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace cosq::qnet {

// -- Binary .qnet format constants --
static const uint32_t QNET_MAGIC = 0x54454E51; // "QNET" as LE uint32
static const uint32_t QNET_VERSION = 1;

static void write_u32(std::ofstream& f, uint32_t v) {
    uint32_t le = v;
    f.write(reinterpret_cast<const char*>(&le), sizeof(le));
}

static uint32_t read_u32(std::ifstream& f) {
    uint32_t v = 0;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

static void write_float(std::ofstream& f, float v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

static float read_float(std::ifstream& f) {
    float v = 0;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

static void write_tensor(std::ofstream& f, const Tensor& t) {
    auto shape = t.shape();
    write_u32(f, static_cast<uint32_t>(shape.size()));
    for (auto s : shape) write_u32(f, static_cast<uint32_t>(s));
    write_u32(f, static_cast<uint32_t>(t.size()));
    for (size_t i = 0; i < t.size(); ++i) write_float(f, t.data()[i]);
}

static Tensor read_tensor(std::ifstream& f) {
    uint32_t ndim = read_u32(f);
    std::vector<size_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; ++i) shape[i] = read_u32(f);
    uint32_t data_size = read_u32(f);
    std::vector<float> data(data_size);
    for (uint32_t i = 0; i < data_size; ++i) data[i] = read_float(f);
    return Tensor(shape, data);
}

// -- GraphSerializer --

void GraphSerializer::save(const Graph& graph, const std::string& filepath) {
    const auto& nodes = graph.nodes();
    std::ofstream f(filepath, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file for writing: " + filepath);

    write_u32(f, QNET_MAGIC);
    write_u32(f, QNET_VERSION);
    write_u32(f, static_cast<uint32_t>(nodes.size()));

    std::unordered_map<Node*, uint32_t> node_index;
    for (uint32_t i = 0; i < nodes.size(); ++i) {
        node_index[nodes[i].get()] = i;
    }

    for (auto& node : nodes) {
        write_u32(f, static_cast<uint32_t>(node->op_type));

        std::string name = node->name;
        write_u32(f, static_cast<uint32_t>(name.size()));
        f.write(name.data(), name.size());

        write_u32(f, static_cast<uint32_t>(node->inputs.size()));
        for (auto& input : node->inputs) {
            auto it = node_index.find(input.get());
            if (it == node_index.end()) {
                throw std::runtime_error("Node input not found in graph");
            }
            write_u32(f, it->second);
        }

        bool has_output = (node->output.size() > 0);
        write_u32(f, has_output ? 1 : 0);
        if (has_output) write_tensor(f, node->output);

        bool has_grad = (node->gradient.size() > 0);
        write_u32(f, has_grad ? 1 : 0);
        if (has_grad) write_tensor(f, node->gradient);
    }
}

Graph GraphSerializer::load(const std::string& filepath,
                             std::unordered_map<std::string,
                                                std::shared_ptr<Node>>* named_nodes) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file for reading: " + filepath);

    uint32_t magic = read_u32(f);
    if (magic != QNET_MAGIC) {
        throw std::runtime_error("Invalid .qnet file (bad magic)");
    }
    uint32_t version = read_u32(f);
    if (version > QNET_VERSION) {
        throw std::runtime_error("Unsupported .qnet version: " + std::to_string(version));
    }
    uint32_t node_count = read_u32(f);

    struct NodeData {
        OpType op_type;
        std::string name;
        std::vector<uint32_t> input_indices;
        Tensor output;
        Tensor gradient;
    };
    std::vector<NodeData> node_data(node_count);

    for (uint32_t i = 0; i < node_count; ++i) {
        node_data[i].op_type = static_cast<OpType>(read_u32(f));

        uint32_t name_len = read_u32(f);
        node_data[i].name.resize(name_len);
        f.read(node_data[i].name.data(), name_len);

        uint32_t input_count = read_u32(f);
        node_data[i].input_indices.resize(input_count);
        for (uint32_t j = 0; j < input_count; ++j) {
            node_data[i].input_indices[j] = read_u32(f);
        }

        if (read_u32(f)) node_data[i].output = read_tensor(f);
        if (read_u32(f)) node_data[i].gradient = read_tensor(f);
    }

    Graph graph;
    std::vector<std::shared_ptr<Node>> created_nodes(node_count);
    std::unordered_map<std::string, std::shared_ptr<Node>> name_map;

    for (uint32_t i = 0; i < node_count; ++i) {
        auto& nd = node_data[i];
        auto node = std::make_shared<Node>(nd.op_type);
        node->name = nd.name;
        created_nodes[i] = node;
        if (!nd.name.empty()) {
            name_map[nd.name] = node;
        }
    }

    for (uint32_t i = 0; i < node_count; ++i) {
        auto& nd = node_data[i];
        auto node = created_nodes[i];
        for (auto idx : nd.input_indices) {
            node->inputs.push_back(created_nodes[idx]);
        }
        node->output = std::move(nd.output);
        node->gradient = std::move(nd.gradient);
    }

    for (auto& n : created_nodes) {
        graph.add_node(n);
    }

    if (named_nodes) {
        *named_nodes = std::move(name_map);
    }

    return graph;
}

// -- SafeTensorsWriter --

SafeTensorsWriter::SafeTensorsWriter() {}

void SafeTensorsWriter::add_tensor(const std::string& name, const Tensor& tensor) {
    entries_.push_back({name, tensor});
}

std::string SafeTensorsWriter::build_header() const {
    std::ostringstream json;
    json << "{";
    bool first = true;
    uint64_t offset = 0;
    for (auto& entry : entries_) {
        if (!first) json << ",";
        first = false;

        uint64_t byte_size = entry.tensor.size() * sizeof(float);

        json << "\"" << entry.name << "\":{";
        json << "\"dtype\":\"F32\",";
        json << "\"shape\":[";
        for (size_t i = 0; i < entry.tensor.shape().size(); ++i) {
            if (i > 0) json << ",";
            json << entry.tensor.shape()[i];
        }
        json << "],\"data_offsets\":[";
        json << offset;
        json << ",";
        json << offset + byte_size;
        json << "]}";

        offset += byte_size;
    }
    json << "}";
    return json.str();
}

void SafeTensorsWriter::write(const std::string& filepath) const {
    std::string header = build_header();
    uint64_t header_size = header.size();

    std::vector<float> data;
    for (auto& entry : entries_) {
        size_t old_size = data.size();
        data.resize(old_size + entry.tensor.size());
        std::memcpy(data.data() + old_size, entry.tensor.data(),
                    entry.tensor.size() * sizeof(float));
    }

    std::ofstream f(filepath, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open safetensors file for writing: " + filepath);

    f.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    f.write(header.data(), header.size());
    f.write(reinterpret_cast<const char*>(data.data()),
            data.size() * sizeof(float));
}

// -- Weights import/export via safetensors --

void GraphSerializer::export_safetensors(const Graph& graph, const std::string& path) {
    SafeTensorsWriter writer;
    const auto& nodes = graph.nodes();
    for (size_t i = 0; i < nodes.size(); ++i) {
        auto& node = nodes[i];
        if (node->op_type == OpType::PARAMETER && node->output.size() > 0) {
            std::string name = node->name.empty()
                ? "param_" + std::to_string(i)
                : node->name;
            writer.add_tensor(name, node->output);
        }
    }
    writer.write(path);
}

void GraphSerializer::import_safetensors(Graph& graph, const std::string& path) {
    SafeTensorsReader reader(path);
    auto& nodes = graph.nodes();
    for (auto& node : nodes) {
        if (node->op_type == OpType::PARAMETER) {
            std::string name = node->name;
            if (name.empty()) continue;
            if (reader.contains(name)) {
                node->output = reader.load_tensor(name);
            }
        }
    }
}

} // namespace cosq::qnet
