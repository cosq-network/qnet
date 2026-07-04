#pragma once

#include <qnet/graph.hpp>
#include <qnet/tensor.hpp>

#include <string>
#include <unordered_map>
#include <vector>

namespace cosq::qnet {

class SafeTensorsWriter {
public:
    SafeTensorsWriter();
    void add_tensor(const std::string& name, const Tensor& tensor);
    void write(const std::string& filepath) const;

private:
    struct Entry {
        std::string name;
        Tensor tensor;
    };
    std::vector<Entry> entries_;
    std::string build_header() const;
};

struct GraphSerializer {
    static void save(const Graph& graph, const std::string& filepath);

    static Graph load(const std::string& filepath,
                      std::unordered_map<std::string,
                                         std::shared_ptr<Node>>* named_nodes = nullptr);

    static void export_safetensors(const Graph& graph, const std::string& path);

    static void import_safetensors(Graph& graph, const std::string& path);
};

} // namespace cosq::qnet
