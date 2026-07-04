#include <qnet/graph.hpp>
#include <qnet/ops.hpp>
#include <qnet/serializer.hpp>

#include <cassert>
#include <iostream>
#include <string>

using namespace cosq::qnet;

static bool approx(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

static void test_save_load() {
    Graph graph;
    Tensor X({1, 4}, {1, 2, 3, 4});
    Tensor W({4, 2}, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f});

    auto x = graph.variable(X);
    auto w = graph.parameter(W);
    auto out = graph.matmul(x, w);

    graph.forward(out);

    float expected_0 = out->output.data()[0];
    float expected_1 = out->output.data()[1];

    std::string path = "/tmp/test_qnet_graph.qnet";
    GraphSerializer::save(graph, path);

    std::unordered_map<std::string, std::shared_ptr<Node>> named_nodes;
    Graph loaded = GraphSerializer::load(path, &named_nodes);

    assert(loaded.nodes().size() == graph.nodes().size());

    std::string safetensors_path = "/tmp/test_qnet_weights.safetensors";
    GraphSerializer::export_safetensors(graph, safetensors_path);

    std::cout << "  [PASS] test_save_load\n";
    std::remove(path.c_str());
    std::remove(safetensors_path.c_str());
}

static void test_safetensors_roundtrip() {
    Graph graph;
    Tensor W1({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor W2({3, 1}, {0.5f, 1.0f, 1.5f});

    auto w1 = graph.parameter(W1);
    auto w2 = graph.parameter(W2);

    std::string path = "/tmp/test_qnet_st.safetensors";
    GraphSerializer::export_safetensors(graph, path);

    Graph graph2;
    Tensor X({1, 2}, {7, 8});
    Tensor W1_new({2, 3}, {0, 0, 0, 0, 0, 0});
    auto x = graph2.variable(X);
    auto w1_loaded = graph2.parameter(W1_new);
    w1_loaded->name = "W1";

    auto out = graph2.matmul(x, w1_loaded);
    graph2.forward(out);

    std::cout << "  [PASS] test_safetensors_roundtrip\n";
    std::remove(path.c_str());
}

int main() {
    std::cout << "Serializer tests:\n";
    test_save_load();
    test_safetensors_roundtrip();
    std::cout << "All serializer tests passed!\n";
    return 0;
}
