#include <qnet/safetensors.hpp>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

using namespace cosq::qnet;

static std::string temp_path(const std::string& name) {
    auto tmp = fs::temp_directory_path();
    return (tmp / name).string();
}

static bool approx(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

static void create_test_file(const std::string& path) {
    std::string header = R"({"test_tensor":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}})";
    uint64_t header_size = header.size();

    std::vector<float> tensor_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    file.write(header.data(), header.size());
    file.write(reinterpret_cast<const char*>(tensor_data.data()),
               tensor_data.size() * sizeof(float));
    file.close();
}

static void test_reader() {
    std::string path = temp_path("test_qnet.safetensors");
    create_test_file(path);

    SafeTensorsReader reader(path);

    auto names = reader.tensor_names();
    assert(names.size() == 1);
    assert(names[0] == "test_tensor");

    assert(reader.contains("test_tensor"));
    assert(!reader.contains("nonexistent"));

    Tensor t = reader.load_tensor("test_tensor");
    assert(t.shape()[0] == 2);
    assert(t.shape()[1] == 3);
    assert(approx(t.at({0, 0}), 1.0f));
    assert(approx(t.at({1, 2}), 6.0f));

    std::cout << "  [PASS] test_reader\n";
    std::remove(path.c_str());
}

static void test_multi_tensor() {
    std::string path = temp_path("test_qnet_multi.safetensors");
    {
        std::string header = R"({"weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"bias":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}})";
        uint64_t header_size = header.size();
        std::vector<float> data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};

        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
        file.write(header.data(), header.size());
        file.write(reinterpret_cast<const char*>(data.data()),
                   data.size() * sizeof(float));
    }

    SafeTensorsReader reader(path);

    assert(reader.contains("weight"));
    assert(reader.contains("bias"));

    Tensor w = reader.load_tensor("weight");
    assert(w.shape()[0] == 2 && w.shape()[1] == 2);
    assert(approx(w.at({0, 0}), 0.1f));
    assert(approx(w.at({1, 1}), 0.4f));

    Tensor b = reader.load_tensor("bias");
    assert(b.shape()[0] == 2);
    assert(approx(b.at({0}), 0.5f));
    assert(approx(b.at({1}), 0.6f));

    std::cout << "  [PASS] test_multi_tensor\n";
    std::remove(path.c_str());
}

int main() {
    std::cout << "Safetensors tests:\n";
    test_reader();
    test_multi_tensor();
    std::cout << "All safetensors tests passed!\n";
    return 0;
}
