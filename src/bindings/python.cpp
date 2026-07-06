#include <qnet/blas.hpp>
#include <qnet/graph.hpp>
#include <qnet/ops.hpp>
#include <qnet/optimizer.hpp>
#include <qnet/safetensors.hpp>
#include <qnet/tensor.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace qnet = cosq::qnet;

PYBIND11_MODULE(pyqnet, m) {
    m.doc() = "qnet - Lightweight C++ Neural Network Graph Framework";

    py::class_<qnet::Tensor>(m, "Tensor")
        .def(py::init<const std::vector<size_t>&>(),
             py::arg("shape"))
        .def(py::init<const std::vector<size_t>&,
                      const std::vector<float>&>(),
             py::arg("shape"), py::arg("data"))
        .def("ndim", &qnet::Tensor::ndim)
        .def("shape", &qnet::Tensor::shape)
        .def("size", &qnet::Tensor::size)
        .def("at", [](qnet::Tensor& t, const std::vector<size_t>& idx) {
            return t.at(idx);
        })
        .def("__repr__", [](const qnet::Tensor& t) {
            return "<Tensor shape=" + t.shape_string() + ">";
        })
        .def("clone", &qnet::Tensor::clone)
        .def("zero_grad", &qnet::Tensor::zero_grad)
        .def("to_list", [](const qnet::Tensor& t) {
            std::vector<float> result(t.data(), t.data() + t.size());
            return result;
        });

    py::class_<qnet::Node, std::shared_ptr<qnet::Node>>(m, "Node")
        .def_readonly("op_type", &qnet::Node::op_type)
        .def_property_readonly("output", [](const qnet::Node& n) {
            return n.output;
        })
        .def_property_readonly("gradient", [](const qnet::Node& n) {
            return n.gradient;
        });

    py::enum_<qnet::OpType>(m, "OpType")
        .value("INPUT", qnet::OpType::INPUT)
        .value("PARAMETER", qnet::OpType::PARAMETER)
        .value("MATMUL", qnet::OpType::MATMUL)
        .value("ADD", qnet::OpType::ADD)
        .value("MUL", qnet::OpType::MUL)
        .value("RELU", qnet::OpType::RELU)
        .value("SIGMOID", qnet::OpType::SIGMOID)
        .value("SOFTMAX", qnet::OpType::SOFTMAX)
        .value("CONV2D", qnet::OpType::CONV2D)
        .value("EMBEDDING", qnet::OpType::EMBEDDING)
        .value("CROSS_ENTROPY_LOSS", qnet::OpType::CROSS_ENTROPY_LOSS)
        .value("MSE_LOSS", qnet::OpType::MSE_LOSS)
        .value("BINARY_CROSS_ENTROPY_LOSS", qnet::OpType::BINARY_CROSS_ENTROPY_LOSS)
        .export_values();

    py::class_<qnet::Graph>(m, "Graph")
        .def(py::init<>())
        .def("variable", &qnet::Graph::variable)
        .def("parameter", &qnet::Graph::parameter)
        .def("matmul", &qnet::Graph::matmul)
        .def("add", &qnet::Graph::add)
        .def("relu", &qnet::Graph::relu)
        .def("sigmoid", &qnet::Graph::sigmoid)
        .def("softmax", &qnet::Graph::softmax)
        .def("cross_entropy_loss", &qnet::Graph::cross_entropy_loss)
        .def("mse_loss", &qnet::Graph::mse_loss)
        .def("binary_cross_entropy", &qnet::Graph::binary_cross_entropy)
        .def("forward", &qnet::Graph::forward)
        .def("backward", &qnet::Graph::backward)
        .def("zero_grad", &qnet::Graph::zero_grad)
        .def("parameters", &qnet::Graph::parameters);

    py::class_<qnet::Optimizer, std::shared_ptr<qnet::Optimizer>>(m, "Optimizer")
        .def("zero_grad", &qnet::Optimizer::zero_grad)
        .def("parameters", &qnet::Optimizer::parameters);

    py::class_<qnet::SGD, qnet::Optimizer, std::shared_ptr<qnet::SGD>>(m, "SGD")
        .def(py::init<std::vector<std::shared_ptr<qnet::Node>>, float, float, float, bool>(),
             py::arg("parameters"),
             py::arg("learning_rate"),
             py::arg("momentum") = 0.0f,
             py::arg("weight_decay") = 0.0f,
             py::arg("nesterov") = false)
        .def("step", &qnet::SGD::step);

    py::class_<qnet::Adam, qnet::Optimizer, std::shared_ptr<qnet::Adam>>(m, "Adam")
        .def(py::init<std::vector<std::shared_ptr<qnet::Node>>, float, float, float, float, float>(),
             py::arg("parameters"),
             py::arg("learning_rate") = 1e-3f,
             py::arg("beta1") = 0.9f,
             py::arg("beta2") = 0.999f,
             py::arg("epsilon") = 1e-8f,
             py::arg("weight_decay") = 0.0f)
        .def("step", &qnet::Adam::step);

    py::class_<qnet::AdamW, qnet::Adam, std::shared_ptr<qnet::AdamW>>(m, "AdamW")
        .def(py::init<std::vector<std::shared_ptr<qnet::Node>>, float, float, float, float, float>(),
             py::arg("parameters"),
             py::arg("learning_rate") = 1e-3f,
             py::arg("beta1") = 0.9f,
             py::arg("beta2") = 0.999f,
             py::arg("epsilon") = 1e-8f,
             py::arg("weight_decay") = 0.01f)
        .def("step", &qnet::AdamW::step);

    py::class_<qnet::SafeTensorsReader>(m, "SafeTensorsReader")
        .def(py::init<const std::string&>())
        .def("contains", &qnet::SafeTensorsReader::contains)
        .def("tensor_names", &qnet::SafeTensorsReader::tensor_names)
        .def("load_tensor", &qnet::SafeTensorsReader::load_tensor);
}
