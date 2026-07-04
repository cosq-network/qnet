#include <qnet/blas.hpp>
#include <qnet/graph.hpp>
#include <qnet/ops.hpp>
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
        .value("RELU", qnet::OpType::RELU)
        .value("SIGMOID", qnet::OpType::SIGMOID)
        .value("SOFTMAX", qnet::OpType::SOFTMAX)
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
        .def("forward", &qnet::Graph::forward)
        .def("backward", &qnet::Graph::backward)
        .def("zero_grad", &qnet::Graph::zero_grad);

    py::class_<qnet::SafeTensorsReader>(m, "SafeTensorsReader")
        .def(py::init<const std::string&>())
        .def("contains", &qnet::SafeTensorsReader::contains)
        .def("tensor_names", &qnet::SafeTensorsReader::tensor_names)
        .def("load_tensor", &qnet::SafeTensorsReader::load_tensor);
}
