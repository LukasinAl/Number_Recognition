#include <vector>
#include "../../extern/pybind11/include/pybind11/functional.h"
#include "../../extern/pybind11/include/pybind11/operators.h"
#include "../../extern/pybind11/include/pybind11/pybind11.h"
#include "../../extern/pybind11/include/pybind11/stl.h"
#include "../Net/net.h"

namespace py = pybind11;

PYBIND11_MODULE(network_core, core) {
  core.doc() = "Neural network core";
  py::enum_<ActivationFunctions>(core, "ActivationFunction")
      .value("NO", ActivationFunctions::NO)
      .value("RELU", ActivationFunctions::RELU)
      .value("SOFTMAX", ActivationFunctions::SOFTMAX)
      .value("SIGMOID", ActivationFunctions::SIGMOID);

  py::enum_<LossFunctions>(core, "LossFunction")
      .value("MSE", LossFunctions::MSE)
      .value("CROSSENROPY", LossFunctions::CROSSENTROPY);

  py::class_<Net>(core, "Net")
      .def(py::init<size_t, const std::vector<int>&>())
      .def("FillByZeros", &Net::fill_by_zeros)
      .def("FillByRandomValues", &Net::FillBySmallRandomValues)
      .def("ForwardPass", py::overload_cast<const std::vector<double>&>(
                              &Net::ForwardPass, py::const_))
      .def("SetLayersAcivation", &Net::SetLayersActivations)
      .def("CalculateLoss", &Net::CalculateLoss)
      .def("SetLoss", &Net::SetLoss)
      .def("Train", &Net::Train)
      .def("BatchTrain", &Net::BatchTrain)
      .def("DumpToBinary", &Net::DumpToBinary)
      .def("ReadFromBinary", &Net::ReadFromBinary)
      .def_readwrite("learning_step", &Net::learning_step);

  py::class_<Matrix>(core, "Matrix")
      .def(py::init<size_t, size_t, double>(), py::arg("height"),
           py::arg("width"), py::arg("default_values"))
      .def(py::init<const std::vector<std::vector<double>>&>(),
           py::arg("values"))
      .def(py::init<const std::vector<double>&>(), py::arg("input"))
      .def(py::init<>())
      .def_readonly("n", &Matrix::n)
      .def_readonly("m", &Matrix::m)
      .def_readonly("matrix_values", &Matrix::matrix_values)
      .def("__mul__",
           py::overload_cast<const Matrix&>(&Matrix::operator*, py::const_),
           py::is_operator(), py::arg("other"))
      .def("__mul__", py::overload_cast<double>(&Matrix::operator*, py::const_),
           py::is_operator(), py::arg("other"))
      .def(
          "__rmul__", [](const Matrix& m, double d) { return m * d; },
          py::is_operator())
      .def("__add__",
           py::overload_cast<const Matrix&>(&Matrix::operator+, py::const_),
           py::is_operator(), py::arg("other"))
      .def("__sub__",
           py::overload_cast<const Matrix&>(&Matrix::operator-, py::const_),
           py::is_operator(), py::arg("other"))
      .def("__eq__",
           py::overload_cast<const Matrix&>(&Matrix::operator==, py::const_),
           py::is_operator(), py::arg("other"))
      .def("__imul__", py::overload_cast<const Matrix&>(&Matrix::operator*=),
           py::is_operator(), py::arg("other"))
      .def("__imul__", py::overload_cast<double>(&Matrix::operator*=),
           py::is_operator(), py::arg("other"))
      .def("__iadd__", py::overload_cast<const Matrix&>(&Matrix::operator+=),
           py::is_operator(), py::arg("other"))
      .def("__isub__", py::overload_cast<const Matrix&>(&Matrix::operator-=),
           py::is_operator(), py::arg("other"))
      .def("transpose", &Matrix::Transpose)
      .def("element_wise_multiplication", &Matrix::ElementWiseMultiplication,
           py::arg("other"))
      .def("load_from_txt", &Matrix::LoadFromTxt, py::arg("file_name"))
      .def_static("identity", &Matrix::identity, py::arg("size"))
      .def_static("zeros", &Matrix::zero, py::arg("height"), py::arg("width"))
      .def_static("ones", &Matrix::one, py::arg("height"), py::arg("width"));
}