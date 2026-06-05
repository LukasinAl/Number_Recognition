#include <vector>
#include "../../extern/pybind11/include/pybind11/stl.h"
#include "../../extern/pybind11/include/pybind11/pybind11.h"
#include "../../extern/pybind11/include/pybind11/functional.h"
#include "../Net/net.h"

namespace py = pybind11;

PYBIND11_MODULE(network_core, core) {
  core.doc() = "Neural network core";
  py::enum_<ActivationFunnctions>(core, "ActivationFunction")
    .value("NO", ActivationFunnctions::NO)
    .value("RELU", ActivationFunnctions::RELU)
    .value("SIGMOID", ActivationFunnctions::SIGMOID);

  py::class_<Net>(core, "Net")
    .def(py::init<size_t, const std::vector<int>&>())
    .def("FillByZeros", &Net::fill_by_zeros)
    .def("FillByRandomValues", &Net::FillBySmallRandomValues)
    .def("ForwardPass", py::overload_cast<const std::vector<double>&>(&Net::ForwardPass, py::const_))
    .def("TrainingForwardPass", &Net::TrainingForwardPass)
    .def("SetLayersAcivation", &Net::SetLayersActivations)
    .def("CalculateLoss", &Net::CalculateLoss)
    .def("SetLossMSE", &Net::SetLossMSE)
    .def("Train", &Net::Train)
    .def("BatchTrain", Net::BatchTrain)
    .def("DumpToBinary", &Net::DumpToBinary)
    .def("ReadFromBinary", &Net::ReadFromBinary)
    .def_readwrite("learning_step", &Net::learning_step);
}