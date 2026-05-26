#include "net.h"
#include <cmath>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <vector>
#include "../Matrix/Matrix.h"

Net::Net(size_t lay, const std::vector<int>& sizes)
    : layers_(lay), layer_sizes_(sizes) {
  if (layers_ < 2) {
    throw std::runtime_error("Too litte layers");
  }
  if (layers_ != layer_sizes_.size()) {
    throw std::runtime_error("Too little layer sizes");
  }
  fill_by_zeros();
  SetActivationRelU();
}

Net::Net(size_t lay, std::initializer_list<int> sizes)
    : Net(lay, std::vector<int>(sizes)) {}

std::vector<double> Net::ForwardPass(const std::vector<double>& input) const {
  Matrix temp(input);
  if (input.size() != layer_sizes_[0]) {
    throw std::runtime_error("Wrong dimentions");
  }
  for (size_t i = 0; i < layers_ - 1; ++i) {
    temp = weights_[i] * temp + biases_[i];
    ApplyActivation(temp);
  }
  temp.Transpose();
  return temp.Get_values()[0];
}

void Net::ApplyActivation(Matrix& vector) const {
  for (size_t i = 0; i < vector.matrix_values.size(); ++i) {
    vector.matrix_values[i][0] = activation_function_(vector.matrix_values[i][0]);
  }
}

std::vector<double> Net::ForwardPass(
    std::initializer_list<double> input) const {
  return ForwardPass(std::vector<double>(input));
}

void Net::fill_by_zeros() {
  weights_.resize(layers_ - 1);
  biases_.resize(layers_ - 1);
  for (size_t i = 0; i < layers_ - 1; ++i) {
    weights_[i] = Matrix(std::vector<std::vector<double>>(
        layer_sizes_[i + 1], std::vector<double>(layer_sizes_[i], 0)));
    biases_[i] = Matrix(std::vector<std::vector<double>>(
        layer_sizes_[i + 1], std::vector<double>(1, 0)));
  }
}

void Net::SetActivationRelU() {
  activation_function_ = [](double x) {
    if (x > 0) {
      return x;
    }
    return 0.0;
  };
}

void Net::SetActivationSigmoid() {
  activation_function_ = [](double x) {
    return 1.0 / (1.0 + std::exp(-x));
  };
}

double Net::CalculateLoss(std::vector<double>& result,
                           std::vector<double>& expected) const {
  return loss_function_(result, expected);
}