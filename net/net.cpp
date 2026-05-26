#include "net.h"
#include <cmath>
#include <functional>
#include <initializer_list>
#include <random>
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
  activateOutput = false;
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
    if (i < layers_ - 2 || activateOutput) {
      ApplyActivation(temp);
    }
  }
  return temp.Transpose().matrix_values[0];
}

void Net::ApplyActivation(Matrix& vector) const {
  for (size_t i = 0; i < vector.matrix_values.size(); ++i) {
    vector.matrix_values[i][0] =
        activation_function_(vector.matrix_values[i][0]);
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

void Net::FillBySmallRandomValues() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-0.1, 0.1);
  weights_.resize(layers_ - 1);
  biases_.resize(layers_ - 1);
  for (size_t i = 0; i < layers_ - 1; ++i) {
    size_t input = layer_sizes_[i];
    size_t output = layer_sizes_[i + 1];
    std::vector<std::vector<double>> temp_w(output, std::vector<double>(input));
    std::vector<double> temp_b(output);
    for (size_t row = 0; row < output; ++row) {
      for (size_t column = 0; column < input; ++column) {
        temp_w[row][column] = dist(gen);
      }
      temp_b[row] = dist(gen);
    }
    weights_[i] = Matrix(temp_w);
    biases_[i] = Matrix(temp_b);
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
  if (result.size() != expected.size()) {
    throw std::runtime_error("Wrong dimentions");
  }
  return loss_function_(result, expected);
}

void Net::SetLossMSE() {
  loss_function_ = [](std::vector<double>& first, std::vector<double>& second) {
    double result = 0.0;
    for (size_t i = 0; i < first.size(); ++i) {
      result += std::pow(first[i] - second[i], 2);
    }
    result /= first.size();
    return result;
  };
}

void Net::SetActivateOutput(bool value) {
  activateOutput = value;
}

std::pair<std::vector<Matrix>, std::vector<Matrix>> Net::CalculateGradients(
    std::vector<double>& result, std::vector<double>& expected) const {
  //Loss function MSE support for now
  std::vector<Matrix> weights_gradients(layers_ - 1);
  std::vector<Matrix> biases_gradients(layers_ - 1);
  biases_gradients[layers_ - 2] = 2 * (Matrix(result) - Matrix(expected));
  for (size_t i = layers_ - 1; i > 0; --i) {
    
  }
}