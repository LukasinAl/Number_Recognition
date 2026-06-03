#include "net.h"
#include <cmath>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
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
  learning_step = 0.01;
  fill_by_zeros();
  activation_functions_.resize(layers_ - 1, ActivationFunnctions::RELU);
  activation_functions_.back() = ActivationFunnctions::NO;
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
    ApplyActivation(temp, i);
  }
  return temp.Transpose().matrix_values[0];
}

std::vector<double> Net::ForwardPass(
    std::initializer_list<double> input) const {
  return ForwardPass(std::vector<double>(input));
}

std::vector<double> Net::TrainingForwardPass(const std::vector<double>& input) {
  Matrix temp(input);
  if (input.size() != layer_sizes_[0]) {
    throw std::runtime_error("Wrong dimentions");
  }
  last_pass_activation.clear();
  last_pass_pre_acctivation.clear();
  last_pass_activation.push_back(Matrix(input));
  for (size_t i = 0; i < layers_ - 1; ++i) {
    temp = weights_[i] * temp + biases_[i];
    last_pass_pre_acctivation.push_back(temp);
    ApplyActivation(temp, i);
    last_pass_activation.push_back(temp);
  }
  return temp.Transpose().matrix_values[0];
}

void Net::ApplyActivation(Matrix& vector, int layer) const {
  for (size_t i = 0; i < vector.matrix_values.size(); ++i) {
    vector.matrix_values[i][0] = ResolveActivation(activation_functions_[layer],
                                                   vector.matrix_values[i][0]);
  }
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
    return result;
  };
}

std::pair<std::vector<Matrix>, std::vector<Matrix>> Net::CalculateGradients(
    const std::vector<double>& result,
    const std::vector<double>& expected) const {
  //Loss function MSE support for now
  std::vector<Matrix> weights_gradients(layers_ - 1);
  std::vector<Matrix> biases_gradients(layers_ - 1);
  biases_gradients[layers_ - 2] = 2 * (Matrix(result) - Matrix(expected));
  for (int i = layers_ - 2; i >= 0; --i) {
    if (i > 0) {
      Matrix derivative_pre_activation = last_pass_pre_acctivation[i - 1];
      for (auto& item : derivative_pre_activation.matrix_values) {
        item[0] = ResolveDerivative(activation_functions_[i - 1], item[0]);
      }
      biases_gradients[i - 1] =
          (weights_[i].Transpose() * biases_gradients[i])
              .ElementWiseMultiplication(derivative_pre_activation);
    }
    weights_gradients[i] =
        biases_gradients[i] * last_pass_activation[i].Transpose();
  }
  return {weights_gradients, biases_gradients};
}

void Net::Step(std::pair<std::vector<Matrix>, std::vector<Matrix>> gradients) {
  for (size_t layer = 0; layer < layers_ - 1; ++layer) {
    size_t input = layer_sizes_[layer];
    size_t output = layer_sizes_[layer + 1];
    for (size_t row = 0; row < output; ++row) {
      for (size_t column = 0; column < input; ++column) {
        weights_[layer].matrix_values[row][column] -=
            gradients.first[layer].matrix_values[row][column] * learning_step;
      }
      biases_[layer].matrix_values[row][0] -=
          gradients.second[layer].matrix_values[row][0] * learning_step;
    }
  }
}

void Net::DumpToBinary(const std::string& file_name) const {
  std::ofstream out(file_name, std::ios::binary);
  out.write(reinterpret_cast<const char*>(&layers_), sizeof(layers_));
  out.write(reinterpret_cast<const char*>(&learning_step),
            sizeof(learning_step));
  out.write(reinterpret_cast<const char*>(layer_sizes_.data()),
            layer_sizes_.size() * sizeof(int));
  for (size_t i = 0; i < biases_.size(); ++i) {
    out.write(reinterpret_cast<const char*>(
                  biases_[i].Transpose().matrix_values[0].data()),
              biases_[i].Transpose().matrix_values[0].size() * sizeof(double));
  }
  for (size_t i = 0; i < weights_.size(); ++i) {
    for (size_t row = 0; row < weights_[i].matrix_values.size(); ++row) {
      out.write(
          reinterpret_cast<const char*>(weights_[i].matrix_values[row].data()),
          weights_[i].matrix_values[row].size() * sizeof(double));
    }
  }
}

void Net::ReadFromBinary(const std::string& file_name) {
  std::ifstream in(file_name, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file" + file_name);
  }
  in.read(reinterpret_cast<char*>(&layers_), sizeof(layers_));
  layer_sizes_.resize(layers_);
  in.read(reinterpret_cast<char*>(&learning_step), sizeof(learning_step));
  in.read(reinterpret_cast<char*>(layer_sizes_.data()),
          layer_sizes_.size() * sizeof(int));
  biases_.resize(layers_ - 1);
  for (size_t i = 0; i < biases_.size(); ++i) {
    biases_[i] = Matrix(1, layer_sizes_[i + 1], 0);
    in.read(reinterpret_cast<char*>(biases_[i].matrix_values[0].data()),
            sizeof(double) * layer_sizes_[i + 1]);
    biases_[i] = biases_[i].Transpose();
  }
  weights_.resize(layers_ - 1);
  for (size_t i = 0; i < weights_.size(); ++i) {
    weights_[i] = Matrix(layer_sizes_[i + 1], layer_sizes_[i], 0);
    for (size_t row = 0; row < layer_sizes_[i + 1]; ++row) {
      in.read(reinterpret_cast<char*>(weights_[i].matrix_values[row].data()),
              layer_sizes_[i] * sizeof(double));
    }
  }
}

void Net::Train(const std::vector<std::vector<double>>& inputs,
                const std::vector<std::vector<double>>& targets, int epochs,
                int batch_size) {
  if (inputs.size() != targets.size()) {
    throw std::runtime_error("Number of inputs and targets is different");
  }
  for (int i = 0; i < epochs; ++i) {
    int total_count = 0;
    while (total_count < inputs.size()) {
      std::vector<double> result;
      std::vector<Matrix> weight_grad;
      std::vector<Matrix> biases_grad;
      weight_grad.resize(layers_ - 1);
      biases_grad.resize(layers_ - 1);
      for (size_t item = 0; item < batch_size; ++item) {
        if (total_count == inputs.size()) {
          break;
        }
        result = TrainingForwardPass(inputs[total_count]);
        auto grads = CalculateGradients(result, targets[total_count]);
        for (size_t lay = 0; lay < layers_ - 1; ++lay) {
          if (weight_grad[lay].m == 0) {
            weight_grad[lay] = grads.first[lay];
            biases_grad[lay] = grads.second[lay];
          } else {
            weight_grad[lay] += grads.first[lay];
            biases_grad[lay] += grads.second[lay];
          }
        }
        total_count++;
      }
      for (size_t lay = 0; lay < layers_ - 1; ++lay) {
        int size;
        if (total_count != inputs.size() || total_count % batch_size == 0) {
          size = batch_size;
        } else {
          size = total_count % batch_size;
        }
        weight_grad[lay] *= (1.0 / size);
        biases_grad[lay] *= (1.0 / size);
      }
      Step({weight_grad, biases_grad});
    }
  }
}

void Net::SetLayersActivations(
    const std::vector<ActivationFunnctions>& functions) {
  activation_functions_ = functions;
}

double Net::ResolveActivation(ActivationFunnctions func, double x) const {
  if (func == ActivationFunnctions::RELU) {
    return ReLu(x);
  }
  if (func == ActivationFunnctions::SIGMOID) {
    return Sigmoid(x);
  }
  if (func == ActivationFunnctions::NO) {
    return x;
  }
  return x;
}

double Net::ResolveDerivative(ActivationFunnctions func, double x) const {
  if (func == ActivationFunnctions::NO) {
    return 1;
  }
  if (func == ActivationFunnctions::RELU) {
    if (x > 0) {
      return 1;
    }
    return 0;
  }
  if (func == ActivationFunnctions::SIGMOID) {
    return Sigmoid(x) * (1 - Sigmoid(x));
  }
  return 1;
}

double Net::ReLu(double x) {
  if (x > 0) {
    return x;
  } else {
    return 0;
  }
}

double Net::Sigmoid(double x) {
  return 1.0 / (1 + std::exp(-x));
}