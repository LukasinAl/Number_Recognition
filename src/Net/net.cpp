#include "net.h"

#include <cmath>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../Matrix/Matrix.h"

Net::Net(size_t lay, const std::vector<int>& sizes)
    : layers_(lay), layer_sizes_(sizes) {
  if (layers_ < 2) {
    throw std::invalid_argument("Too litte layers");
  }
  if (layers_ != layer_sizes_.size()) {
    throw std::length_error("Too little layer sizes");
  }
  learning_step = 0.01;
  fill_by_zeros();
  activation_functions_.resize(layers_ - 1, ActivationFunctions::NO);
  activation_functions_.back() = ActivationFunctions::NO;
  lossfunc = LossFunctions::MSE;
}

Net::Net(size_t lay, std::initializer_list<int> sizes)
    : Net(lay, std::vector<int>(sizes)) {}

std::vector<double> Net::ForwardPass(const std::vector<double>& input) const {
  Matrix temp(input);
  if (input.size() != layer_sizes_[0]) {
    throw std::length_error("Wrong dimentions");
  }
  for (size_t i = 0; i < layers_ - 1; ++i) {
    temp = weights_[i] * temp + biases_[i];
    ApplyActivation(temp, i);
  }
  return temp.Transpose().matrix_values[0];
}

Matrix Net::BatchForwardPass(const Matrix& input) const {
  Matrix temp = input.Transpose();
  if (input.matrix_values.size() < 1) {
    throw std::invalid_argument("No data given");
  }
  if (input.matrix_values[0].size() != layer_sizes_[0]) {
    throw std::length_error("Wrong dimentions");
  }
  for (size_t i = 0; i < layers_ - 1; ++i) {
    std::vector<std::vector<double>> mult(
        1, std::vector<double>(temp.matrix_values.size(), 1));
    temp = weights_[i] * temp + biases_[i] * Matrix(mult);
    ApplyActivation(temp, i);
  }
  return temp.Transpose();
}

std::vector<double> Net::ForwardPass(
    std::initializer_list<double> input) const {
  return ForwardPass(std::vector<double>(input));
}

std::vector<double> Net::TrainingForwardPass(
    const std::vector<double>& input, std::vector<Matrix>& last_pass_activation,
    std::vector<Matrix>& last_pass_pre_acctivation) {
  Matrix temp(input);
  if (input.size() != layer_sizes_[0]) {
    throw std::length_error("Wrong dimentions");
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

Matrix Net::BatchTrainingForwardPass(
    const Matrix& input, std::vector<Matrix>& last_pass_activation,
    std::vector<Matrix>& last_pass_pre_acctivation) {
  Matrix temp = input.Transpose();
  if (input.matrix_values.size() < 1) {
    throw std::invalid_argument("No data given");
  }
  if (input.matrix_values[0].size() != layer_sizes_[0]) {
    throw std::length_error("Wrong dimentions");
  }
  last_pass_activation.clear();
  last_pass_pre_acctivation.clear();
  last_pass_activation.push_back(temp);
  for (size_t i = 0; i < layers_ - 1; ++i) {
    std::vector<std::vector<double>> mult(
        1, std::vector<double>(input.matrix_values.size(), 1));
    temp = weights_[i] * temp + biases_[i] * Matrix(mult);
    last_pass_pre_acctivation.push_back(temp);
    ApplyActivation(temp, i);
    last_pass_activation.push_back(temp);
  }
  return temp.Transpose();
}

void Net::ApplyActivation(Matrix& vector, int layer) const {
  if (activation_functions_[layer] == ActivationFunctions::SOFTMAX) {
    for (size_t vec = 0; vec < vector.matrix_values[0].size(); ++vec) {
      double statsum = 0;
      double min = vector.matrix_values[0][vec];
      for (size_t i = 0; i < vector.matrix_values.size(); ++i) {
        min = std::min(min, vector.matrix_values[i][vec]);
      }
      for (size_t i = 0; i < vector.matrix_values.size(); ++i) {
        vector.matrix_values[i][vec] =
            std::exp(vector.matrix_values[i][vec] - min);
        statsum += vector.matrix_values[i][vec];
      }
      for (size_t i = 0; i < vector.matrix_values.size(); ++i) {
        vector.matrix_values[i][vec] /= statsum;
      }
    }
    return;
  }
  for (size_t vec = 0; vec < vector.matrix_values[0].size(); ++vec) {
    for (size_t i = 0; i < vector.matrix_values.size(); ++i) {
      vector.matrix_values[i][vec] = ResolveActivation(
          activation_functions_[layer], vector.matrix_values[i][vec]);
    }
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
  if (result.size() != expected.size() || result.size() == 0) {
    throw std::runtime_error("Wrong dimentions");
  }
  if (lossfunc == LossFunctions::MSE) {
    double ans = 0;
    for (size_t i = 0; i < result.size(); ++i) {
      ans += std::pow(expected[i] - result[i], 2);
    }
    return ans;
  }
  if (lossfunc == LossFunctions::CROSSENTROPY) {
    double ans = 0;
    for (size_t i = 0; i < result.size(); ++i) {
      ans -= expected[i] * std::log(result[i] + 1e-12);
    }
    return ans;
  }
  return -1;
}

void Net::SetLoss(const LossFunctions func) { lossfunc = func; }

Gradients Net::CalculateGradients(
    const std::vector<double>& result, const std::vector<double>& expected,
    std::vector<Matrix>& last_pass_activation,
    std::vector<Matrix>& last_pass_pre_acctivation) const {
  std::vector<Matrix> weights_gradients(layers_ - 1);
  std::vector<Matrix> biases_gradients(layers_ - 1);
  Matrix input_grad;

  if (lossfunc == LossFunctions::MSE &&
      activation_functions_.back() == ActivationFunctions::NO) {
    biases_gradients[layers_ - 2] = 2 * (Matrix(result) - Matrix(expected));
  } else if (lossfunc == LossFunctions::CROSSENTROPY &&
             activation_functions_.back() == ActivationFunctions::SOFTMAX) {
    biases_gradients[layers_ - 2] = Matrix(result) - Matrix(expected);
  } else {
    throw std::runtime_error("This combination is not supported yet");
  }

  for (int i = layers_ - 2; i >= 0; --i) {
    if (i > 0) {
      Matrix derivative_pre_activation = last_pass_pre_acctivation[i - 1];
      ApplyDerivative(derivative_pre_activation, i - 1);
      biases_gradients[i - 1] =
          (weights_[i].Transpose() * biases_gradients[i])
              .ElementWiseMultiplication(derivative_pre_activation);
    } else {
      input_grad = weights_[0].Transpose() * biases_gradients[0];
    }
    weights_gradients[i] =
        biases_gradients[i] * last_pass_activation[i].Transpose();
  }
  return Gradients({weights_gradients, biases_gradients, input_grad});
}

Gradients Net::BatchCalculateGradients(
    const Matrix& result, const Matrix& expected,
    std::vector<Matrix>& last_pass_activation,
    std::vector<Matrix>& last_pass_pre_acctivation) const {
  std::vector<Matrix> weights_gradients(layers_ - 1);
  std::vector<Matrix> biases_gradients(layers_ - 1);
  Matrix input_grad;

  if (lossfunc == LossFunctions::MSE &&
      activation_functions_.back() == ActivationFunctions::NO) {
    biases_gradients[layers_ - 2] =
        2 * (result.Transpose() - expected.Transpose());
  } else if (lossfunc == LossFunctions::CROSSENTROPY &&
             activation_functions_.back() == ActivationFunctions::SOFTMAX) {
    biases_gradients[layers_ - 2] = result.Transpose() - expected.Transpose();
  } else {
    throw std::runtime_error("This combination is not supported yet");
  }

  for (int i = layers_ - 2; i >= 0; --i) {
    if (i > 0) {
      Matrix derivative_pre_activation = last_pass_pre_acctivation[i - 1];
      ApplyDerivative(derivative_pre_activation, i - 1);
      biases_gradients[i - 1] =
          (weights_[i].Transpose() * biases_gradients[i])
              .ElementWiseMultiplication(derivative_pre_activation);
    } else {
      input_grad = weights_[0].Transpose() * biases_gradients[0] *
                   (1.0 / result.matrix_values.size());
    }
    weights_gradients[i] = biases_gradients[i] *
                           last_pass_activation[i].Transpose() *
                           (1.0 / result.matrix_values.size());
    biases_gradients[i] =
        biases_gradients[i] *
        Matrix(std::vector<std::vector<double>>(result.matrix_values.size(),
                                                std::vector<double>(1, 1))) *
        (1.0 / result.matrix_values.size());
  }
  return Gradients({weights_gradients, biases_gradients, input_grad});
}

void Net::ApplyDerivative(Matrix& vector, int layer) const {
  for (auto& item : vector.matrix_values) {
    for (size_t i = 0; i < item.size(); ++i) {
      item[i] = ResolveDerivative(activation_functions_[layer], item[i]);
    }
  }
}

void Net::Step(Gradients gradients) {
  for (size_t layer = 0; layer < layers_ - 1; ++layer) {
    size_t input = layer_sizes_[layer];
    size_t output = layer_sizes_[layer + 1];
    for (size_t row = 0; row < output; ++row) {
      for (size_t column = 0; column < input; ++column) {
        weights_[layer].matrix_values[row][column] -=
            gradients.weight_grads[layer].matrix_values[row][column] *
            learning_step;
      }
      biases_[layer].matrix_values[row][0] -=
          gradients.bias_grads[layer].matrix_values[row][0] * learning_step;
    }
  }
}

void Net::DumpToBinary(const std::string& file_name) const {
  std::ofstream out(file_name, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("Cannot open file" + file_name);
  }
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
  int val = 0;
  if (lossfunc == LossFunctions::MSE) {
    val = 1;
  } else if (lossfunc == LossFunctions::CROSSENTROPY) {
    val = 2;
  }
  if (val == 0) {
    throw std::runtime_error("No Loss was set");
  }
  out.write(reinterpret_cast<const char*>(&val), sizeof(val));
  for (size_t i = 0; i < activation_functions_.size(); ++i) {
    int v = 0;
    if (activation_functions_[i] == ActivationFunctions::NO) {
      v = 1;
    } else if (activation_functions_[i] == ActivationFunctions::RELU) {
      v = 2;
    } else if (activation_functions_[i] == ActivationFunctions::SIGMOID) {
      v = 3;
    } else if (activation_functions_[i] == ActivationFunctions::SOFTMAX) {
      v = 4;
    }
    if (v == 0) {
      throw std::runtime_error("No activation was set for layer " +
                               std::to_string(i));
    }
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
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
  int val = 0;
  in.read(reinterpret_cast<char*>(&val), sizeof(val));
  if (val == 1) {
    lossfunc = LossFunctions::MSE;
  } else if (val == 2) {
    lossfunc = LossFunctions::CROSSENTROPY;
  }
  activation_functions_.resize(layers_ - 1);
  for (size_t i = 0; i < layers_ - 1; ++i) {
    int v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (v == 1) {
      activation_functions_[i] = ActivationFunctions::NO;
    } else if (v == 2) {
      activation_functions_[i] = ActivationFunctions::RELU;
    } else if (v == 3) {
      activation_functions_[i] = ActivationFunctions::SIGMOID;
    } else if (v == 4) {
      activation_functions_[i] = ActivationFunctions::SOFTMAX;
    } else {
      throw std::runtime_error("Unexpected activation functio");
    }
  }
}

void Net::Train(const std::vector<std::vector<double>>& inputs,
                const std::vector<std::vector<double>>& targets, int epochs,
                int batch_size) {
  if (inputs.size() != targets.size()) {
    throw std::length_error("Number of inputs and targets is different");
  }
  for (int i = 0; i < epochs; ++i) {
    int total_count = 0;
    while (total_count < inputs.size()) {
      std::vector<double> result;
      std::vector<Matrix> weight_grad;
      std::vector<Matrix> biases_grad;
      std::vector<Matrix> pre_active;
      std::vector<Matrix> post_active;
      weight_grad.resize(layers_ - 1);
      biases_grad.resize(layers_ - 1);
      for (size_t item = 0; item < batch_size; ++item) {
        if (total_count == inputs.size()) {
          break;
        }
        result =
            TrainingForwardPass(inputs[total_count], post_active, pre_active);
        auto grads = CalculateGradients(result, targets[total_count],
                                        post_active, pre_active);
        for (size_t lay = 0; lay < layers_ - 1; ++lay) {
          if (weight_grad[lay].m == 0) {
            weight_grad[lay] = grads.weight_grads[lay];
            biases_grad[lay] = grads.bias_grads[lay];
          } else {
            weight_grad[lay] += grads.weight_grads[lay];
            biases_grad[lay] += grads.bias_grads[lay];
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
      Step({weight_grad, biases_grad, Matrix(1, 1, 0)}); //input grads is dummy matrix which is not used in Step prosess
    }
  }
}

void Net::BatchTrain(const std::vector<std::vector<double>>& inputs,
                     const std::vector<std::vector<double>>& targets) {
  if (inputs.size() != targets.size() || inputs.size() == 0) {
    throw std::length_error("Number of inputs and targets is different");
  }
  Matrix result;
  std::vector<Matrix> weight_grad;
  std::vector<Matrix> biases_grad;
  std::vector<Matrix> pre_active;
  std::vector<Matrix> post_active;
  weight_grad.resize(layers_ - 1);
  biases_grad.resize(layers_ - 1);
  result = BatchTrainingForwardPass(Matrix(inputs), post_active, pre_active);
  auto grads =
      BatchCalculateGradients(result, Matrix(targets), post_active, pre_active);
  Step({grads.weight_grads, grads.bias_grads, grads.input_grad});
}

void Net::SetLayersActivations(
    const std::vector<ActivationFunctions>& functions) {
  if (functions.size() != layers_ - 1) {
    throw std::length_error(
        "Number of functions does not match number of layers with activation");
  }
  for (size_t i = 0; i < functions.size() - 1; ++i) {
    if (functions[i] == ActivationFunctions::SOFTMAX) {
      throw std::logic_error("Cannot apply softmax to hidden layers");
    }
  }
  activation_functions_ = functions;
}

double Net::ResolveActivation(ActivationFunctions func, double x) const {
  if (func == ActivationFunctions::RELU) {
    return ReLu(x);
  }
  if (func == ActivationFunctions::SIGMOID) {
    return Sigmoid(x);
  }
  if (func == ActivationFunctions::NO) {
    return x;
  }
  return x;
}

double Net::ResolveDerivative(ActivationFunctions func, double x) const {
  if (func == ActivationFunctions::NO) {
    return 1;
  }
  if (func == ActivationFunctions::RELU) {
    if (x > 0) {
      return 1;
    }
    return 0;
  }
  if (func == ActivationFunctions::SIGMOID) {
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

double Net::Sigmoid(double x) { return 1.0 / (1 + std::exp(-x)); }