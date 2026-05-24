#include "net.h"
#include <vector>
#include "../Matrix/Matrix.h"

std::vector<double> Net::forward_pass(std::vector<double>& input) {
  Matrix temp(input);
  if (input.size() != layer_sizes_[0]) {
    throw std::runtime_error("Wrong dimentions");
  }
  for (size_t i = 0; i < layers_ - 1; ++i) {
    temp = weights_[i] * temp + biases_[i];
  }
}

void Net::fill_by_zeros() {
  weights_.resize(layers_ - 1);
  for (size_t i = 0; i < layers_ - 1; ++i) {
    weights_[i] = Matrix(std::vector<std::vector<double>>(
        layer_sizes_[i + 1], std::vector<double>(layer_sizes_[i], 0)));
    biases_[i] = Matrix(std::vector<std::vector<double>>(
        layer_sizes_[i], std::vector<double>(1, 0)));
  }
}

Net::Net(int lay, std::initializer_list<int> sizes)
    : layers_(lay), layer_sizes_(sizes) {
  if (layers_ < 2) {
    throw std::runtime_error("Too litte layers");
  }
  if (layers_ + 1 != layer_sizes_.size()) {
    throw std::runtime_error("Too little layer sizes");
  }
}

Net::Net(size_t lay, std::vector<int>& sizes) : layers_(lay), layer_sizes_(sizes) {
  if (layers_ < 2) {
    throw std::runtime_error("Too litte layers");
  }
  if (layers_ + 1 != layer_sizes_.size()) {
    throw std::runtime_error("Too little layer sizes");
  }
}