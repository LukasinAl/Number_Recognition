#pragma once
#include <initializer_list>
#include <stdexcept>
#include <vector>
#include "../Matrix/Matrix.h"

class Net {
 private:
  size_t layers_;
  std::vector<int> layer_sizes_;
  std::vector<Matrix> weights_;
  std::vector<Matrix> biases_;

 public:
  Net(size_t lay, const std::vector<int>& sizes);
  Net(size_t lay, std::initializer_list<int> sizes);
  void fill_by_zeros();
  std::vector<double> forward_pass(const std::vector<double>& input) const;
  std::vector<double> forward_pass(std::initializer_list<double> input) const;
};