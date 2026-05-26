#pragma once
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <vector>
#include "../Matrix/Matrix.h"

class Net {
 private:
  bool activateOutput;
  size_t layers_;
  std::vector<int> layer_sizes_;
  std::vector<Matrix> weights_;
  std::vector<Matrix> biases_;
  std::function<double(double)> activation_function_;
  std::function<double(std::vector<double>&, std::vector<double>&)>
      loss_function_;

 public:
  Net(size_t lay, const std::vector<int>& sizes);
  Net(size_t lay, std::initializer_list<int> sizes);
  void fill_by_zeros();
  void FillBySmallRandomValues();
  std::vector<double> ForwardPass(const std::vector<double>& input) const;
  std::vector<double> ForwardPass(std::initializer_list<double> input) const;
  void ApplyActivation(Matrix& vector) const;
  void SetActivationRelU();
  void SetActivationSigmoid();
  void SetActivateOutput(bool value);
  double CalculateLoss(std::vector<double>& result,
                       std::vector<double>& expected) const;
  void SetLossMSE();
  std::pair<std::vector<Matrix>, std::vector<Matrix>> CalculateGradients(
      std::vector<double>& result, std::vector<double>& expected) const;
};