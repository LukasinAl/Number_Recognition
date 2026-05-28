#pragma once
#include <functional>
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
  std::vector<Matrix> last_pass_activation;
  std::vector<Matrix> last_pass_pre_acctivation;
  std::function<double(double)> activation_function_;
  std::function<double(double)> activation_funtion_derivative_;
  std::function<double(std::vector<double>&, std::vector<double>&)>
      loss_function_;

 public:
  bool activateOutput;
  double learning_step;
  Net(size_t lay, const std::vector<int>& sizes);
  Net(size_t lay, std::initializer_list<int> sizes);
  void fill_by_zeros();
  void FillBySmallRandomValues();
  std::vector<double> ForwardPass(const std::vector<double>& input) const;
  std::vector<double> ForwardPass(std::initializer_list<double> inputT) const;
  std::vector<double> TrainingForwardPass(const std::vector<double>& input);
  void ApplyActivation(Matrix& vector) const;
  void SetActivationRelU();
  void SetActivationSigmoid();
  double CalculateLoss(std::vector<double>& result,
                       std::vector<double>& expected) const;
  void SetLossMSE();
  std::pair<std::vector<Matrix>, std::vector<Matrix>> CalculateGradients(
      const std::vector<double>& result,
      const std::vector<double>& expected) const;
  void Step(std::pair<std::vector<Matrix>, std::vector<Matrix>> gradients);
  void DumpToBinary(const std::string& filename) const;
  void ReadFromBinary(const std::string& filename);
};