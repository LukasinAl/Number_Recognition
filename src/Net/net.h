#pragma once
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <vector>
#include "../Matrix/Matrix.h"

enum class ActivationFunnctions { RELU, SIGMOID, SOFTMAX, NO };
enum class LossFunctions { MSE, CROSSENTROPY };

class Net {
 private:
  size_t layers_;
  std::vector<int> layer_sizes_;
  std::vector<Matrix> weights_;
  std::vector<Matrix> biases_;
  std::vector<ActivationFunnctions> activation_functions_;
  LossFunctions lossfunc;

 public:
  static double ReLu(double x);
  static double Sigmoid(double x);
  double learning_step;
  Net(size_t lay, const std::vector<int>& sizes);
  Net(size_t lay, std::initializer_list<int> sizes);
  void fill_by_zeros();
  void FillBySmallRandomValues();
  std::vector<double> ForwardPass(const std::vector<double>& input) const;
  Matrix BatchForwardPass(const Matrix& input) const;
  std::vector<double> ForwardPass(std::initializer_list<double> inputT) const;
  std::vector<double> TrainingForwardPass(
      const std::vector<double>& input,
      std::vector<Matrix>& last_pass_activation,
      std::vector<Matrix>& last_pass_pre_acctivation);
  Matrix BatchTrainingForwardPass(
      const Matrix& input, std::vector<Matrix>& last_pass_activation,
      std::vector<Matrix>& last_pass_pre_acctivation);
  void ApplyActivation(Matrix& vector, int layer) const;
  void ApplyDerivative(Matrix& vector, int layer) const;
  void SetLayersActivations(const std::vector<ActivationFunnctions>& functions);
  void SetLoss(const LossFunctions func);
  double ResolveActivation(ActivationFunnctions func, double x) const;
  double ResolveDerivative(ActivationFunnctions func, double x) const;
  double CalculateLoss(std::vector<double>& result,
                       std::vector<double>& expected) const;
  std::pair<std::vector<Matrix>, std::vector<Matrix>> CalculateGradients(
      const std::vector<double>& result, const std::vector<double>& expected,
      std::vector<Matrix>& last_pass_activation,
      std::vector<Matrix>& last_pass_pre_acctivation) const;
  std::pair<std::vector<Matrix>, std::vector<Matrix>> BatchCalculateGradients(
      const Matrix& result, const Matrix& expected,
      std::vector<Matrix>& last_pass_activation,
      std::vector<Matrix>& last_pass_pre_acctivation) const;
  void Train(const std::vector<std::vector<double>>& inputs,
             const std::vector<std::vector<double>>& targets, int epochs,
             int batch_size);
  void BatchTrain(const std::vector<std::vector<double>>& inputs,
             const std::vector<std::vector<double>>& targets, int epochs,
             int batch_size);
  void Step(std::pair<std::vector<Matrix>, std::vector<Matrix>> gradients);
  void DumpToBinary(const std::string& filename) const;
  void ReadFromBinary(const std::string& filename);
};