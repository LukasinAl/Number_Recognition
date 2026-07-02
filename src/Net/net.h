#pragma once
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <vector>

#include "../Matrix/Matrix.h"

/**
 * @brief enum class with classical activation functions for layers in MLP
 * @note last layer activattion require correct loss function
 * @see LossFunctions
 */
enum class ActivationFunctions { RELU, SIGMOID, SOFTMAX, NO };
/**
 * @brief enum class with classical loss fucntioins
 * @see ActivationFunctions
 */
enum class LossFunctions { MSE, CROSSENTROPY };

/**
 * @brief struct Gradients for compact passing of gradiennts in training proccess
 */
struct Gradients {
  std::vector<Matrix> weight_grads;  ///< gradient af weights
  std::vector<Matrix> bias_grads;    ///< gradients of biases
  Matrix input_grad;                 ///< input gradient for embendings etc.
};

class Net {
 private:
  size_t layers_;                 ///< number of layers in MLP
  std::vector<int> layer_sizes_;  ///< layerSizes in MLP
  std::vector<Matrix> weights_;   ///< weights of edges between layers
  std::vector<Matrix>
      biases_;  ///< biases of vertexes in layers (input layer does not have biasee)
  std::vector<ActivationFunctions>
      activation_functions_;  ///< ActivationFunctions by layer (input layer does not have activbateion)
  LossFunctions lossfunc;  ///< LossFunction

 public:
  /**
   * @brief Relu activation implementation
   * @param x input of function
   * @return result of activation
   */
  static double ReLu(double x);
  /**
   * @brief Sigmoid Actiavation implementation
   * @param x
   * @return result of activation
   */
  static double Sigmoid(double x);
  double learning_step;  ///< learning step
  /**
   * @brief Constuct net from number of layers and sizes fo layers
   * @param lay number of layers
   * @param sizes sizes of layers
   * @throw std::length_error if number of sizes does not equal number oof layers
   * @throw std::invalid argument if number of layers less than 2
   * @see Net(size_t lay, std::initializer_list<int> sizes) for inplace initialization
   * @note default activation NO, default loss MSE
   */
  Net(size_t lay, const std::vector<int>& sizes);
  /**
   * @brief Constuct net from number of layers and sizes fo layers
   * @param lay number of layers
   * @param sizes sizes of layers
   * @throw std::length_error if number of sizes does not equal number oof layers
   * @throw std::invalid argument if number of layers less than 2
   * @note delegates construction to Net(size_t lay, const std::vector<int>& sizes)
   */
  Net(size_t lay, std::initializer_list<int> sizes);
  /**
   * @brief fills all weights and biases with 0
   * @note practically useless
   */
  void fill_by_zeros();
  /**
   * @brief initializes all weights and biases with rnadom values
   */
  void FillBySmallRandomValues();
  /**
   * @brief takes an input vector and passes it into MLP
   * @param input input data
   * @throw std::length_error if input size does not match input layer size
   * @return result of forward pass throw network
   */
  std::vector<double> ForwardPass(const std::vector<double>& input) const;
  /**
   * @brief same as forward pass but for multiple smaples at a time
   * @param input Matrix of samples in row-like format
   * @throw std::length_error if sample size does not match input layer size
   * @throw std::invalid_argument if no data is given (0 rows)
   * @return result of Forward pass as Matrix in row-like format
   */
  Matrix BatchForwardPass(const Matrix& input) const;
  /**
   * @see ForwardPass
   */
  std::vector<double> ForwardPass(std::initializer_list<double> input) const;
  /**
   * @brief forward pass for training process
   * @param input - iput vector
   * @param last_pass_activation reference to a vector for saving activation values.
   * @param last_pass_pre_activation reference for saving preactivation values
   * @return result of forward passs
   * @throw std::length error if dimentiions of input vector does not match input_layer dimentions
   */
  std::vector<double> TrainingForwardPass(
      const std::vector<double>& input,
      std::vector<Matrix>& last_pass_activation,
      std::vector<Matrix>& last_pass_pre_acctivation);
    /**
    * @brief same as TrainingForwardPass but for batches
    * @param input - iput vector
    * @param last_pass_activation reference to a vector for saving activation values.
    * @param last_pass_pre_activation reference for saving preactivation values
    * @return result of forward passs
    * @throw std::length error if dimentiions of input vector does not match input_layer dimentions
    * @throw std::invalid argument if inputs are empty
    * @note inputs are treated in row like format
     */
  Matrix BatchTrainingForwardPass(
      const Matrix& input, std::vector<Matrix>& last_pass_activation,
      std::vector<Matrix>& last_pass_pre_acctivation);
  /**
   * @brief applies activation to each weight in matrix
   * @param vector input matrix
   * @param layer layer number (needed to resolve activation function)
   */
  void ApplyActivation(Matrix& vector, int layer) const;
  /**
   * @brief same as ApplyActivation but for derivative of activation function
   * @param vector input matrix
   * @param layer lyaer number
   */
  void ApplyDerivative(Matrix& vector, int layer) const;
  /**
   * @brief setup function used to set activations for each layer
   * @param functuions vectror of functions
   * @throw std::length error if number of functions does not match number of layers - 1
   * @throw std::logic error if softmax if applyed to hidden layer
   */
  void SetLayersActivations(const std::vector<ActivationFunctions>& functions);
  /**
   * @brief setup function used to set loss function
   * @param lossfunction sets lossfunction to given function
   */
  void SetLoss(const LossFunctions func);
  /**
   * @brief helper function which is used to compress ifs for activation
   * @param func - activation function
   * @param x - value
   * @return result of activation
   */
  double ResolveActivation(ActivationFunctions func, double x) const;
  /**
   * @brief helper function which is used to compress ifs for activation detivative
   * @param func - activation function
   * @param x - value
   * @return calculated derivative
   */
  double ResolveDerivative(ActivationFunctions func, double x) const;
  /**
   * @brief function which claculates loss 
   * @param result - result of forward pass
   * @param expected - expected result of forward pass
   * @return calculated loss
   */
  double CalculateLoss(std::vector<double>& result,
                       std::vector<double>& expected) const;
  /**
   * @brief backpropagation algorithm implementation
   * @param result - result of forward pass
   * @param expected - expected result
   * @param last_pass_activation - from Training forward pass
   * @param last_pass_pre_acctivation - from Training forward pass
   * @return Gradients - structure of biases, weigths, input gradients
   * @throw std::runtime error if combination of loss + output activations is not supported
   */
  Gradients CalculateGradients(
      const std::vector<double>& result, const std::vector<double>& expected,
      std::vector<Matrix>& last_pass_activation,
      std::vector<Matrix>& last_pass_pre_acctivation) const;
  /**
   * @brief same as CalculateGradients but for batches
   * @note gradients are averaged before returning
   * @note result and expected are supposed to be in row like order
   * @param result - result of forward pass
   * @param expected - expected result
   * @param last_pass_activation - from Training forward pass
   * @param last_pass_pre_acctivation - from Training forward pass
   * @return Gradients - structure of biases, weigths, input gradients
   */
  Gradients BatchCalculateGradients(
      const Matrix& result, const Matrix& expected,
      std::vector<Matrix>& last_pass_activation,
      std::vector<Matrix>& last_pass_pre_acctivation) const;
  /**
   * @brief High level training function
   * @param input - input vecors in row like iorder
   * @param expected - expected vectors in row like order
   * @param batch_size - size of training batch 
   * @param epochs - number of repetitions of training process
   * @throw std::length_error if length of inputs does not equal length of expected or size == 0
   */
  void Train(const std::vector<std::vector<double>>& inputs,
             const std::vector<std::vector<double>>& targets, int epochs,
             int batch_size);
  /**
   * @brief Trains model on whole inputs and targets.
   * @param inputs - input vectors
   * @param targets - expected results
   * @note all inputs are used as one batch
   * @throw std::length_error if length of inputs does not equal length of targets or size == 0
   */
  void BatchTrain(const std::vector<std::vector<double>>& inputs,
                  const std::vector<std::vector<double>>& targets);
  /**
   * @brief apply calculated derivatives from CalculateGradients
   * @param gradients
   */
  void Step(Gradients gradients);
  /**
   * @brief method used to save all weigths and configuration to binary file
   * @param filename - name of target file
   * @throw std::runtime_error if file is not opened successfully
   * @throw std::runtime error if loss or activation functions are missing
   */
  void DumpToBinary(const std::string& filename) const;
  /**
   * @brief method to read all model weigths and configuration from file
   * @param filename - target file
   * @note all model weigths and configuration would be overriden when using this method
   * @throw std::runtime_error if file is not opened successfully
   * @throw std::runtime error if loss or activation functions are missing or invalid
   */
  void ReadFromBinary(const std::string& filename);
};