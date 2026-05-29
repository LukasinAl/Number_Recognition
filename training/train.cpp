#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "src/net/net.h"

std::pair<int, std::vector<double>> ParceCsvLine(const std::string& line) {
  std::stringstream ss(line);
  std::string token;
  int label;
  std::vector<double> input;
  input.resize(784);
  std::getline(ss, token, ',');
  label = std::stoi(token);

  int count = 0;
  while (std::getline(ss, token, ',')) {
    input[count] = static_cast<double>(std::stoi(token));
    input[count] /= 255;
    count++;
  }
  if (count != 784) {
    throw std::runtime_error("corrupted line");
  }
  return {label, input};
}

int main() {
  Net net(4, {784, 500, 128, 10});
  net.SetActivationRelU();
  net.SetLossMSE();
  net.FillBySmallRandomValues();
  net.learning_step = 0.001;
  net.DumpToBinary("weights/weights.bin");

  int epoch = 30;
  int batchsize = 64;
  int iter = 0;

  for (int ep = 0; ep < epoch; ++ep) {
    std::ifstream file("data/mnist_train.csv");
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file");
    }

    std::string line;
    while (std::getline(file, line)) {
      if (line.empty()) {
        continue;
      }
      std::vector<double> input;
      int label;
      auto data = ParceCsvLine(line);
      label = data.first;
      input = data.second;
      std::vector<double> expected(10, 0);
      std::vector<double> result;
      expected[label] = 1;

      result = net.TrainingForwardPass(input);
      auto grads = net.CalculateGradients(result, expected);
      net.Step(grads);
      iter++;
      if (iter % 10000 == 0) {
        net.DumpToBinary("weights/weights.bin");
        std::cout << iter << '\n';
      }
    }
    double total_loss = 0;
    int correct = 0;
    int total = 0;
    std::ifstream test("data/mnist_test.csv");
    if (!test.is_open()) {
      throw std::runtime_error("Cannot open file");
    }
    while (std::getline(test, line)) {
      if (line.empty()) {
        continue;
      }
      std::vector<double> input;
      int label;
      auto data = ParceCsvLine(line);
      label = data.first;
      input = data.second;
      std::vector<double> expected(10, 0);
      std::vector<double> result;
      expected[label] = 1;

      result = net.ForwardPass(input);
      total_loss += net.CalculateLoss(result, expected);
      total++;
      double max = 1e6;
      for (double item : result) {
        max = std::max(max, item);
      }
      if (max == result[label] || result[label] > 0.01) {
        correct++;
      }
    }
    std::cout << "Total loss:" << total_loss << '\n';
    std::cout << "Coorrect:" << static_cast<double>(correct) / static_cast<double>(total) * 100.0 << "%\n";
  }
}
