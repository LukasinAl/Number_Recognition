#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../src/Net/net.h"

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
  net.SetLossMSE();
  net.ReadFromBinary("weights/weights2.bin");
  net.learning_step = 0.003;
  net.SetLayersActivations({ActivationFunnctions::RELU,
                            ActivationFunnctions::RELU,
                            ActivationFunnctions::SIGMOID});

  const int epochs = 1;
  const int batch_size = 64;
  int total_samples_processed = 0;  // counts all samples across epochs

  for (int ep = 0; ep < epochs; ++ep) {
    std::ifstream file("data/emnist-digits-train.csv");
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open data/emnist-digits-train.csv");
    }
    std::string line;

    std::vector<std::vector<double>> batch_inputs;
    std::vector<std::vector<double>> batch_targets;
    batch_inputs.reserve(batch_size);
    batch_targets.reserve(batch_size);

    int samples_in_epoch = 0;

    while (std::getline(file, line)) {
      if (line.empty())
        continue;

      auto data = ParceCsvLine(line);
      int label = data.first;
      std::vector<double> input = data.second;

      // One‑hot target
      std::vector<double> target(10, 0.0);
      target[label] = 1.0;

      batch_inputs.push_back(input);
      batch_targets.push_back(target);
      samples_in_epoch++;

      // When batch is full, train on it
      if (batch_inputs.size() == batch_size) {
        net.Train(batch_inputs, batch_targets, 1, batch_size);
        total_samples_processed += batch_size;

        // Save weights every 10,000 samples
        if (total_samples_processed % 1000 == 0) {
          net.DumpToBinary("weights/weights2.bin");
          std::cout << "Saved weights after " << total_samples_processed
                    << " samples\n";
        }

        // Clear batch
        batch_inputs.clear();
        batch_targets.clear();
      }
    }

    // Process any remaining samples (partial batch)
    if (!batch_inputs.empty()) {
      int actual_batch_size = batch_inputs.size();
      net.Train(batch_inputs, batch_targets, 1, actual_batch_size);
      total_samples_processed += actual_batch_size;
      if (total_samples_processed % 10000 == 0 ||
          total_samples_processed % 10000 < actual_batch_size) {
        net.DumpToBinary("weights/weights2.bin");
        std::cout << "Saved weights after " << total_samples_processed
                  << " samples (final batch)\n";
      }
    }

    std::cout << "Epoch " << ep + 1 << " finished. Total samples processed: "
              << total_samples_processed << "\n";
  }

  // Final save
  net.DumpToBinary("weights/weights2.bin");
  std::cout << "Training complete. Final weights saved.\n";

  return 0;
}