#include "Matrix.h"
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <vector>

Matrix::Matrix(size_t height, size_t width, double default_values)
    : n_(height), m_(width) {
  matrix_values_ = std::vector<std::vector<double>>(
      n_, std::vector<double>(m_, default_values));
}

Matrix::Matrix(const std::vector<std::vector<double>>& values) {
  if (values.empty()) {
    throw std::runtime_error("Zero length");
  }
  if (values[0].empty()) {
    throw std::runtime_error("Zero length");
  }
  n_ = values.size();
  m_ = values[0].size();
  for (const std::vector<double> row : values) {
    if (row.size() != m_) {
      throw std::runtime_error("Inconsistent lenght");
    }
  }
  matrix_values_ = values;
}

void Matrix::Transpose() {
  std::vector<std::vector<double>> temp(m_, std::vector<double>(n_));
  for (size_t i = 0; i < n_; ++i) {
    for (size_t j = 0; j < m_; ++j) {
      temp[j][i] = matrix_values_[i][j];
    }
  }
  matrix_values_ = std::move(temp);
}