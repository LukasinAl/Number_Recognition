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

Matrix Matrix::operator*(const Matrix& other) const {
  if (m_ != other.n_) {
    throw std::runtime_error("Bad dimentions");
  }
  std::vector<std::vector<double>> temp(n_, std::vector<double>(other.m_, 0));
  int c = 0;
  for (const std::vector<double>& row : matrix_values_) {
    for (size_t i = 0; i < other.m_; ++i) {
      for (size_t j = 0; j < other.n_; ++j) {
        temp[c][i] += row[j] * other.matrix_values_[i][j];
      }
    }
    c++;
  }
  return Matrix(temp);
}

Matrix& Matrix::operator*=(const Matrix& other) {
  *this = *this * other;
  return *this;
}

Matrix Matrix::operator+(const Matrix& other) const {
  if (n_ != other.n_ || m_ != other.m_) {
    throw std::runtime_error("Wrong dimentions");
  }
  std::vector<std::vector<double>> temp(n_, std::vector<double>(m_, 0));
  for (size_t i = 0; i < n_; ++i) {
    for (size_t j = 0; j < m_; ++j) {
      temp[i][j] = matrix_values_[i][j] + other.matrix_values_[i][j];
    }
  }
  return Matrix(temp);
}

Matrix& Matrix::operator+=(const Matrix& other) {
  *this = *this + other;
  return *this;
}

Matrix Matrix::operator*(double other) const {
  std::vector<std::vector<double>> temp(n_, std::vector<double>(m_, 0));
  for (size_t i = 0; i < n_; ++i) {
    for (size_t j = 0; j < m_; ++j) {
      temp[i][j] = matrix_values_[i][j] * other;
    }
  }
}

Matrix& Matrix::operator*=(double other) {
  for (std::vector<double>& row : matrix_values_) {
    for (double& item : row) {
      item *= other;
    }
  }
}

Matrix operator*(double first, const Matrix& other) {
  return other * first;
}