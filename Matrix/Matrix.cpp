#include "Matrix.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string>
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
  for (const std::vector<double>& row : values) {
    if (row.size() != m_) {
      throw std::runtime_error("Inconsistent lenght");
    }
  }
  matrix_values_ = values;
}

Matrix::Matrix() {
  n_ = 0;
  m_ = 0;
}

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> list) {
  n_ = list.size();
  if (n_ == 0) {
    throw std::runtime_error("Zero length");
  }
  m_ = list.begin()->size();
  if (m_ == 0) {
    throw std::runtime_error("Zero length");
  }
  for (std::initializer_list<double> row : list) {
    if (row.size() != m_) {
      throw std::runtime_error("Inconsistent lenght");
    }
  }
  matrix_values_.reserve(n_);
  for (auto& row : list) {
    matrix_values_.emplace_back(row.begin(), row.end());
  }
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
        temp[c][i] += row[j] * other.matrix_values_[j][i];
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
  return Matrix(temp);
}

Matrix& Matrix::operator*=(double other) {
  for (std::vector<double>& row : matrix_values_) {
    for (double& item : row) {
      item *= other;
    }
  }
  return *this;
}

Matrix operator*(double first, const Matrix& other) {
  return other * first;
}

bool Matrix::operator==(const Matrix& other) const {
  if (n_ != other.n_ || m_ != other.m_) {
    return false;
  }
  for (size_t i = 0; i < n_; ++i) {
    for (size_t j = 0; j < m_; ++j) {
      if (matrix_values_[i][j] != other.matrix_values_[i][j]) {
        return false;
      }
    }
  }
  return true;
}

Matrix Matrix::zero(size_t height, size_t weight) {
  return Matrix(
      std::vector<std::vector<double>>(height, std::vector<double>(weight, 0)));
}

Matrix Matrix::one(size_t height, size_t weight) {
  return Matrix(
      std::vector<std::vector<double>>(height, std::vector<double>(weight, 1)));
}

Matrix Matrix::identity(size_t size) {
  std::vector<std::vector<double>> temp(size, std::vector<double>(size, 0));
  for (size_t i = 0; i < size; ++i) {
    temp[i][i] = 1;
  }
  return Matrix(temp);
}

void Matrix::load_from_txt(std::string file_name) {
  std::ifstream file(file_name);
  std::vector<double> values;
  double val;
  while (file >> val) {
    values.push_back(val);
  }
  if (values.size() < 2) {
    throw std::runtime_error("Invalid format");
  }
  n_ = static_cast<size_t>(values[0]);
  m_ = static_cast<size_t>(values[1]);
  if (values.size() != 2 + n_ * m_) {
    throw std::runtime_error("Too short input");
  }
  matrix_values_.resize(n_);
  int cnt = 2;
  for (size_t i = 0; i < n_; ++i) {
    for (size_t j = 0; j < m_; ++j) {
      matrix_values_[i].push_back(values[cnt]);
      cnt++;
    }
  }
}