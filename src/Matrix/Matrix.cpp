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
    : n(height), m(width) {
  if (height == 0 || width == 0) {
    throw std::invalid_argument("Zero length");
  }
  matrix_values = std::vector<std::vector<double>>(
      n, std::vector<double>(m, default_values));
}

Matrix::Matrix(const std::vector<std::vector<double>>& values) {
  if (values.empty()) {
    throw std::length_error("Zero length");
  }
  if (values[0].empty()) {
    throw std::length_error("Zero length");
  }
  n = values.size();
  m = values[0].size();
  for (const std::vector<double>& row : values) {
    if (row.size() != m) {
      throw std::length_error("Inconsistent lenght");
    }
  }
  matrix_values = values;
}

Matrix::Matrix() {
  n = 0;
  m = 0;
}

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> list) {
  n = list.size();
  if (n == 0) {
    throw std::runtime_error("Zero length");
  }
  m = list.begin()->size();
  if (m == 0) {
    throw std::runtime_error("Zero length");
  }
  for (std::initializer_list<double> row : list) {
    if (row.size() != m) {
      throw std::runtime_error("Inconsistent lenght");
    }
  }
  matrix_values.reserve(n);
  for (auto& row : list) {
    matrix_values.emplace_back(row.begin(), row.end());
  }
}

Matrix::Matrix(const std::vector<double>& input) : Matrix(input.size(), 1, 0) {
  for (size_t i = 0; i < input.size(); ++i) {
    matrix_values[i][0] = input[i];
  }
}

Matrix Matrix::Transpose() const {
  if (n == 0 || m == 0) {
    throw std::length_error("Zero dimnention in Transpose()");
  }
  std::vector<std::vector<double>> temp(m, std::vector<double>(n));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      temp[j][i] = matrix_values[i][j];
    }
  }
  return Matrix(temp);
}

Matrix Matrix::operator*(const Matrix& other) const {
  if (m != other.n) {
    throw std::length_error("Wrong dimentions in operator*" +
                            std::to_string(m) + " " + std::to_string(other.n));
  }
  std::vector<std::vector<double>> temp(n, std::vector<double>(other.m, 0));
  if (n * m * other.m < 10e4) {  // use cache friendly loop
    for (size_t i = 0; i < n; ++i) {
      for (size_t k = 0; k < m; ++k) {
        for (size_t j = 0; j < other.m; ++j) {
          temp[i][j] += matrix_values[i][k] * other.matrix_values[k][j];
        }
      }
    }
  } else { //use single vector to ensure coonstinious data layout
    std::vector<double> first_flattened;
    std::vector<double> second_flattened;
    std::vector<double> temp_flattened(n * other.m, 0.0);
    first_flattened.resize(n * m);
    second_flattened.resize(other.n * other.m);
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        first_flattened[m * j + i] = matrix_values[j][i];
      }
      for (size_t j = 0; j < other.m; ++j) {
        second_flattened[i * other.m + j] = other.matrix_values[i][j];
      }
    }
    for (size_t i = 0; i < n; ++i) {
      for (size_t k = 0; k < m; ++k) {
        for (size_t j = 0; j < other.m; ++j) {
          temp_flattened[i * other.m + j] += first_flattened[i * m + k] * second_flattened[k * other.m + j];
        }
      }
    }
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < other.m; ++j) {
        temp[i][j] = temp_flattened[i * other.m + j];
      }
    }
  }
  return Matrix(temp);
}

Matrix& Matrix::operator*=(const Matrix& other) {
  *this = *this * other;
  return *this;
}

Matrix Matrix::operator+(const Matrix& other) const {
  if (n != other.n || m != other.m) {
    throw std::length_error("Wrong dimentions in operator+");
  }
  std::vector<std::vector<double>> temp(n, std::vector<double>(m, 0));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      temp[i][j] = matrix_values[i][j] + other.matrix_values[i][j];
    }
  }
  return Matrix(temp);
}

Matrix& Matrix::operator+=(const Matrix& other) {
  *this = *this + other;
  return *this;
}

Matrix Matrix::operator*(double other) const {
  std::vector<std::vector<double>> temp(n, std::vector<double>(m, 0));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      temp[i][j] = matrix_values[i][j] * other;
    }
  }
  return Matrix(temp);
}

Matrix& Matrix::operator*=(double other) {
  for (std::vector<double>& row : matrix_values) {
    for (double& item : row) {
      item *= other;
    }
  }
  return *this;
}

Matrix operator*(double first, const Matrix& other) { return other * first; }

bool Matrix::operator==(const Matrix& other) const {
  if (n != other.n || m != other.m) {
    return false;
  }
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      if (matrix_values[i][j] != other.matrix_values[i][j]) {
        return false;
      }
    }
  }
  return true;
}

Matrix Matrix::operator-() const { return (*this) * -1; }

Matrix Matrix::operator-(const Matrix& other) const {
  return (*this) + (other * -1);
}

Matrix& Matrix::operator-=(const Matrix& other) {
  return ((*this) += (other * -1));
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

void Matrix::LoadFromTxt(const std::string& file_name) {
  std::ifstream file(file_name);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file");
  }
  std::vector<double> values;
  double val;
  while (file >> val) {
    values.push_back(val);
  }
  if (values.size() < 2) {
    throw std::runtime_error("Invalid format");
  }
  n = static_cast<size_t>(values[0]);
  m = static_cast<size_t>(values[1]);
  if (values.size() != 2 + n * m) {
    throw std::runtime_error("Too short input");
  }
  matrix_values.resize(n);
  int cnt = 2;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      matrix_values[i].push_back(values[cnt]);
      cnt++;
    }
  }
}

Matrix Matrix::ElementWiseMultiplication(const Matrix& other) const {
  if (n != other.n || m != other.m) {
    throw std::runtime_error("Wrong dimentions in ElementWiseMultiplication");
  }
  Matrix res(n, m, 0);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < m; ++j)
      res.matrix_values[i][j] = matrix_values[i][j] * other.matrix_values[i][j];
  return res;
}