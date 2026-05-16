#pragma once
#include <vector>

class Matrix {
 private:
  size_t n_;
  size_t m_;
  std::vector<std::vector<double>> matrix_values_;

 public:
  Matrix(size_t height, size_t width, double default_values);
  Matrix(const std::vector<std::vector<double>>& values);
  void Transpose();
  Matrix operator*(const Matrix& other) const;
  Matrix operator+(const Matrix& other) const;
  Matrix& operator*=(const Matrix& other);
  Matrix& operator+=(const Matrix& other);
  Matrix operator*(double other) const;
  static Matrix identity(size_t size);
  static Matrix zero(size_t height, size_t width);
  static Matrix one(size_t height, size_t width);
};

Matrix operator*(double first, const Matrix& other);