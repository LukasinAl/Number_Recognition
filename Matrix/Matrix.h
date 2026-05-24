#pragma once
#include <initializer_list>
#include <string>
#include <vector>

class Matrix {
 private:
  size_t n_;
  size_t m_;
  std::vector<std::vector<double>> matrix_values_;

 public:
  Matrix(size_t height, size_t width, double default_values);
  Matrix(const std::vector<std::vector<double>>& values);
  Matrix(std::initializer_list<std::initializer_list<double>> list);
  Matrix(const std::vector<double>& input);
  Matrix();
  void Load_from_txt(const std::string& file_name);
  Matrix& Transpose();
  Matrix operator*(const Matrix& other) const;
  Matrix operator+(const Matrix& other) const;
  Matrix& operator*=(const Matrix& other);
  Matrix& operator+=(const Matrix& other);
  Matrix operator*(double other) const;
  Matrix& operator*=(double other);
  std::vector<std::vector<double>> Get_values() const ;
  bool operator==(const Matrix& other) const;
  static Matrix identity(size_t size);
  static Matrix zero(size_t height, size_t width);
  static Matrix one(size_t height, size_t width);
};

Matrix operator*(double first, const Matrix& other);