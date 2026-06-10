#pragma once
#include <initializer_list>
#include <string>
#include <vector>

/**
 * @brief Custom implementations of Matrix containg doubles class.
 * 
 * @note All values are stored in 2d array
 */
class Matrix {
 public:
  size_t n; ///< Numbers of rows (height)
  size_t m; ///< Number of columns (width)
  std::vector<std::vector<double>> matrix_values; ///< 2d Storage array
  /**
   * @brief Constructor of matrix with fixed dimensions and same values
   * @param height Height of constructed matrix
   * @param width Width of constructed matrix
   * @param default_values Default value for matrix elements
   * @throws std::invalid_argument if height == 0 or width == 0
   * @see zero to get matrix initialized with zeros
   */
  Matrix(size_t height, size_t width, double default_values);
  /**
   * @brief Construct matrix from 2d vector. Outer dimentions is height, inner dimention is width
   * @param values 2d array of values
   * @throws std::length_error if eather of dimentiomns is 0 or inner dimention is inconsistent within rows
   * @see Matrix(std::initializer_list<std::initializer_list<double>> list) to create Matrix from initializer list
   */
  Matrix(const std::vector<std::vector<double>>& values);
  /**
   * @brief Construct matrix from 2d initializer list
   * @param list 2d initializer list
   * @throws std::length_error if eather of dimentiomns is 0 or inner dimention is inconsistent within rows
   */
  Matrix(std::initializer_list<std::initializer_list<double>> list);
  /**
   * @brief Construct matrix from vector. Output is column of height input.size()
   * @param input vector of values
   * @throws std::length_error if input.size() == 0
   * @note delegates construction to Matrix(height, width, default), that initializes with values
   */
  Matrix(const std::vector<double>& input);
  /**
   * @brief Defalut constructor
   * @warning Only use this constructor if Matrix would be redefined before usage
   */
  Matrix();
  /**
   * @brief Returns transposed matrix
   * @throws std::length_error if dimentions is 0
   * @returns transposed matrix
   */
  Matrix Transpose() const;
  /**
   * @brief Overload for * operator
   * @throws std::length_error if dimentions do not match
   * @param other second operand
   * @returns result of multiplication
   */
  Matrix operator*(const Matrix& other) const;
  /**
   * @brief Overload for + operator
   * @throws std::length_error if dimentions do not match
   * @param other second operand
   * @return result of addition
   */
  Matrix operator+(const Matrix& other) const;
  /**
   * @brief Overload for *= operator
   * @throws std::length_error if dimentions do not match
   * @param other second operand
   * @returns reference to left operand
   * @note modifies matrix
   */
  Matrix& operator*=(const Matrix& other);
  /**
   * @brief Overload for += operator
   * @throws std::length_error if dimentions do not match
   * @param other second operand
   * @returns reference to left operand
   * @note modifies matrix
   */
  Matrix& operator+=(const Matrix& other);
  /**
   * @brief Multiplication on double values, (elementwise multiplication on doublw)
   * @param other second operand
   * @returns result of multiplication
   */
  Matrix operator*(double other) const;
  /**
   * @brief Multiplication on double
   * @param other second operand
   * @returns reference to left operand
   */
  Matrix& operator*=(double other);
  /**
   * @brief multiples matrix by -1
   * @returns result of multiplication
   */
  Matrix operator-() const;
  /**
   * @brief overload of - operator
   * @param other second operand
   * @returns result of subtractions
   */
  Matrix operator-(const Matrix& other) const;
  /**
   * @brief overload of -= operator
   * @param other second operand
   * @returns reference to left operand
   */
  Matrix& operator-=(const Matrix& other);
  /**
   * @brief overload of == operand
   * @param other second operand
   * @returns bool : True if equal False if not
   */
  bool operator==(const Matrix& other) const;
  /**
   * @brief implement elementwise multiplication
   * @throws std::length_error if dimentions do not match
   * @param other second operand
   * @returns result of multiplication
   */
  Matrix ElementWiseMultiplication(const Matrix& other) const;
  /**
   * @brief method which loads matrix from txt.
   * @throws std::runtime_error if cannot open file, or invalid format
   * @param file_name std::string path to file
   * @note Format: first line dimentions, other lines values
   * @code
   * 3 3
   * 1 0 0
   * 0 1 0
   * 0 0 1
   * @endcode
   */
  void LoadFromTxt(const std::string& file_name);
  /**
   * @brief Static method to get identity matrix
   * @param size - size of matrix
   */
  static Matrix identity(size_t size);
  /**
   * @brief Static method to get matrix full of zeros
   * @param size - size of matrix
   */
  static Matrix zero(size_t height, size_t width);
  /**
   * @brief Static method to get matrix full of ones
   * @param size - size of matrix
   */
  static Matrix one(size_t height, size_t width);
};
/**
 * @brief function to allow left side multilplication of matrix
 * @param first - double
 * @param other matrix
 * @returns result of multiplication
 */
Matrix operator*(double first, const Matrix& other);