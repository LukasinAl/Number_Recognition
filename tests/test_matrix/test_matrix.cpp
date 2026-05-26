#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include "../../src/Matrix/Matrix.h"

bool fileExists(const std::string& path) {
  std::ifstream f(path.c_str());
  return f.is_open();
}

TEST(Matrix, test_equality) {
  Matrix mat1;
  Matrix mat2;
  mat1.LoadFromTxt("data/mat1.txt");
  mat2.LoadFromTxt("data/mat2.txt");
  EXPECT_EQ(mat1, mat1);
  EXPECT_EQ(mat2, mat2);
}

TEST(Matrix, test_addition) {
  Matrix mat1;
  Matrix mat2;
  mat1.LoadFromTxt("data/mat1.txt");
  mat2.LoadFromTxt("data/mat2.txt");
  EXPECT_EQ(mat1 + mat1, 2 * mat1);
  EXPECT_EQ(mat2 + mat2 + mat2, mat2 * 3);
  EXPECT_EQ(mat2 + mat2 + mat2, 3 * mat2);
  EXPECT_EQ(mat1 + mat2, mat2 + mat1);
}

TEST(matrix, test_multiplication) {
  Matrix mat1;
  Matrix mat2;
  Matrix mat3;
  Matrix mat2_3;
  mat1.LoadFromTxt("data/mat1.txt");
  mat2.LoadFromTxt("data/mat2.txt");
  mat3.LoadFromTxt("data/mat3.txt");
  mat2_3.LoadFromTxt("data/mat2_3_exp.txt");
  EXPECT_EQ(mat1 * mat2, mat2);
  EXPECT_EQ(mat1 * mat3, mat3);
  EXPECT_EQ(mat1, Matrix::identity(3));
  EXPECT_EQ(mat2 * mat3, mat2_3);
}

TEST(matrix, test_big) {
  int cnt = 1;
  while (true) {
    Matrix mat1;
    Matrix mat2;
    Matrix mat3;
    if (fileExists("data/big_mat_first" + std::to_string(cnt) + ".txt")) {
      mat1.LoadFromTxt("data/big_mat_first" + std::to_string(cnt) + ".txt");
      mat2.LoadFromTxt("data/big_mat_second" + std::to_string(cnt) + ".txt");
      mat3.LoadFromTxt("data/big_mat_exp" + std::to_string(cnt) + ".txt");
    } else {
      break;
    }
    EXPECT_EQ(mat1 * mat2, mat3);
    cnt++;
  }
}

int main() {
  ::testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}