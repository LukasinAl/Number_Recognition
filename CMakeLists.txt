cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_subdirectory(src/Matrix)
add_subdirectory(src/Net)
