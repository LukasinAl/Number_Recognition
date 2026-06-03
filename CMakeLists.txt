cmake_minimum_required(VERSION 3.12)
project(NeuralNetwork VERSION 2.0.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS "-O3")
add_subdirectory(extern/pybind11)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

set(MATRIX_SRC src/Matrix/Matrix.cpp)
set(NET_SRC src/Net/net.cpp)
set(BIND_SRC src/Bind/bind.cpp)

pybind11_add_module(network_core
    ${MATRIX_SRC}
    ${NET_SRC}
    ${BIND_SRC}
)
target_link_libraries(network_core PRIVATE ${Python3_LIBRARIES})

add_executable(test_net training/test.cpp ${NET_SRC} ${MATRIX_SRC})
add_executable(train_net training/train.cpp ${NET_SRC} ${MATRIX_SRC})
target_include_directories(test_net PRIVATE src/Net src/Matrix)
