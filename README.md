# Number Recognition

A handwritten digit recognition project built around a **fully custom neural network implementation in C++20** with Python bindings provided through **pybind11**.

The project implements every core component of a multilayer perceptron (MLP) from scratch including matrix operations, forward propagation, backpropagation, activation functions, loss functions, weight serialization, and batch inference without relying on machine learning frameworks.

A PyQt6 application demonstrates the trained model by allowing users to draw digits and classify them in real time.

---

## Features

- Custom matrix library
- Fully connected neural network (MLP)
- Forward propagation
- Backpropagation
- Batch training and inference
- Activation functions
  - ReLU
  - Sigmoid
  - Softmax
  - Linear
- Loss functions
  - Mean Squared Error
  - Cross Entropy
- Binary weight serialization
- Python bindings by pybind11
- Interactive PyQt6 drawing interface
- Unit tests for matrix operations
- Pre-Trained weights with test accuracy of 98.64%

---

## Project Structure

```
.
├── app/                 # Python GUI application
│   ├── gui.py
│   └── network_core.so
│
├── src/
│   ├── Matrix/          # Matrix implementation
│   ├── Net/             # Neural network implementation
│   └── Bind/            # pybind11 bindings
│
├── tests/
│   ├── test_matrix/
│   └── test_net/
│
├── weights/
│   └── FinalWeights.bin  # Weights for pre-trained model
│
├── build.sh
├── launch.sh
└── CMakeLists.txt
```

---

# Neural Network

The project implements a classic feed-forward multilayer perceptron.

Example architecture used for digit recognition:

```
784
 ↓
500
 ↓
128
 ↓
10
```

Input:
- 28×28 grayscale image
- flattened into a vector of 784 values

Output:
- probabilities for digits **0–9** (if using softmax activation for last layer)

---

## Implemented Components

### Matrix Library

Custom matrix implementation supporting

- matrix multiplication (using cache friendly loops)
- addition
- subtraction
- transpose
- element-wise operations

---

### Forward Pass

Each layer computes

```
z = Wx + b
a = activation(z)
```

Implemented for both

- single samples
- batches

---

### Backpropagation

Training computes gradients for

- weights
- biases
- network input

Gradient information is stored inside a dedicated `Gradients` structure, making it easy to extend the project with additional layers or embedding layers.

---

### Activation Functions

- ReLU
- Sigmoid
- Softmax
- Linear (No activation)

---

### Loss Functions

- Mean Squared Error
- Cross Entropy

---

### Serialization

The network supports saving and loading weights from binary files.

Example:

```cpp
net.DumpToBinary("weights/FinalWeights.bin")
net.ReadFromBinary("weights/FinalWeights.bin");
```

This allows trained models to be reused directly by the GUI application.

---

# Python Interface

The C++ implementation is exposed to Python using **pybind11**.

This combines

- C++ execution speed
- Python development speed

---

# GUI

The application provides a simple drawing canvas where users can

- draw a handwritten number (multy digit numbers are supported)
- clear the canvas
- Classify number using trained model

Internally each continious part of drawing is converted into a 28×28 representation and passed into MLP

---

# Running

## Requirements

- C++20 compiler
- CMake ≥ 3.12
- Python3
- pybind11

## Cloning

```bash
git clone https://github.com/LukasinAl/Number_Recognition.git
git submodule update --init
```

## Building

Build library
```bash
mkdir build
cd build
cmake ..
make
cd ..
```

or simply run

```bash
./build.sh
```

Library is expected to be in /include folder under name network_core.so

## Run GUI

In order to run GUI creation of venv is required
```bash
cd app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..
```

in order to run 

```bash
python3 app/gui.py
```

## Launch GUI rightaway

In oreder to launch GUI without any additional actions it is possible to run

```bash
./launch.sh
```

---

# Testing

Matrix unit tests are located in

```
tests/test_matrix
```

These verify correctness of the custom matrix implementation using predefined datasets.

---

# Technologies

- C++20
- Python
- pybind11
- gtest (if tests are needed)
- PyQt6
- CMake

---

# Future Improvements

Possible extensions include

- Better unit test for Matrix
- Convolutional Neural Networks (CNNs)
- GPU acceleration
- Refactor Matrix class to store Matrix in single vector instead of 2d vector

---

# Motivation

The purpose of this project is educational: to understand how modern neural networks work by implementing every major component from scratch rather than relying on existing machine learning libraries.

The project demonstrates how matrix algebra, activation functions, gradient computation, and optimization come together to perform handwritten digit recognition.

---

# License

This project is distributed under GPLv3 licence in order to comply with copyleft clause in PyQt6 licence.
This also allows to comply with pybind11 and gtest licences.