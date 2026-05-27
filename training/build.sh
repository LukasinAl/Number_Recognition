#!/bin/bash
mkdir build
cd build
cmake ..
make

g++ -std=c++17 -I src/Matrix -I src/net -L include -L include train.cpp -lNet -lMatrix -o train
