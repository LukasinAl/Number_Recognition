#!/bin/bash
mkdir build
cd build
cmake ..
make
cd ..

g++ -std=c++17 -O3 -I src/Matrix -I src/net -L build/src/net -L build/src/Matrix test.cpp -lNet -lMatrix -o test
