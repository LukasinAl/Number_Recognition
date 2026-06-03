#!/bin/bash
mkdir build
cd build
cmake ..
make

cd ..
mv build/network_core.cpython-312-x86_64-linux-gnu.so include/network_core.so
mv build/test_net training/test_net
mv build/train_net training/train_net
rm -rf build