#!/bin/bash
set -e  # Exit on error

echo "Cleaning build directory..."
rm -rf build
mkdir -p build
cd build

echo "Configuring with CMake..."
cmake ..

echo "Building..."
make -j4

echo "Build complete."
cd ..
