#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling SPMMax kernels by nvcc..."
nvcc -c -o spmmax_pooling_kernel.cu.o spmmax_pooling_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../
python build.py
