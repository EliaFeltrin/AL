#!/bin/bash

# Check if CUDA source file is provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <CUDA_SOURCE_FILE>"
    exit 1
fi

# Compiler
NVCC=nvcc

# Flags for nvcc
NVCCFLAGS="-arch=sm_61"

# nvprof command
NVPROF=nvprof

# CUDA source file
CUDA_FILE=$1

# Target executable name (first letter of the CUDA file)
EXEC=$(basename $CUDA_FILE | cut -c1)

echo "Compiling CUDA program $CUDA_FILE"
# Compile CUDA program
$NVCC $NVCCFLAGS -o $EXEC $CUDA_FILE

echo "Running CUDA program $EXEC"
# Run nvprof
$NVPROF ./$EXEC
