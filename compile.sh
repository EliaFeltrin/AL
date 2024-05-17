#!/bin/bash

#./param_visualization_n_dim_par2 -mN 10 -MN 10 -mM 3 -MM 3 -ml=0 -mu 0.1 -l -a 1000 -i 300 -r 0.1 -PCRl 0.3,2,0,1 -f -c 


# Check if CUDA source file is provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <CUDA_SOURCE_FILE>"
    exit 1
fi

# Compiler
COMPILER=nvcc
# Flags for nvcc
COMPILER_FLAGS="-arch=sm_61 -o exe"


# Profiler
PROFILER=nvprof

# Requested command
COMMAND=$1

# CUDA source file
CUDA_FILE=$2


# Check if we used the keyword all, compile or profile
if [ $COMMAND == "all" ]; then
    # Compile and profile
    echo "Compiling and executing CUDA program $CUDA_FILE"
    $COMPILER $COMPILER_FLAGS $CUDA_FILE
    ./exe
    exit 0
elif [ $COMMAND == "compile" ]; then
    # Compile only
    echo "Compiling CUDA program $CUDA_FILE"
    $COMPILER $COMPILER_FLAGS $CUDA_FILE
    exit 0
elif [ $COMMAND == "profile" ]; then
    # Profile only
    echo "Compiling and profiling CUDA program $CUDA_FILE"
    $COMPILER $COMPILER_FLAGS $CUDA_FILE
    $PROFILER ./exe
    exit 0
else
    echo "Usage: $0 <all|compile|profile> <CUDA_SOURCE_FILE>"
    exit 1
fi




