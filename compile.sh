#!/bin/bash

# Check if CUDA source file is provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <CUDA_SOURCE_FILE>"
    exit 1
fi

# Compiler
COMPILER=nvcc
# Flags for nvcc
COMPILER_FLAGS="-arch=sm_61 -g -lineinfo --expt-relaxed-constexpr -o exe"


# Profiler
PROFILER=nvprof

# Profiler flags
PROFILER_TRACE="--print-gpu-trace --print-api-trace "
PROFILER_EVENTS="--events all "
PROFILER_METRICS="--metrics all "
# Exec flags
#-s strong verbose
# -f final stampa solo la media di tutto, senza ti mostra tutte le combinazioni di M e N
#PCRb prob,b con Q diagoanle identità
#PCRl prob,b,lower,upper con Q diagonale
#PCRq prob,b,lower,upper con Q triang sup

EXEC_FLAGS="-mN 20 -MN 20 -mM 10 -MM 10 -ml=0 -mu 0.1 -l -a 1000 -i 1 -r 0.1 -PCRl 0.3,2,1,2 -c -s"

# Requested command
COMMAND=$1



# ALL the extra flags passed in input
shift
EXTRA_FLAGS=$@


# CUDA source file
CUDA_FILE=" param_visualization_n_dim_par2.cu"


# Check if we used the keyword all, compile or profile
if [ $COMMAND == "all" ]; then
    # Compile and profile
    echo "Compiling and executing CUDA program $CUDA_FILE"
    $COMPILER $COMPILER_FLAGS $EXTRA_FLAGS $CUDA_FILE
    ./exe $EXEC_FLAGS
    exit 0
elif [ $COMMAND == "compile" ]; then
    # Compile only
    echo "Compiling CUDA program $CUDA_FILE"
    $COMPILER $COMPILER_FLAGS $EXTRA_FLAGS $CUDA_FILE
    exit 0
elif [ $COMMAND == "profile" ]; then
    # Profile only
    echo "Compiling and profiling CUDA program $CUDA_FILE"
    $COMPILER $COMPILER_FLAGS $CUDA_FILE
    $PROFILER $EXTRA_FLAGS ./exe $EXEC_FLAGS -f
    exit 0
elif [ $COMMAND == "profileTrace" ]; then
    # Profile only
    echo "Compiling and profiling CUDA program $CUDA_FILE"
    $COMPILER $COMPILER_FLAGS $CUDA_FILE
    $PROFILER $EXTRA_FLAGS $PROFILER_TRACE ./exe $EXEC_FLAGS -f
    exit 0
elif [ $COMMAND == "profileEvents" ]; then
    # Profile only
    echo "Compiling and profiling CUDA program $CUDA_FILE"
    $COMPILER $COMPILER_FLAGS $CUDA_FILE
    sudo $PROFILER $EXTRA_FLAGS $PROFILER_EVENTS ./exe $EXEC_FLAGS -f
    exit 0
elif [ $COMMAND == "profileMetrics" ]; then
    # Profile only
    echo "Compiling and profiling CUDA program $CUDA_FILE"
    $COMPILER $COMPILER_FLAGS $CUDA_FILE
    sudo $PROFILER $EXTRA_FLAGS $PROFILER_METRICS ./exe $EXEC_FLAGS -f
    exit 0
elif [ $COMMAND == "debug" ]; then
    # Run only
    echo "Debugging CUDA program $CUDA_FILE"
    $COMPILER $COMPILER_FLAGS $CUDA_FILE
    cuda-gdb --args ./exe $EXEC_FLAGS
    exit 0
elif [ $COMMAND == "run" ]; then
    # Run only
    clear
    echo "Running CUDA program $CUDA_FILE"
    ./exe $EXEC_FLAGS
    exit 0
else
    echo "Usage: $0 <all|compile|profile> <CUDA_SOURCE_FILE>"
    exit 1
fi




