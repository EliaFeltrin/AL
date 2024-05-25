#!/bin/bash



# Compiler
COMPILER=nvcc
# Flags for nvcc
COMPILER_FLAGS="-arch=sm_61 -g -lineinfo --expt-relaxed-constexpr -o exe"

CUDA_FILE=" param_visualization_n_dim_par2.cu"

# Compile the file
$COMPILER $COMPILER_FLAGS $CUDA_FILE

K="4"



PROFILER=nvprof

# Cycle fromm N = 4 to N = 25
for N in {4..22}; do
    PROFILER_FLAGS=" --log-file ./coarse_test/${N}_${K}.csv --csv "
    EXEC_FLAGS="-mN ${N} -MN ${N} -mM 3 -MM 3 -ml=0 -mu 0.1 -l -a 1000 -i 1000 -r 0.1 -PCRq 0.3,2,1,2 -c -s -f"
    
    
    $PROFILER $PROFILER_FLAGS ./exe $EXEC_FLAGS
    
done






