#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>
#include <float.h>
#include "types.h"

//typedef double Q_Type;
//typedef double b_Type;
//typedef double A_Type;
//typedef double fx_Type;
//typedef unsigned char dim_Type;
//typedef float lambda_Type;

#define N_THREADS 1024

__device__ volatile int sem = 0;

__device__ __forceinline__  void acquire_semaphore(volatile int *lock){
    while (atomicCAS((int *)lock, 0, 1) != 0);
}

__device__ __forceinline__ void release_semaphore(volatile int *lock){
    *lock = 0;
    __threadfence();
}

__device__ __forceinline__ void atomicMin(double * addr_min, int* addr_argmin, double value_min, int value_argmin) {

    acquire_semaphore(&sem);
    __syncthreads();
    //begin critical section
    
    if(*addr_min > value_min){
        *addr_min = value_min;
        *addr_argmin = value_argmin;
    }
    //end critical section
    
    __threadfence(); // not strictly necessary for the lock, but to make any global updates in the critical section visible to other threads in the grid
    __syncthreads();
    release_semaphore(&sem);
    __syncthreads();   

}

__device__ __forceinline__ void atomicMax(double * addr_max, double value_max) {

    
}



__global__ void brute_force(const Q_Type* __restrict__ Q, const A_Type* __restrict__ A, const b_Type* __restrict__ b, int N, int M,
                            bool* __restrict__ feasible, fx_Type* __restrict__ fx_vals) {
    
    const int x = blockIdx.x * blockDim.x + threadIdx.x;

    bool* x_bin = (bool*)malloc(N * sizeof(bool));

    for(int i = 0; i< N; i++){
        x_bin[i] = (x >> i) & 0b1;
    }

    
    double* Ax_b = (double*)malloc(M * sizeof(double));

    for(int i = 0; i < M; i++){
        Ax_b[i] = 0;
    }


    bool is_feasible = true;
    //FACCIAMO A * x - b
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            Ax_b[i] += A[i*N + j] * x_bin[j];
        }
        Ax_b[i] -= b[i];
        
        if(Ax_b[i] > 0){
            is_feasible = false;
        }
    }

    fx_Type fx = 0;
    if(feasible[x] = is_feasible){       
        //FACCIAMO  x^T * Qx considerando la codifica particolare di Q
        for(int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                fx += x_bin[i] * Q[i*N + j] * x_bin[j];
            }
        }
    }
    fx_vals[x] = fx;

    free(x_bin);
}


__global__ void reduce_argmin_feasible(fx_Type* __restrict__ input, bool* __restrict__ feasible, fx_Type* __restrict__ min, int* __restrict__ x_min){

  	// Declare shared memory of N_THREADS elements
  	__shared__ double s_input[N_THREADS]; // Shared memory for the block
  	//__shared__ bool s_feasible[N_THREADS]; // Shared memory for the block
    __shared__ int s_x[N_THREADS]; // Shared memory for the block
    

  	// Position in the input array from which to start the reduction  
  	const unsigned int i = threadIdx.x;

  	// Offset the pointers to the correct block
  	input += blockDim.x * blockIdx.x * 2;
  	feasible += blockDim.x * blockIdx.x * 2;

  	// perform first reduction step to copy the data from global memory to shared memory
  	
  	s_input[i] = DBL_MAX;
	s_x[i] = -1;


  	// Perform the reduction for each block indipendently
  	for (unsigned int stride = blockDim.x; i < stride; stride /= 2) {
  	    
		__syncthreads(); //needs to be moved up since the first iteration is outside
		
		if(feasible[i] && s_input[i] > s_input[i + stride]){
  	    	s_input[i] = s_input[i + stride];
            s_x[i] = i + stride;
        }
  	}

  	// Write result for this block to global memory
  	if (i == 0) {
  		atomicMin(min, x_min, s_input[0], s_x[0]);
  	}

	//retrun di minimum e x corrispondente
    
}

__global__ void reduce_max_feasible(fx_Type* __restrict__ input, bool* __restrict__ feasible, fx_Type* __restrict__ max){

  	// Declare shared memory of N_THREADS elements
  	__shared__ double s_input[N_THREADS]; // Shared memory for the block
  	//__shared__ bool s_feasible[N_THREADS]; // Shared memory for the block
    __shared__ int s_x[N_THREADS]; // Shared memory for the block
    

  	// Position in the input array from which to start the reduction  
  	const unsigned int i = threadIdx.x;

  	// Offset the pointers to the correct block
  	input += blockDim.x * blockIdx.x * 2;
  	feasible += blockDim.x * blockIdx.x * 2;

  	// perform first reduction step to copy the data from global memory to shared memory
  	
  	s_input[i] = DBL_MAX;
	s_x[i] = -1;


  	// Perform the reduction for each block indipendently
  	for (unsigned int stride = blockDim.x; i < stride; stride /= 2) {
  	    
		__syncthreads(); //needs to be moved up since the first iteration is outside
		
		if(feasible[i] && s_input[i] < s_input[i + stride]){
  	    	s_input[i] = s_input[i + stride];
            s_x[i] = i + stride;
        }
  	}

  	// Write result for this block to global memory
  	if (i == 0) {
  		atomicMax(max, s_input[0]);
  	}

	//retrun di minimum e x corrispondente
    
}



