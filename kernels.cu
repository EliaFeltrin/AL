#pragma once

#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>
#include <float.h>
#include "types.h"


#define N_THREADS 1024

__device__ volatile int sem = 0;

__device__ __forceinline__  void acquire_semaphore(volatile int *lock){
    while (atomicCAS((int *)lock, 0, 1) != 0);
}

__device__ __forceinline__ void release_semaphore(volatile int *lock){
    *lock = 0;
    __threadfence();
}

__device__ __forceinline__ void atomicMin(fx_Type * addr_min, x_dec_Type* addr_argmin, fx_Type value_min, x_dec_Type value_argmin) {

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


__global__ void brute_force(const Q_Type* __restrict__ Q, const A_Type* __restrict__ A, const b_Type* __restrict__ b, const dim_Type N, const dim_Type M, const bool Q_DIAG,//input
                            bool* __restrict__ feasible, fx_Type* __restrict__ fx_vals) { //output
    
    const unsigned long x = blockIdx.x * blockDim.x + threadIdx.x;

    bool x_bin[sizeof(x_dec_Type) * 8];// we might want to set a max N and max M and assign statically the memory as that value 
    
    //bool* x_bin = all_x_bin + x * N;
    for(dim_Type i = 0; i < N; i++){
        x_bin[i] = (x >> i) & 1;
    }
    
    double Ax_b[32];
    //b_Type* Ax_b = all_Ax_b + x * M;

    for(int i = 0; i < M; i++){
        Ax_b[i] = 0;
    }


    bool is_feasible = true;
    //RISCRIVIAMO A*x - b facendo in modo che se x == 0 skippiamo i conti
    for(dim_Type i = 0; i < N; i++){
        if(x_bin[i] == 0){
            continue;
        }
        for(dim_Type j = 0; j < M; j++){
            Ax_b[j] += A[j + i*M];
        }
    } 

    //check if x is feasible
    for(dim_Type i = 0; i < M; i++){
        Ax_b[i] -= b[i];
        if(Ax_b[i] > 0){
            is_feasible = false;
        }
    }


    fx_Type fx = 0;
    feasible[x] = is_feasible;

    if(Q_DIAG){ //Q is encoded as an array with only the diagonal elements
        for(dim_Type i = 0; i < N; i++){
            fx += Q[i] * x_bin[i];
        }

    }else{
        if(is_feasible){       
            //FACCIAMO  x^T * Qx considerando la codifica particolare di Q
            for(dim_Type i = 0; i < N; i++){
                for(dim_Type j = i; j < N; j++){
                    fx += x_bin[i] * Q[i*N + j - i - i*(i-1)/2] * x_bin[j];
                }
            }
        }else{
            fx = DBL_MAX;
        }
    }
    fx_vals[x] = fx;

}


__global__ void reduce_argmin_feasible(fx_Type* __restrict__ input, bool* __restrict__ feasible, fx_Type* __restrict__ min, x_dec_Type* __restrict__ x_min){

  	// Declare shared memory of N_THREADS elements
  	__shared__ fx_Type s_input[N_THREADS]; // Shared memory for the block
  	__shared__ bool s_feasible[N_THREADS]; // Shared memory for the block
    __shared__ x_dec_Type s_x[N_THREADS];         // Shared memory for the block


  	// Position in the input array from which to start the reduction  
  	const unsigned int i = threadIdx.x;

  	// Offset the pointers to the correct block
  	input += blockDim.x * blockIdx.x;
  	feasible += blockDim.x * blockIdx.x;

  	// perform first reduction step to copy the data from global memory to shared memory
  	
  	s_input[i] = input[i];
    s_feasible[i] = feasible[i];
    s_x[i] = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(threadIdx.x + blockIdx.x * blockDim.x == 0){
        *min = DBL_MAX;
    }

  	// Perform the reduction for each block indipendently
  	for (unsigned int stride = blockDim.x/2; i < stride; stride /= 2) {
  	    
		__syncthreads(); //needs to be moved up since the first iteration is outside

		if( !s_feasible[i] || (s_feasible[i + stride] && s_input[i] > s_input[i + stride])){
  	    	s_input[i] = s_input[i + stride];
            s_x[i] = s_x[i + stride];
            s_feasible[i] = s_feasible[i + stride];
        }

  	}

  	// Write result for this block to global memory
  	if (i == 0) {
        //printf("Block min: %f, x: %d\n", s_input[0], s_x[0]);
        atomicMin(min, x_min, s_input[0], s_x[0]);
  	}

	//retrun di minimum e x corrispondente
    
}


__global__ void brute_force_AL(const Q_Type* __restrict__ Q_prime, const dim_Type N, //input
                               fx_Type* __restrict__ fx_vals) { //output
    
    const unsigned long x = blockIdx.x * blockDim.x + threadIdx.x;

    bool x_bin[sizeof(x_dec_Type) * 8];// we might want to set a max N and max M and assign statically the memory as that value 
    //bool* x_bin = all_x_bin + x * N;
    
    for(dim_Type i = 0; i < N; i++){
        x_bin[i] = (x >> i) & 1;
    }
    

    fx_Type fx = 0;

    //FACCIAMO  x^T * Q' * x considerando la codifica particolare di Q
    for(dim_Type i = 0; i < N; i++){
        for(dim_Type j = i; j < N; j++){
            fx += x_bin[i] * Q_prime[(int)(i * (N-0.5f) - i*i/2.0f+j)] * x_bin[j];
        }
    }

    
    fx_vals[x] = fx;

}



__global__ void reduce_argmin(fx_Type* __restrict__ input, fx_Type* __restrict__ min, x_dec_Type* __restrict__ x_min){

  	// Declare shared memory of N_THREADS elements
  	__shared__ fx_Type s_input[N_THREADS]; // Shared memory for the block
    __shared__ x_dec_Type s_x[N_THREADS];         // Shared memory for the block


  	// Position in the input array from which to start the reduction  
  	const unsigned int i = threadIdx.x;

  	// Offset the pointers to the correct block
  	input += blockDim.x * blockIdx.x;

  	// perform first reduction step to copy the data from global memory to shared memory
  	
  	s_input[i] = input[i];
    s_x[i] = threadIdx.x + blockIdx.x * blockDim.x;


    if(threadIdx.x + blockIdx.x * blockDim.x == 0){
        *min = DBL_MAX;
    }

  	// Perform the reduction for each block indipendently
  	for (unsigned int stride = blockDim.x/2; i < stride; stride /= 2) {
  	    
		__syncthreads(); //needs to be moved up since the first iteration is outside

		if(s_input[i] > s_input[i + stride]){
  	    	s_input[i] = s_input[i + stride];
            s_x[i] = s_x[i + stride];
        }

  	}

  	// Write result for this block to global memory
  	if (i == 0) {
        //printf("Block min: %f, x: %d\n", s_input[0], s_x[0]);
        atomicMin(min, x_min, s_input[0], s_x[0]);
  	}

	//retrun di minimum e x corrispondente
    
}

