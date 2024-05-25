#pragma once

#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>
#include <float.h>
#include <limits>

#include "types.h"


#define N_THREADS 1024
#define MAX_N_GPU sizeof(x_dec_Type) * 8
#define MAX_M_GPU 16

#define COARSENING 7


__constant__ A_Type A_const[MAX_M_GPU * MAX_N_GPU];
__constant__ Q_Type Q_const[MAX_N_GPU * (MAX_N_GPU + 1) / 2];
__constant__ b_Type b_const[MAX_M_GPU];

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


__global__ void brute_force( //input
                            const dim_Type N, const dim_Type M, const bool Q_DIAG, //consts
                            fx_Type* __restrict__ fx_vals, x_dec_Type* __restrict__ x_min) { //output
    
    const x_dec_Type stride = pow(2, COARSENING);
    const x_dec_Type x_start = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    //printf("blockIdx.x: %d\t blockDim.x: %d\t threadIdx.x: %d\t x_start: %d\n", blockIdx.x, blockDim.x, threadIdx.x, x_start);
    //printf("x_start: %d\n", x_start);

    fx_Type fx_min = std::numeric_limits<fx_Type>::max();
    x_dec_Type x_argmin = x_start;
    
    extern __shared__ char shared_mem[];
    b_Type* Ax_b_shared = (b_Type*) shared_mem;

    
    for(x_dec_Type x = x_start; x < x_start + stride; x++){

        //fill Ax_b_shared with zeros
        for(dim_Type i = 0; i < M; i++){
            Ax_b_shared[i * blockDim.x + threadIdx.x] = 0;
        }


        bool is_feasible = true;
        //RISCRIVIAMO A*x - b facendo in modo che se x == 0 skippiamo i conti
        #pragma unroll 
        for(dim_Type i = 0; i < N; i++){
            if(((x >> i) & 0b1) != 0){
                for(dim_Type j = 0; j < M; j++){
                    Ax_b_shared[j * blockDim.x + threadIdx.x] += A_const[j + i*M];
                }    
            }
        } 

        //check if x is feasible
        #pragma unroll
        for(dim_Type i = 0; i < M; i++){
            Ax_b_shared[i * blockDim.x + threadIdx.x] -= b_const[i];

            if(Ax_b_shared[i * blockDim.x + threadIdx.x] > 0){
                is_feasible = false;
            }
        }


        fx_Type fx = std::numeric_limits<fx_Type>::max();


        // POSSIAMO FERMARCI QUANDO X >> i == 0 PERCHÈ TANTO POI SONO TUTTI ZERI STESSA COSA CON X >> j == 0           <<<<<<<<<<<<<<<<<<
        if(is_feasible){
            fx = 0;
            if(Q_DIAG){ //Q is encoded as an array with only the diagonal elements
                for(dim_Type i = 0; i < N; i++){
                    if((x >> i) & 0b1)
                        fx += Q_const[i];
                }
            }else{

                int Q_idx = 0;
                //FACCIAMO  x^T * Qx considerando la codifica particolare di Q
                for(dim_Type i = 0; i < N; i++){

                    if((x >> i) & 0b1){
                        for(dim_Type j = i; j < N; j++){
                            if((x >> j) & 0b1)
                                fx +=  Q_const[Q_idx];
                            Q_idx++;
                        }
                    }else{
                        Q_idx += N - i;
                    }
                }
            }
        }

        if(fx < fx_min){
            fx_min = fx;
            x_argmin = x;
        }

    }

    fx_vals[blockIdx.x * blockDim.x + threadIdx.x] = fx_min;
    x_min[blockIdx.x * blockDim.x + threadIdx.x] = x_argmin;

}



__global__ void brute_force_AL(const dim_Type N, //input
                               fx_Type* __restrict__ fx_vals, x_dec_Type* __restrict__ x_min) { //output
    
    const x_dec_Type stride = pow(2, COARSENING);
    const x_dec_Type x_start = (blockIdx.x * blockDim.x + threadIdx.x) * stride;

    fx_Type fx_min = std::numeric_limits<fx_Type>::max();
    x_dec_Type x_argmin = 0;


    for(x_dec_Type x = x_start; x < x_start + stride; x++){
        fx_Type fx = 0;
        int Q_idx = 0;

        //FACCIAMO  x^T * Q' * x considerando la codifica particolare di Q
        for(dim_Type i = 0; i < N; i++){
            if((x >> i) & 0b1){
                for(dim_Type j = i; j < N; j++){
                    if((x >> j) & 0b1)
                        fx +=  Q_const[Q_idx];
                    Q_idx++;
                }
            }else{
                Q_idx += N - i;
            }
        }

       
        if(fx < fx_min){
            fx_min = fx;
            x_argmin = x;
        }

       
    }
    
    //printf("%ld AL\twriting index %d\n",x_start, blockIdx.x * blockDim.x + threadIdx.x);

    fx_vals[blockIdx.x * blockDim.x + threadIdx.x] = fx_min;
    x_min[blockIdx.x * blockDim.x + threadIdx.x] = x_argmin;

}



__global__ void reduce_argmin(fx_Type* __restrict__ input, x_dec_Type* __restrict__ x_input, fx_Type* __restrict__ min, x_dec_Type* __restrict__ x_min){

  	// Declare shared memory of N_THREADS elements
  	__shared__ fx_Type s_input[N_THREADS]; // Shared memory for the block
    __shared__ x_dec_Type s_x[N_THREADS];         // Shared memory for the block


  	// Position in the input array from which to start the reduction  
  	const unsigned int i = threadIdx.x;

  	// Offset the pointers to the correct block
  	input += blockDim.x * blockIdx.x;
    x_input += blockDim.x * blockIdx.x;

  	// perform first reduction step to copy the data from global memory to shared memory
  	
  	s_input[i] = input[i];
    s_x[i] = x_input[i];


    if(threadIdx.x + blockIdx.x * blockDim.x == 0){
        *min = std::numeric_limits<fx_Type>::max();                         
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

