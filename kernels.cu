#include <iostream>
#include <cuda_runtime.h>

typedef double Q_Type;
typedef double b_Type;
typedef double A_Type;
typedef double fx_Type;



__global__ void brute_force(const Q_Type* __restrict__ Q, const A_Type* __restrict__ A, const b_Type* __restrict__ b, const int N, const int M, 
                            bool* __restrict__ feasible, fx_Type* __restrict__ fx_vals) {
    
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    bool x_bin[N];

    for(int i = 0; i< N; i++){
        x_bin[i] = (x >> i) & 0b1;
    }

    
    double Ax_b[M];
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

    if(feasible[x] = is_feasible){
        /*//FACCIAMO PRIMA Q * x
        double Qx[N];
        for(int i = 0; i < N; i++){
            Qx[i] = 0;
        }
        for(int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                Qx[i] += Q[i*N + j] * x_bin[j];
            }
        }

        //FACCIAMO  x^T * Qx
        for(int i = 0; i < N; i++){
            fx_vals[x] += x_bin[i] * Qx[i];
        }*/

        //FACCIAMO  x^T * Qx considerando la codifica particolare di Q
        fx_Type fx = 0;
        for(int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                fx += x_bin[i] * Q[i*N + j] * x_bin[j];
            }
        }
    }
}


