#include "stdio.h"
#define N 20
#define M 4

__global__ void MyKernel(const double* __restrict__ Q, const double* __restrict__ A, const double* __restrict__ b, const bool Q_DIAG,//input
                        bool* __restrict__ feasible, double* __restrict__ fx_vals) 
{ 
 const unsigned long x = blockIdx.x * blockDim.x + threadIdx.x;

    bool x_bin[sizeof(int) * 8];// we might want to set a max N and max M and assign statically the memory as that value 
    
    //bool* x_bin = all_x_bin + x * N;
    #pragma unroll
    for(int i = 0; i < N; i++){
        x_bin[i] = (x >> i) & 1;
    }
    
    double Ax_b[32] = {0};
    //b_Type* Ax_b = all_Ax_b + x * M;


    bool is_feasible = true;
    //RISCRIVIAMO A*x - b facendo in modo che se x == 0 skippiamo i conti
    #pragma unroll
    for(int i = 0; i < N; i++){
        if(x_bin[i] == 0){
            continue;
        }
        for(int j = 0; j < M; j++){
            Ax_b[j] += A[j + i*M];
        }
    } 

    //check if x is feasible
    #pragma unroll
    for(int i = 0; i < M; i++){
        Ax_b[i] -= b[i];
        if(Ax_b[i] > 0){
            is_feasible = false;
        }
    }


    double fx = 0;
    feasible[x] = is_feasible;

    if(Q_DIAG){ //Q is encoded as an array with only the diagonal elements
        for(int i = 0; i < N; i++){
            fx += Q[i] * x_bin[i];
        }

    }else{
        if(is_feasible){       
            //FACCIAMO  x^T * Qx considerando la codifica particolare di Q
            for(int i = 0; i < N; i++){
                for(int j = i; j < N; j++){
                    fx += x_bin[i] * Q[i*N + j - i - i*(i-1)/2] * x_bin[j];
                }
            }
        }else{
            fx = DBL_MAX;
        }
    }
    fx_vals[x] = fx;

} 

void launchMyKernel(int *array, int arrayCount) 
{ 
  int blockSize;   // The launch configurator returned block size 
  int minGridSize; // The minimum grid size needed to achieve the 
                   // maximum occupancy for a full device launch 
  int gridSize;    // The actual grid size needed, based on input size 

  cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
                                      MyKernel, 0, 0); 
  printf("Min grid size: %d\n", minGridSize);
  printf("Block size: %d\n", blockSize);


  // Round up according to array size 
  gridSize = (arrayCount + blockSize - 1) / blockSize; 

  MyKernel<<< gridSize, blockSize >>>(array, arrayCount); 

  cudaDeviceSynchronize(); 

  // calculate theoretical occupancy
  int maxActiveBlocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, 
                                                 MyKernel, blockSize, 
                                                 0);

  printf("Max active blocks: %d\n", maxActiveBlocks);

  int device;
  cudaDeviceProp props;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);

  float occupancy = (maxActiveBlocks * blockSize / props.warpSize) / 
                    (float)(props.maxThreadsPerMultiProcessor / 
                            props.warpSize);

  printf("Launched blocks of size %d. Theoretical occupancy: %f\n", 
         blockSize, occupancy);
}

int main(){
    int arrayCount = 100000;
    int *array;
    cudaMallocManaged(&array, arrayCount * sizeof(int));
    for (int i = 0; i < arrayCount; i++) 
    { 
        array[i] = i; 
    } 
    launchMyKernel(array, arrayCount);
    for (int i = 0; i < arrayCount; i++) 
    { 
        //printf("%d\n", array[i]); 
    } 
    cudaFree(array);
    return 0;
}