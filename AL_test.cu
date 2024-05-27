#pragma once

/*--------------------------------------- INCLUDES ------------------------------------------------ */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <time.h>
#include <chrono>
#include <ctime>
#include <sstream>
#include <cstdlib>
#include <unistd.h>
#include <float.h>
#include <functional>
#include <cstring>
#include <cstdarg>
#include <random>
#include <unordered_set>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <math.h>

#include "types.h"
#include "kernels.cu"

#define DEBUG
 
#ifndef DEBUG
    #define RAND_GEN_INIT std::random_device rd; std::mt19937 g(rd());
#else
    unsigned long seed = 42;
    #define RAND_GEN_INIT std::mt19937 g(seed++);
#endif

template <typename T>
constexpr T max_val(){
    return std::numeric_limits<T>::max();
}


/*--------------------------------------- MACROS -------------------------------------------------- */

#define call)                                                                         \
	{                                                                                       \
		const cudaError_t err = call;                                                       \
		if (err != cudaSuccess) {                                                           \
			printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);   \
			exit(EXIT_FAILURE);                                                             \
		}                                                                                   \
	}

#define CHECK_KERNELCALL()                                                                  \
	{                                                                                       \
		const cudaError_t err = cudaGetLastError();                                         \
		if (err != cudaSuccess) {                                                           \
			printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);   \
			exit(EXIT_FAILURE);                                                             \
		}                                                                                   \
	} 


/*--------------------------------------- GLOBAL VARIABLES ---------------------------------------- */

extern double MAX_MU;
extern double MAX_LAMBDA;

extern char name_suffix[20];  
extern char results_path[100];

extern bool Q_DIAG;
extern bool Q_ID;
extern bool PCR_PROBLEM;


/*--------------------------------------- ENUMS --------------------------------------------------- */ 
enum stop_conditions_names {max_Al_attempts, max_mu, max_lambda, stop_conditions_end};
enum fill_distributions {uniform, MMF, PCR, PCRL, fill_distributions_end};


/*--------------------------------------- STRUCTS ------------------------------------------------- */

struct test_results{
    dim_Type N;
    dim_Type M;
    double correct_ratio;
    double unfinished_ratio;
    double normalized_error_mean;
    float mean_al_attempts_on_correct_solutions;
    float mean_al_attempts_on_wrong_solutions;
    float mean_al_attempts_on_unfinished_solutions;
    lambda_Type mean_lambda_on_correct_solutions;
    lambda_Type mean_lambda_on_unfinished_solutions;
    lambda_Type mean_lambda_on_wrong_solutions;
    mu_Type mean_mu_on_correct_solutions;
    mu_Type mean_mu_on_unfinished_solutions;
    mu_Type mean_mu_on_wrong_solutions;
    lambda_Type lambda_min_on_correct_solutions;
    lambda_Type lambda_min_on_unfinished_solutions;
    lambda_Type lambda_min_on_wrong_solutions;
    lambda_Type lambda_max_on_correct_solutions;
    lambda_Type lambda_max_on_unfinished_solutions;
    lambda_Type lambda_max_on_wrong_solutions;
    double duration;
};


/*--------------------------------------- PARTIAL RESULT COMPUTING FUNCTIONS ---------------------- */

int test_at_dimension_coarsening(   const unsigned int K,
                                    dim_Type N, dim_Type M, int MAXITER, int N_AL_ATTEMPTS, mu_Type initial_mu, lambda_Type initial_lambda,  mu_Type rho, 
                                    void (*fill_Q)(Q_Type *Q, const dim_Type N, const Q_Type lowerbound_or_unused, const Q_Type upperbound_or_unused), Q_Type lb_Q, Q_Type ub_Q, 
                                    void (*fill_A)(A_Type* A, const dim_Type M, const dim_Type N, const float one_probability_or_unused, const b_Type b_or_unused), float one_prob, 
                                    void (*fill_b)(b_Type* b, const dim_Type M, const b_Type b_val_or_unused), b_Type b_val, 
                                    std::function<bool(const int i, const int N_AL_ATTEMPTS, const dim_Type N, const dim_Type M, const lambda_Type* __restrict__ lambda, const mu_Type mu, const b_Type* __restrict__ c)> al_end_condition, 
                                    mu_Type (*update_mu)(const mu_Type mu, const mu_Type rho), 
                                    test_results* results, bool verbose, bool strong_verbose);

Q_Type compute_xQx(const Q_Type* __restrict__ Q, const bool* __restrict__ x, dim_Type N);

Q_Type compute_max(const Q_Type* __restrict__ Q, dim_Type N);


void compute_Q_plus_AT_A_upper_triangular_lin(const Q_Type* __restrict__ Q, A_Type* __restrict__ A, A_Type* __restrict__ Q_plus_AT_A, const dim_Type M, const dim_Type N);


/*--------------------------------------- FILL FUNCTIONS ------------------------------------------ */

void fill_Q_id_lin(Q_Type*  Q, const dim_Type N, const Q_Type not_used_1, const Q_Type not_used_2);
void fill_Q_diag_lin(Q_Type*  Q, const dim_Type N, const Q_Type lowerbound, const Q_Type upperbund);
void fill_Q_upper_trianular_lin(Q_Type *Q, const dim_Type N, const Q_Type lowerbound, const Q_Type upperbound);
void fill_Q_manual_lin(Q_Type*  Q, const dim_Type N, const Q_Type unused_1, const Q_Type unused_2);

void fill_A_neg_binary_lin(A_Type*  A, const dim_Type M, const dim_Type N, const float one_probability, const b_Type b);
void fill_A_manual_lin(A_Type*  A, const dim_Type M, const dim_Type N, const float unused_1, const b_Type unused_2);

void fill_b_vector_lin(b_Type* b, const dim_Type M, const b_Type b_val);
void fill_b_manual_lin(b_Type* b, const dim_Type M, const b_Type unused);

void fill_lambda_lin(lambda_Type* lambda, const dim_Type M, lambda_Type initial_lambda);
void fill_lambda_lin(lambda_Type* lambda, const dim_Type M, lambda_Type initial_lambda, lambda_Type noise_amplitude);


/*--------------------------------------- INDEX CONVERTION FUNCTIONS ------------------------------ */

inline unsigned int triang_index(dim_Type i, dim_Type j, dim_Type N){
    return (unsigned int)(i * (N - 0.5f) - i * i/2.0f + j);
}

/*--------------------------------------- HALT CONDITION FUNCTIONS -------------------------------- */

inline bool max_Al_attempts_condition(const int i, const int N_AL_ATTEMPTS, const dim_Type N, const dim_Type M, const lambda_Type* __restrict__ lambda, const mu_Type mu, const b_Type* __restrict__ c){
    return i < N_AL_ATTEMPTS;
}

inline bool max_mu_condition(const int i, const int N_AL_ATTEMPTS, const dim_Type N, const dim_Type M, const lambda_Type* __restrict__ lambda, const mu_Type mu, const b_Type* __restrict__ c){
    return mu < MAX_MU;
}

inline bool max_lambda_condition(const int i, const int N_AL_ATTEMPTS, const dim_Type N, const dim_Type M, const lambda_Type* __restrict__ lambda, const mu_Type mu, const b_Type* __restrict__ c){
    for(int i = 0; i < M; i++){
        if(lambda[i] >= MAX_LAMBDA){
            return false;
        }
    }
    return true;
}


/*--------------------------------------- MU UPDATING FUNCTIONS ----------------------------------- */

inline mu_Type update_mu_exp(const mu_Type mu, const mu_Type rho){
    return mu * rho;
}

inline mu_Type update_mu_lin(const mu_Type mu, const mu_Type rho){
    return mu + rho;
}


/*--------------------------------------- PRINTING FUNCTIONS -------------------------------------- */

void print_Q(const Q_Type* Q, const dim_Type N);
void print_A(const A_Type* A, const dim_Type M, const dim_Type N);
void print_b(const b_Type* b, const dim_Type M);


/*--------------------------------------- SAVE ON FILE FUNCTIONS ---------------------------------- */

void finalize(test_results mean_results);
void finalize(std::vector<test_results> results);
void print_file_stdout(FILE *file, const char *format, ...);


/* #############################################################################################################################################*/ 


//NB: viene memorizzata solo la diagonale, pertanto Q è di lunghezza N
void fill_Q_diag_lin(Q_Type* Q, const dim_Type N, const Q_Type lowerbound, const Q_Type upperbund){
    //std::random_device rd;
    //std::mt19937 g(rd());
    RAND_GEN_INIT

    for(dim_Type i = 0; i < N; i++){
        Q[i] = lowerbound + (upperbund-lowerbound)*((Q_Type)g()/g.max());
    }
}

//NB: viene memorizzata solo la diagonale, pertanto Q è di lunghezza N
void fill_Q_id_lin(Q_Type* Q, const dim_Type N, const Q_Type not_used_1, const Q_Type not_used_2){
    for(dim_Type i = 0; i < N; i++){
            Q[i] = 1;
    }
    printf("WARNING: you're using fill_Q_id_lin, which is quite useless since Q is the identity matrix\n");
}

//NB: non vengono memorizzati gli zeri della matrice triangolare inferiore
void fill_Q_upper_trianular_lin(Q_Type *Q, const dim_Type N, const Q_Type lowerbound, const Q_Type upperbound){
    const unsigned int Q_len = N*(N+1)/2;

    //std::random_device rd;
    //std::mt19937 g(rd());
    RAND_GEN_INIT


    for(unsigned int i = 0; i < Q_len; i++){
        Q[i] = lowerbound + (upperbound-lowerbound)*((Q_Type)g()/g.max());
    }
}

void fill_Q_manual_lin(Q_Type*  Q, const dim_Type N, const Q_Type unused_1, const Q_Type unused_2){
    Q_Type useless_val;

    printf("ENTER Q MATRIX\n");

    unsigned int q_counter = 0;
    if(!Q_DIAG){
        for(dim_Type i = 0; i < N; i++){
            for(dim_Type j = 0; j < N; j++){
                if(j >= i){
                    scanf("%f", &Q[q_counter++]);
                } else {
                    scanf("%f", &useless_val);
                }
            }
        }
    } else {
        for(dim_Type i = 0; i < N; i++){
            for(dim_Type j = 0; j < N; j++){
                if(j == i){
                    scanf("%f", &Q[i]);
                } else {
                    scanf("%f", &useless_val);
                }
            }
        }
    }

    print_Q(Q, N);

}

void fill_A_neg_binary_lin(A_Type*  A, const dim_Type M, const dim_Type N, const float one_probability, const b_Type b_){
    b_Type b = -b_;

    int Mxb = (M*b);
    
    for(int i = 0; i < Mxb; i++){
        A[i] = (A_Type)-1;
    }

    const unsigned int n_missing_ones = (unsigned int)(M * N * one_probability - M * b);
    

    const unsigned int aux_vec_len = M * (N - b); 
    std::vector<A_Type> aux_vec(aux_vec_len, (A_Type)0);
    
    for(int i = 0; i < n_missing_ones; i++){
        aux_vec[i] = -1;
    }
    
    RAND_GEN_INIT
    //std::random_device rd;
    //std::mt19937 g(rd());
    

    std::shuffle(aux_vec.begin(), aux_vec.end(), g);


    unsigned int c = 0;
    for(dim_Type i = 0; i < M; i++){
        for(dim_Type j = b; j < N; j++){
            A[i+j*M] = aux_vec[c++];
        }
    }

    //shuffle content of each row
    for(dim_Type i = 0; i < M; i++){
        for(dim_Type j = 0; j < N; j++){
            dim_Type rand_idx = g() % N;

            //switch A[i][j] with A[i][rand_idx]
            A_Type temp = A[i+j*M];
            A[i+j*M] = A[i+rand_idx*M];
            A[i+rand_idx*M] = temp;
        }
    }

}

void fill_A_manual_lin(A_Type*  A, const dim_Type M, const dim_Type N, const float unused_1, const b_Type unused_2){

    printf("ENTER A MATRIX\n");

    for(dim_Type i = 0; i < M; i++){
        for(dim_Type j = 0; j < N; j++){
            scanf("%d", &A[j*M + i]);
        }
    }

    print_A(A, M, N);
}

void fill_b_vector_lin(b_Type* b, const dim_Type M, const b_Type b_val){
    for(dim_Type i = 0; i < M; i++){
        b[i] = b_val;
    }
}

void fill_b_manual_lin(b_Type* b, const dim_Type M, const b_Type unused){
    printf("ENTER B VECTOR\n");
    
    for(dim_Type i = 0; i < M; i++){
        scanf("%d", &b[i]);
    }

    print_b(b, M);

}

void fill_lambda_lin(lambda_Type* lambda, const dim_Type M, lambda_Type initial_lambda){
    for(dim_Type i = 0; i < M; i++){
        lambda[i] = initial_lambda;
    }
}

void fill_lambda_lin(lambda_Type* lambda, const dim_Type M, lambda_Type initial_lambda, lambda_Type noise_amplitude){
    const lambda_Type RAND_MAX_ = (lambda_Type)RAND_MAX;
    for(dim_Type i = 0; i < M; i++){
        lambda[i] = initial_lambda + noise_amplitude * (lambda_Type)rand() / RAND_MAX_;
    }
}


void compute_Q_plus_AT_A_upper_triangular_lin(const Q_Type* __restrict__ Q, A_Type* __restrict__ A, Q_Type* __restrict__ Q_plus_AT_A, const dim_Type M, const dim_Type N){
    if(!Q_DIAG){
        for(dim_Type i = 0; i < N; i++){
            for(dim_Type j = i; j < N; j++){
                unsigned int triang_idx = triang_index(i,j,N);
                Q_plus_AT_A[triang_idx] = 0;
                for(dim_Type k = 0; k < M; k++){
                    Q_plus_AT_A[triang_idx] += A[k+i*M] * A[k+j*M];
                }
                if(i != j){
                    Q_plus_AT_A[triang_idx] *= 2;
                }
                Q_plus_AT_A[triang_idx] += Q[triang_idx]; 
            }
        }
    } else if(!Q_ID){
        for(dim_Type i = 0; i < N; i++){
            for(dim_Type j = i; j < N; j++){
                unsigned int triang_idx = triang_index(i,j,N);
                Q_plus_AT_A[triang_idx] = 0;
                for(dim_Type k = 0; k < M; k++){
                    Q_plus_AT_A[triang_idx] += A[k+i*M] * A[k+j*M];
                }
                if(i != j){
                    Q_plus_AT_A[triang_idx] *= 2;
                } 
            }
            Q_plus_AT_A[triang_index(i,i,N)] += Q[i];
        }
    } else {
       for(dim_Type i = 0; i < N; i++){
            for(dim_Type j = i; j < N; j++){
                unsigned int triang_idx = triang_index(i,j,N);
                Q_plus_AT_A[triang_idx] = 0;
                for(dim_Type k = 0; k < M; k++){
                    Q_plus_AT_A[triang_idx] += A[k+i*M] * A[k+j*M];
                }
                if(i != j){
                    Q_plus_AT_A[triang_idx] *= 2;
                } 
            }
            Q_plus_AT_A[triang_index(i,i,N)] += 1;
        }
    }
}

void print_Q(const Q_Type* Q, const dim_Type N){
    printf("Q =\n");
    if(Q_ID){
        for(dim_Type i = 0; i < N; i++){
            for(dim_Type j = 0; j < N; j++){
                if(i != j)
                    printf("0 ");
                else 
                    printf("%.0f ", Q[i]);
            }
            printf("\n");
        }
        printf("\n");
    } else if(Q_DIAG){
        for(dim_Type i = 0; i < N; i++){
            for(dim_Type j = 0; j < N; j++){
                if(i != j)
                    printf("0  ");
                else 
                    printf("%.1f ", Q[i]);
            }
            printf("\n");
        }
        printf("\n");
    } else {
        for(dim_Type i = 0; i < N; i++){
            for(dim_Type j = 0; j < N; j++){
                if(j < i)
                    printf("%4.0f ",0.f);
                else
                    printf("%1.2f ", Q[triang_index(i,j,N)]);
            }
            printf("\n");
        }
    }
}

void print_A(const A_Type* A, const dim_Type M, const dim_Type N){
    printf("A =\n");
    for(dim_Type i = 0; i < M; i++){
        for(dim_Type j = 0; j < N; j++){
            //stampa A con il cout facendo in modo che i valori siano allineati (-1 e 0 non sono allineati)
            std::cout << std::setw(2) << A[i+j*M] << " ";
        }
        printf("\n");
    }
}

void print_b(const b_Type* b, const dim_Type M){
    printf("b^T = [");
    for(dim_Type i = 0; i < M; i++){
        std::cout << std::setw(2) << b[i] << " "; 
    }
    printf("]\n");
}


Q_Type compute_xQx(const Q_Type* __restrict__ Q, const bool* __restrict__ x, dim_Type N){
    Q_Type res = 0;
    if(Q_ID){
        for(dim_Type i = 0; i < N; i++){
            res += x[i];
        }
    } else if(Q_DIAG){
        for(dim_Type i = 0; i < N; i++){
            res += x[i] * Q[i];
        }
    } else {
        for(dim_Type i = 0; i < N; i++){
            for(dim_Type j = i; j < N; j++){
                res += x[i] * x[j] * Q[triang_index(i,j,N)];
            }
        }
    }
    return res;
};

inline Q_Type compute_max(const Q_Type* __restrict__ Q, dim_Type N){
    Q_Type res = 0;
    if(Q_ID){
        return N;
    } else if(Q_DIAG){
        for(dim_Type i = 0; i < N; i++){
            res += Q[i];
        }
    } else {
        unsigned int Q_len = N*(N+1)/2;
        for(unsigned int i = 0; i < Q_len; i++){
            res += Q[i];
        }
    }
    return res;

}


int test_at_dimension_coarsening(   const unsigned int COARSENING,
                                    dim_Type N, dim_Type M, int MAXITER, int N_AL_ATTEMPTS, mu_Type initial_mu, lambda_Type initial_lambda,  mu_Type rho, 
                                    void (*fill_Q)(Q_Type *Q, const dim_Type N, const Q_Type lowerbound_or_unused, const Q_Type upperbound_or_unused), Q_Type lb_Q, Q_Type ub_Q, 
                                    void (*fill_A)(A_Type* A, const dim_Type M, const dim_Type N, const float one_probability_or_unused, const b_Type b_or_unused), float one_prob, 
                                    void (*fill_b)(b_Type* b, const dim_Type M, const b_Type b_val_or_unused), b_Type b_val, 
                                    std::function<bool(const int i, const int N_AL_ATTEMPTS, const dim_Type N, const dim_Type M, const lambda_Type* __restrict__ lambda, const mu_Type mu, const b_Type* __restrict__ c)> al_end_condition, 
                                    mu_Type (*update_mu)(const mu_Type mu, const mu_Type rho), 
                                    test_results* results, bool verbose, bool strong_verbose)
{

    
    
    auto start = std::chrono::high_resolution_clock::now();
    const int progressBarWidth = 100;
    srand(time(0));

    // Allocate
    const unsigned int Q_len = Q_DIAG ? N : N*(N+1)/2;
    const unsigned int A_len = M * N;

    Q_Type* Q = new Q_Type[Q_len];
    A_Type* A = new A_Type[M*N];
    b_Type* b = new b_Type[M];
    lambda_Type* lambda = new lambda_Type[M];
    lambda_Type* old_lambda = new lambda_Type[M];
    bool* expected_min_x = new bool[N];
    bool* min_x = new bool[N];
    b_Type* c = new b_Type[M];


    fx_Type true_max_val, true_min_val, al_min_val;

    mu_Type mu;
    mu_Type old_mu;
    lambda_Type mean_lambda_on_correct_solutions       = 0,    mean_mu_on_correct_solutions      = 0;
    lambda_Type mean_lambda_on_unfinished_solutions    = 0,    mean_mu_on_unfinished_solutions   = 0;
    lambda_Type mean_lambda_on_wrong_solutions         = 0,    mean_mu_on_wrong_solutions        = 0;
    lambda_Type lambda_min_on_correct_solutions        = max_val<lambda_Type>(),  lambda_max_on_correct_solutions       = -max_val<lambda_Type>();     
    lambda_Type lambda_min_on_unfinished_solutions     = max_val<lambda_Type>(),  lambda_max_on_unfinished_solutions    = -max_val<lambda_Type>(); 
    lambda_Type lambda_min_on_wrong_solutions          = max_val<lambda_Type>(),  lambda_max_on_wrong_solutions         = -max_val<lambda_Type>(); 
    double mean_al_attempts_on_correct_solutions     = 0;
    double mean_al_attempts_on_wrong_solutions       = 0;
    double mean_al_attempts_on_unfinished_solutions  = 0;   

    bool correct, unfinished, wrong;

    int correct_counter = 0;
    int unfinished_counter = 0;
    double normalized_error_mean = 0;


    // Creating two cuda streams 
    cudaStream_t stream_BF, stream_BF_AL;
    cudaStreamCreateWithPriority(&stream_BF, cudaStreamNonBlocking, 1);
    cudaStreamCreateWithPriority(&stream_BF_AL, cudaStreamNonBlocking, 0);

    // Allocate GPU memory
    fx_Type*    fx_gpu_BF; // for brute_force
    x_dec_Type* xs_min_gpu_BF; // for brute_force

    fx_Type*    fx_gpu_AL; // for AL
    x_dec_Type* xs_min_gpu_AL; // for AL
    
    
    cudaMallocAsync(&fx_gpu_BF, pow(2,N - COARSENING) * sizeof(fx_Type), stream_BF);
    cudaMallocAsync(&xs_min_gpu_BF, pow(2,N - COARSENING) * sizeof(x_dec_Type), stream_BF);

    cudaMallocAsync(&fx_gpu_AL, pow(2,N - COARSENING) * sizeof(fx_Type), stream_BF_AL);
    cudaMallocAsync(&xs_min_gpu_AL, pow(2,N - COARSENING) * sizeof(x_dec_Type), stream_BF_AL);



    for(int iter = 0; iter < MAXITER; iter++) {
        correct = unfinished = wrong = 0;

        //fill and transfer Q
        fill_Q(Q, N, lb_Q, ub_Q);
        cudaMemcpyToSymbolAsync(Q_const, Q, Q_len * sizeof(Q_Type), 0, cudaMemcpyHostToDevice, stream_BF);
        //fill and transfer A
        fill_A(A, M, N, one_prob, b_val);
        cudaMemcpyToSymbolAsync(A_const, A, A_len * sizeof(A_Type), 0, cudaMemcpyHostToDevice, stream_BF);
        //fill and transfer b
        fill_b(b, M, b_val);
        cudaMemcpyToSymbolAsync(b_const, b, M * sizeof(b_Type), 0, cudaMemcpyHostToDevice, stream_BF);

        //LANCIO BRUTE FORCE
        int n_threads_bf = min(N_THREADS_BF, (int)pow(2, N - COARSENING));
        dim3 threads_per_block_bf(n_threads_bf);
	    dim3 blocks_per_grid_bf(pow(2, N - COARSENING) / n_threads_bf);   
        const int shared_mem_size = n_threads_bf * M * sizeof(b_Type);
        
        //ADD Q_DIAG e Q_ID
        //brute_force<<<blocks_per_grid, threads_per_block>>>(Q_gpu, A_gpu, b_gpu, N, M, Q_DIAG, x_bin_buffer_gpu, Ax_b_buffer_gpu, feasible_gpu, fx_gpu);
        brute_force_coarsening<<<blocks_per_grid_bf, threads_per_block_bf, shared_mem_size, stream_BF>>>(N, M, COARSENING, Q_DIAG, fx_gpu_BF, xs_min_gpu_BF);
	    CHECK_KERNELCALL();
	    //cudaDeviceSynchronize()); ///MAYBE TO REMOVE
        int input_size = (int)pow(2, N - COARSENING);
        while(input_size > 1){
            int n_threads_am = min(N_THREADS_ARGMIN, input_size);
            dim3 threads_per_block_am(n_threads_am);
	        dim3 blocks_per_grid_am(input_size / n_threads_am);   

            reduce_argmin<<<blocks_per_grid_am, threads_per_block_am, 0, stream_BF>>>(fx_gpu_BF, xs_min_gpu_BF);
            CHECK_KERNELCALL();
	        //cudaDeviceSynchronize()); ///MAYBE TO REMOVE
            input_size >>= (int)log2(N_THREADS_ARGMIN);
        }

        //COPY BACK RESULTS FROM BRUTE FORCE
        unsigned int true_min_x_dec;
        printf("2\n\n");
        cudaMemcpyAsync(&true_min_val, fx_gpu_BF, sizeof(fx_Type), cudaMemcpyDeviceToHost, stream_BF);
        printf("3\n\n");
        cudaMemcpyAsync(&true_min_x_dec, xs_min_gpu_BF, sizeof(x_dec_Type), cudaMemcpyDeviceToHost, stream_BF);
        printf("4\n\n");

        //printf(cudaStreamQuery(stream_BF) == cudaSuccess ? "stream_BF is ready\n" : "stream_BF is not ready\n");
        //printf("true_min_val = %.1f\n", true_min_val);
        
        
        //PRINTS
        if(verbose || strong_verbose){
            printf("-------------------------------------------------------------\n");
            print_Q(Q, N);
            print_A(A, M, N);
            print_b(b, M);
        }

        //INITIALIZE AL VARIABLES
        mu = initial_mu;
        fill_lambda_lin(lambda, M, initial_lambda, 0);

        //COMPUTE TRUE MAX
        true_max_val = compute_max(Q, N);

        
        //COPMPUTE Q + A^T A
        Q_Type Q_prime[N*(N+1)/2];
        compute_Q_plus_AT_A_upper_triangular_lin(Q, A, Q_prime, M, N);

        //COPY THE DIAGONAL OF Q + A^T A
        Q_Type Q_ATA_diag[N];
        //copy all the elements of Q_plus_AT_A to Q_prime  
        for(int i = 0; i < N; i++){
            Q_ATA_diag[i] = Q_prime[triang_index(i,i,N)];
        }    



        int attempt = 0;
        bool ok;
        bool al_condition;

        do{
            

            //CALCOLO DI Q'

            //devo calcolare Q' = Q + A^T A + diag((lambda - mu b)^T A)
            //calcolo lambda - bu*b
            lambda_Type lambda_mu_b[M];
            for(dim_Type j = 0; j < M; j++){
                lambda_mu_b[j] = lambda[j] - mu * b[j];
            }

            //calcolo lambda_mu_b^T * A
            lambda_Type lambda_mu_b_A[N];
            for(dim_Type i = 0; i < N; i++){
                lambda_mu_b_A[i] = 0;
                for(dim_Type j = 0; j < M; j++){
                    lambda_mu_b_A[i] += lambda_mu_b[j] * A[j+i*M];
                }
            }

            //calcolo Q'
            for(dim_Type i = 0; i < N; i++){
                Q_prime[triang_index(i,i,N)] = Q_ATA_diag[i] + lambda_mu_b_A[i]; 
            }



            //copy Q_plus_AT_A to GPU
            cudaMemcpyToSymbolAsync(Q_prime_const, Q_prime, N*(N+1)/2 * sizeof(Q_Type), 0, cudaMemcpyHostToDevice, stream_BF_AL);

            //LANCIO AL
            brute_force_AL_coarsening<<<blocks_per_grid_bf, threads_per_block_bf, 0, stream_BF_AL>>>(N, COARSENING, fx_gpu_AL, xs_min_gpu_AL);
            CHECK_KERNELCALL();
            //cudaDeviceSynchronize());

            int input_size = (int)pow(2, N - COARSENING);
            while(input_size > 1){
                int n_threads_am = min(N_THREADS_ARGMIN, input_size);
                dim3 threads_per_block_am(n_threads_am);
	            dim3 blocks_per_grid_am(input_size / n_threads_am);   

                reduce_argmin<<<blocks_per_grid_am, threads_per_block_am, 0, stream_BF_AL>>>(fx_gpu_AL, xs_min_gpu_AL);
                CHECK_KERNELCALL();
	            //cudaDeviceSynchronize());


                input_size >>= (int)log2(N_THREADS_ARGMIN);
            }
        


            //COPY BACK RESULTS FROM AL
            unsigned int AL_min_x_dec;
            cudaMemcpyAsync(&al_min_val, fx_gpu_AL, sizeof(fx_Type), cudaMemcpyDeviceToHost, stream_BF_AL);
            cudaMemcpyAsync(&AL_min_x_dec, xs_min_gpu_AL, sizeof(x_dec_Type), cudaMemcpyDeviceToHost, stream_BF_AL);




            //STAMPA LAMBDA E MU
            if(strong_verbose){
                printf("AL attempt %d\tmu = %.5f\tlambda^T = [ ", attempt, mu);
                for(dim_Type idx = 0; idx < M; idx++){
                    printf("%.1f ", lambda[idx]);
                }
                printf("]\n");
            }
            


            
            cudaStreamSynchronize(stream_BF_AL);///REMOVE OR MOVE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


            //TRASFORMO x in array binario
            for(int i = 0; i < N; i++){
                min_x[i] = (AL_min_x_dec >> i) & 1;
            }


            //CALCOLO c(x_ottimo) = A x_ottimo - b
            for(dim_Type i = 0; i < M; i++){
                c[i] = -b[i];
                for(dim_Type j = 0; j < N; j++){
                    c[i] += A[i+j*M] * min_x[j];
                }
            }

            
            if(strong_verbose){
                printf("c_x_opt^T = [");
                for(int i = 0; i < M; i++){
                    std::cout << std::setw(2) << c[i] << " ";
                    //std::printf("%.5f ", c[i]);
                }
                printf("]\tx_opt = [ ");
                for(int i = 0; i < N; i++){
                    printf("%d ", min_x[i]);
                }
                printf("]\tmin_val = %.1f\n", al_min_val);
            }

            //UPDATE DI LAMBDA E MU
            for(dim_Type j = 0; j < M; j++){
                old_lambda[j] = lambda[j];
            }
            old_mu = mu;
            
            ok = true;
            for(dim_Type j = 0; j < M; j++){
                if(c[j] > 0){
                    lambda[j] = lambda[j] + mu * c[j];               //ORIGINALEEEEEE
                    //lambda[i][0] = lambda[i][0] + rho * c[i][0];

                    ok = false;
                }
            }

            attempt++;

            mu = update_mu(mu, rho);

            al_condition = al_end_condition(attempt, N_AL_ATTEMPTS, N, M, lambda, mu, c);


        } while (!ok && al_condition);

        cudaStreamSynchronize(stream_BF);  ///REMOVE OR MOVE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        //PRINT DEL MINIMO TROVATO <<<<<<<<<<<<<<<<<<<<<< VA FATTO DOPO CHE BRUTE FORCE HA FINITO
        if(strong_verbose){
            for(int i = 0; i < N; i++){
                expected_min_x[i] = (true_min_x_dec >> i) & 1;
            }
            printf("Expected minimum found in x = [ ");
            for(dim_Type i = 0; i < N; i++){
                std::cout << expected_min_x[i] << " ";
            }
            std::cout << "] with value " << true_min_val << std::endl << std::endl;
        }


        Q_Type current_xQ_prime_x = compute_xQx(Q, min_x, N);
        correct = al_condition && ok && current_xQ_prime_x == true_min_val;
        unfinished = !al_condition;
        if(correct && unfinished){
            printf("ERROR: the same problem is both correct and unfinished\n");
            return 0;
        } else if(!correct && !unfinished){
            wrong = true;
        }

        if(correct){                      //AL has chosen the right minimum (consciously)
            correct_counter++; 
            if(strong_verbose)
                printf("PROBLEM SOLVED CORRECTLY\n");
        } 
        else if(unfinished){                //AL has reached the termination condition without finding a feasible minimum 
            if(strong_verbose)
                printf("PROBLEM NOT SOLVED\n");                            
            unfinished_counter++;
        }
        else if(wrong){                     //AL has chosen the wrong minimum
            if(strong_verbose)
                printf("PROBLEM SOLVED WRONGLY\n");  
            normalized_error_mean += true_max_val-true_min_val != 0 ? (current_xQ_prime_x - true_min_val) / (true_max_val-true_min_val) : 1;
            //It DOESN'T make sesnse that the error is negative. true_min_val is the minimum feasible value of the function, if AL exits the loop beleiving that a lower minimum (that could exists) fulfils the constraints, there is a problem while checking c(x)
            if(normalized_error_mean < 0){
                printf("ERROR!\ntrue max val : %.1f\t true min val: %.1f\t xQx: %.1f\n", true_max_val, true_min_val, current_xQ_prime_x);
                print_Q(Q, N);
                print_A(A, M, N);
                print_b(b, M);
                printf("c = \n");
                for(int i = 0; i < M; i++){
                    std::cout << std::setw(2) << c[i] << " ";
                    //printf("%.1f ", c[i]);
                }
                printf("\n");
                return 0;
            }
        } else {
            printf("ERROR: something went wrong\n");
            return 0;
        }
       
        // Print progress bar
        if(!verbose && !strong_verbose){
            printf("[");
            int pos = progressBarWidth * (iter+1) / MAXITER;
            for (int j = 0; j < progressBarWidth; ++j) {
                if (j < pos) printf("=");
                else if (j == pos) printf(">");
                else printf(" ");
            }
            printf("] %d %%\r", int((iter+1) * 100.0 / MAXITER));
            fflush(stdout);

        } else {
            if(ok){
                printf("Problem solved in %d iterations\n", attempt);
            } else{
                printf("Problem not solved in %d iterations\n", attempt);
            }

            if(!strong_verbose){
                printf("c_x^T =\t\t[");
                for(int i = 0; i < M; i++){
                    std::cout << std::setw(2) << c[i] << " ";
                    //printf("%.1f\t", c[i]);
                }
                printf("]\nlambda^T =\t[\t");
                for(int i = 0; i < M; i++){
                    printf("%.1f\t", lambda[i]);
                }
                printf("]\nmu =\t\t%.1f\n\n", mu);
            }
        }


        if(correct){
            mean_al_attempts_on_correct_solutions += attempt;
            mean_mu_on_correct_solutions += old_mu;
            for(int j = 0; j < M; j++){
                mean_lambda_on_correct_solutions += old_lambda[j]/M;
                if(lambda[j] < lambda_min_on_correct_solutions)
                    lambda_min_on_correct_solutions = lambda[j];
                if(lambda[j] > lambda_max_on_correct_solutions)
                    lambda_max_on_correct_solutions = lambda[j];
            }
        }
        else if(unfinished){
            mean_al_attempts_on_unfinished_solutions += attempt;
            mean_mu_on_unfinished_solutions += old_mu;
            for(int j = 0; j < M; j++){
                mean_lambda_on_unfinished_solutions += old_lambda[j]/M;
                if(lambda[j] < lambda_min_on_unfinished_solutions)
                    lambda_min_on_unfinished_solutions = lambda[j];
                if(lambda[j] > lambda_max_on_unfinished_solutions)
                    lambda_max_on_unfinished_solutions = lambda[j];
            }
        }
        else if(wrong){
            mean_al_attempts_on_wrong_solutions += attempt;
            mean_mu_on_wrong_solutions += old_mu;
            for(int j = 0; j < M; j++){
                mean_lambda_on_wrong_solutions += old_lambda[j]/M;
                if(lambda[j] < lambda_min_on_wrong_solutions)
                    lambda_min_on_wrong_solutions = lambda[j];
                if(lambda[j] > lambda_max_on_wrong_solutions)
                    lambda_max_on_wrong_solutions = lambda[j];
            }
        }
        

    }


    mean_lambda_on_correct_solutions = correct_counter != 0 ? mean_lambda_on_correct_solutions / correct_counter : 0;
    mean_mu_on_correct_solutions = correct_counter != 0 ? mean_mu_on_correct_solutions / correct_counter : 0;
    mean_al_attempts_on_correct_solutions = correct_counter != 0 ? mean_al_attempts_on_correct_solutions / correct_counter : 0;

    mean_lambda_on_unfinished_solutions = unfinished_counter != 0 ? mean_lambda_on_unfinished_solutions / unfinished_counter : 0;
    mean_mu_on_unfinished_solutions = unfinished_counter != 0 ? mean_mu_on_unfinished_solutions / unfinished_counter : 0;
    mean_al_attempts_on_unfinished_solutions = unfinished_counter != 0 ? mean_al_attempts_on_unfinished_solutions / unfinished_counter : 0;

    mean_lambda_on_wrong_solutions = MAXITER - correct_counter - unfinished_counter != 0 ? mean_lambda_on_wrong_solutions / (MAXITER - correct_counter - unfinished_counter) : 0;
    mean_mu_on_wrong_solutions = MAXITER - correct_counter - unfinished_counter != 0 ? mean_mu_on_wrong_solutions / (MAXITER - correct_counter - unfinished_counter) : 0;
    mean_al_attempts_on_wrong_solutions = MAXITER - correct_counter - unfinished_counter != 0 ? mean_al_attempts_on_wrong_solutions / (MAXITER - correct_counter - unfinished_counter) : 0;


    normalized_error_mean = MAXITER - correct_counter != 0 ? normalized_error_mean / (MAXITER - correct_counter) : 0;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    results->N = N;
    results->M = M;
    results->mean_al_attempts_on_correct_solutions = mean_al_attempts_on_correct_solutions;
    results->mean_al_attempts_on_wrong_solutions = mean_al_attempts_on_wrong_solutions;
    results->mean_al_attempts_on_unfinished_solutions = mean_al_attempts_on_unfinished_solutions;
    results->correct_ratio = (double)correct_counter/MAXITER;
    results->unfinished_ratio = (double)unfinished_counter/MAXITER;
    results->normalized_error_mean = normalized_error_mean;
    results->mean_lambda_on_correct_solutions = mean_lambda_on_correct_solutions;
    results->mean_lambda_on_unfinished_solutions = mean_lambda_on_unfinished_solutions;
    results->mean_lambda_on_wrong_solutions = mean_lambda_on_wrong_solutions;
    results->lambda_min_on_correct_solutions = lambda_min_on_correct_solutions;
    results->lambda_min_on_unfinished_solutions = lambda_min_on_unfinished_solutions;
    results->lambda_min_on_wrong_solutions = lambda_min_on_wrong_solutions;
    results->lambda_max_on_correct_solutions = lambda_max_on_correct_solutions;
    results->lambda_max_on_unfinished_solutions = lambda_max_on_unfinished_solutions;
    results->lambda_max_on_wrong_solutions = lambda_max_on_wrong_solutions;
    results->mean_mu_on_correct_solutions = mean_mu_on_correct_solutions;
    results->mean_mu_on_unfinished_solutions = mean_mu_on_unfinished_solutions;
    results->mean_mu_on_wrong_solutions = mean_mu_on_wrong_solutions;
    results->duration = elapsed.count();

    //Free GPU memory
    //cudaFree(A_gpu));
    //cudaFree(Q_gpu));
    //cudaFree(b_gpu));
    
    // cudaFree(x_bin_buffer_gpu));
    // cudaFree(Ax_b_buffer_gpu));

    cudaFree(fx_gpu_BF);
    cudaFree(xs_min_gpu_BF);

    cudaFree(fx_gpu_AL);
    cudaFree(xs_min_gpu_AL);

    cudaStreamDestroy(stream_BF);
    cudaStreamDestroy(stream_BF_AL) ;

    // Deallocate
    delete[] Q;
    delete[] A;
    delete[] b;
    delete[] lambda;
    delete[] old_lambda;
    delete[] expected_min_x;
    delete[] min_x;
    delete[] c;

    return 1;
}

void finalize(std::vector<test_results> results){
    std::time_t t = std::time(nullptr);
    char mbstr[100];
    std::strftime(mbstr, sizeof(mbstr), "%Y%m%d_%H%M%S", std::localtime(&t));

    std::stringstream filename;
    filename << results_path << "/results_" << mbstr;
    if(strlen(name_suffix) > 0){
        filename << "__" << name_suffix;
    }
    filename << ".csv";

    FILE* file = fopen(filename.str().c_str(), "w");
    fprintf(file, "N,M,correct_ratio,unfinished_ratio,normalized_error_mean,mean_al_attempts_on_correct_solutions,mean_al_attempts_on_wrong_solutions,mean_al_attempts_on_unfinished_solutions,mean_lambda_on_correct_solutions,mean_lambda_on_unfinished_solutions,mean_lambda_on_wrong_solutions,mean_mu_on_correct_solutions,mean_mu_on_unfinished_solutions,mean_mu_on_wrong_solutions,lambda_min_on_correct_solutions,lambda_min_on_unfinished_solutions,lambda_min_on_wrong_solutions,lambda_max_on_correct_solutions,lambda_max_on_unfinished_solutions,lambda_max_on_wrong_solutions,duration\n");
    for(int i = 0; i < results.size(); i++){
        fprintf(file, "%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f%s", 
            results[i].N,
            results[i].M,
            results[i].correct_ratio,
            results[i].unfinished_ratio,
            results[i].normalized_error_mean,
            results[i].mean_al_attempts_on_correct_solutions,
            results[i].mean_al_attempts_on_wrong_solutions,
            results[i].mean_al_attempts_on_unfinished_solutions,
            results[i].mean_lambda_on_correct_solutions,
            results[i].mean_lambda_on_unfinished_solutions,
            results[i].mean_lambda_on_wrong_solutions,
            results[i].mean_mu_on_correct_solutions,
            results[i].mean_mu_on_unfinished_solutions,
            results[i].mean_mu_on_wrong_solutions,
            results[i].lambda_min_on_correct_solutions,
            results[i].lambda_min_on_unfinished_solutions,
            results[i].lambda_min_on_wrong_solutions,
            results[i].lambda_max_on_correct_solutions,
            results[i].lambda_max_on_unfinished_solutions,
            results[i].lambda_max_on_wrong_solutions,
            results[i].duration,
            i < results.size()-1 ? "\n" : ""
            );
        }

    fclose(file);
}

void finalize(test_results mean_results){
    std::time_t t = std::time(nullptr);
    char mbstr[100];
    std::strftime(mbstr, sizeof(mbstr), "%Y%m%d_%H%M%S", std::localtime(&t));

    std::stringstream filename;
    filename << results_path << "mean_results_" << mbstr;
    if(strlen(name_suffix) > 0){
        filename << "__" << name_suffix;
    }
    filename << ".csv";

    FILE* file = fopen(filename.str().c_str(), "w");
    fprintf(file, "N,M,correct_ratio,unfinished_ratio,normalized_error_mean,mean_al_attempts_on_correct_solutions,mean_al_attempts_on_wrong_solutions,mean_al_attempts_on_unfinished_solutions,mean_lambda_on_correct_solutions,mean_lambda_on_unfinished_solutions,mean_lambda_on_wrong_solutions,mean_mu_on_correct_solutions,mean_mu_on_unfinished_solutions,mean_mu_on_wrong_solutions,lambda_min_on_correct_solutions,lambda_min_on_unfinished_solutions,lambda_min_on_wrong_solutions,lambda_max_on_correct_solutions,lambda_max_on_unfinished_solutions,lambda_max_on_wrong_solutions,duration\n");
    fprintf(file, "%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f", 
        mean_results.N,
        mean_results.M,
        mean_results.correct_ratio,
        mean_results.unfinished_ratio,
        mean_results.normalized_error_mean,
        mean_results.mean_al_attempts_on_correct_solutions,
        mean_results.mean_al_attempts_on_wrong_solutions,
        mean_results.mean_al_attempts_on_unfinished_solutions,
        mean_results.mean_lambda_on_correct_solutions,
        mean_results.mean_lambda_on_unfinished_solutions,
        mean_results.mean_lambda_on_wrong_solutions,
        mean_results.mean_mu_on_correct_solutions,
        mean_results.mean_mu_on_unfinished_solutions,
        mean_results.mean_mu_on_wrong_solutions,
        mean_results.lambda_min_on_correct_solutions,
        mean_results.lambda_min_on_unfinished_solutions,
        mean_results.lambda_min_on_wrong_solutions,
        mean_results.lambda_max_on_correct_solutions,
        mean_results.lambda_max_on_unfinished_solutions,
        mean_results.lambda_max_on_wrong_solutions,
        mean_results.duration       
        );
    
    fclose(file);
}

void print_file_stdout(FILE *file, const char *format, ...) {
    va_list args;
    va_start(args, format);
    vprintf(format, args); // Print to stdout
    va_end(args);

    va_start(args, format);
    vfprintf(file, format, args); // Print to file
    va_end(args);
}