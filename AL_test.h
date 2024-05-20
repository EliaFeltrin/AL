#pragma once

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

#include "kernels.cu"
#include "types.h"

extern double MAX_MU;
extern double MAX_LAMBDA;

extern char name_suffix[20];  
extern char results_path[100];

enum stop_conditions_names {max_Al_attempts, max_mu, max_lambda, stop_conditions_end};
enum fill_distributions {uniform, MMF, PCR, PCRL, fill_distributions_end};

extern bool Q_DIAG;
extern bool Q_ID;
extern bool PCR_PROBLEM;

#define CHECK(call)                                                                         \
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


void fill_Q_id_lin(Q_Type*  Q, const dim_Type N, const Q_Type not_used_1, const Q_Type not_used_2);
void fill_Q_diag_lin(Q_Type*  Q, const dim_Type N, const Q_Type lowerbound, const Q_Type upperbund);
void fill_Q_upper_trianular_lin(Q_Type *Q, const dim_Type N, const Q_Type lowerbound, const Q_Type upperbound);
void fill_Q_manual_lin(Q_Type*  Q, const dim_Type N, const Q_Type unused_1, const Q_Type unused_2) {printf("not yet implemented"); exit(0);}

void fill_A_neg_binary_lin(A_Type*  A, const dim_Type M, const dim_Type N, const float one_probability, const b_Type b);
void fill_A_manual_lin(A_Type*  A, const dim_Type M, const dim_Type N, const float unused_1, const b_Type unused_2) {printf("not yet implemented"); exit(0);}

void fill_b_vector_lin(b_Type* b, const dim_Type M, const b_Type b_val);
void fill_b_manual_lin(b_Type* b, const dim_Type M, const b_Type unused) {printf("not yet implemented"); exit(0);}

void fill_lambda_lin(lambda_Type* lambda, const dim_Type M, lambda_Type initial_lambda);
void fill_lambda_lin(lambda_Type* lambda, const dim_Type M, lambda_Type initial_lambda, lambda_Type noise_amplitude);

int test_at_dimension(  dim_Type N, dim_Type M, int MAXITER, int N_AL_ATTEMPTS, mu_Type initial_mu, lambda_Type initial_lambda,  mu_Type rho, 
                        void (*fill_Q)(Q_Type *Q, const dim_Type N, const Q_Type lowerbound_or_unused, const Q_Type upperbound_or_unused), Q_Type lb_Q, Q_Type ub_Q, 
                        void (*fill_A)(A_Type* A, const dim_Type M, const dim_Type N, const float one_probability_or_unused, const b_Type b_or_unused), float one_prob, 
                        void (*fill_b)(b_Type* b, const dim_Type M, const b_Type b_val_or_unused), b_Type b_val, 
                        std::function<bool(const int i, const int N_AL_ATTEMPTS, const dim_Type N, const dim_Type M, const lambda_Type* __restrict__ lambda, const mu_Type mu, const b_Type* __restrict__ c)> al_end_condition, 
                        mu_Type (*update_mu)(const mu_Type mu, const mu_Type rho), 
                        test_results* results, bool verbose, bool strong_verbose);

void print_Q(const Q_Type* Q, const dim_Type N);
void print_A(const A_Type* A, const dim_Type M, const dim_Type N);
void print_b(const b_Type* b, const dim_Type M);


Q_Type compute_xQx(const Q_Type* __restrict__ Q, const bool* __restrict__ x, dim_Type N);


inline mu_Type update_mu_exp(const mu_Type mu, const mu_Type rho){
    return mu * rho;
}

inline mu_Type update_mu_lin(const mu_Type mu, const mu_Type rho){
    return mu + rho;
}


void compute_Q_plus_AT_A_upper_triangular_lin(const Q_Type* __restrict__ Q, A_Type* __restrict__ A, A_Type* __restrict__ Q_plus_AT_A, const dim_Type M, const dim_Type N);

inline dim_Type triang_index(dim_Type i, dim_Type j, dim_Type N){
    return i * (N - 0.5) - i*i/2 + j;
}

void finalize(test_results mean_results);
void finalize(std::vector<test_results> results);
void print_file_stdout(FILE *file, const char *format, ...);











