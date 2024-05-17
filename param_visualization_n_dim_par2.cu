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



#include "kernels.cu"

#define add 1
#define sub 0

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

double MAX_MU = DBL_MAX;
double MAX_LAMBDA = DBL_MAX;

char name_suffix[20] = "";  
char results_path[100] = "./results/";

enum stop_conditions_names {max_Al_attempts, max_mu, max_lambda, stop_conditions_end};
enum fill_distributions {uniform, MMF, PCR, PCRL, fill_distributions_end};

bool Q_DIAG = false;
bool PCR_PROBLEM = false;


//struct to store the results of the tests
struct test_results{
    int N;
    int M;
    double correct_ratio;
    double unfinished_ratio;
    double normalized_error_mean;
    float mean_al_attempts_on_correct_solutions;
    float mean_al_attempts_on_wrong_solutions;
    float mean_al_attempts_on_unfinished_solutions;
    double mean_lambda_on_correct_solutions;
    double mean_lambda_on_unfinished_solutions;
    double mean_lambda_on_wrong_solutions;
    double mean_mu_on_correct_solutions;
    double mean_mu_on_unfinished_solutions;
    double mean_mu_on_wrong_solutions;
    double lambda_min_on_correct_solutions;
    double lambda_min_on_unfinished_solutions;
    double lambda_min_on_wrong_solutions;
    double lambda_max_on_correct_solutions;
    double lambda_max_on_unfinished_solutions;
    double lambda_max_on_wrong_solutions;
    double duration;
};

void print_file_stdout(FILE *file, const char *format, ...);

// SYNTETIC DATA FUNCTIONS
//function to fill a square up-triangulare matrix dim*dim with random values between lb and up
void fill_Q_matrix_uniform(double** matrix, int dim, float lb, float ub);

//function to fill a rm*cm matrix with random values between lb and up
void fill_A_matrix_uniform(double** matrix, int rm, int cm, float lb, float ub);

//function to fill vector b of size M with random values between lb and up
void fill_b_vector_uniform(double** vector, int len, float lb, float ub);

void fill_lamnda(double** lambda, int M, float initial_lambda, float noise_amplitude);

//like in "Quantum Multimodel Fitting" paper. 
void fill_Q_MMF(double** matrix, int dim, float not_used_1, float not_used_2);
// since the inequality constraint is Pz >= 1 with P filled with 1s and 0s, it turns into Pz <= -1 with P filled with -1s and 0s
void fill_A_MMF(double** matrix, int rm, int cm, float one_probability, float not_used_2);
// since the inequality constraint is Pz >= 1 with P filled with 1s and 0s, it turns into Pz <= -1 with P filled with -1s and 0s
void fill_b_MMF(double** vector, int len, float not_used_1, float not_used_2);
// fill Q with diagonal values between lb and ub
void fill_Q_PCRL(double** matrix, int dim, float lb, float ub);
// fill b with -b (since >= instead of <=)
void fill_b_PCR(double** vector, int len, float b, float not_used_2);
//fill matrix A manually by std input
void fill_A_manual(double** matrix, int rm, int cm, float not_used_1, float not_used_2) {
    printf("Enter the elements of the A matrix:\n");
    for (int i = 0; i < rm; i++) {
        for (int j = 0; j < cm; j++) {
            scanf("%lf", &matrix[i][j]);
        }
    }
}

// Fill matrix Q manually by std input
void fill_Q_manual(double** matrix, int dim, float not_used_1, float not_used_2) {
    printf("Enter the elements of the Q matrix:\n");
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            scanf("%lf", &matrix[i][j]);
        }
    }
    //print the matrix
    printf("The Q matrix is:\n");
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            printf("%lf ", matrix[i][j]);
        }
        printf("\n");
    }
}

// Fill vector b manually by std input
void fill_b_manual(double** vector, int len, float not_used_1, float not_used_2) {
    printf("Enter the elements of the b vector:\n");
    for (int i = 0; i < len; i++) {
        scanf("%lf", vector[i]);
    }
    //print the vector
    printf("The b vector is:\n");
    for (int i = 0; i < len; i++) {
        printf("%lf ", vector[i][0]);
    }
}
//function to multiply two matrices A and B passing their dimensions
//NB: do not use A or B as C
void multiply_matrices(double** A, double** B, double** C, int rA, int cA, int cB, bool A_diagonal, bool A_diagonal_1, bool B_diagonal, bool B_diagonal_1);

//function to sum/subtract two matrices A and B passing their dimensions
void sum_matrices(double** A, double** B, double** C, int rA, int cA, int add_sub);

//function to calculate the norm of a vector
double norm(double** vector, int len);

//function to copy a vector a to a vector b
void copy_vector(double** a, double** b, int len);

//function to transope a vector
//NB: do not use a as b
void transpose_vector(double** a, double** b, int len);

//function to check if two vectors are equal
bool equal_vec(double** a, double** b, int len);

//function to check if constraints are satisfied
int check_c(double** c, int rA);

//function to find the minimum via brute force
bool find_x_min_brute_force(double** Q, int dim, double** A, int rA, double** b, double** returned_min_x, double* returned_max_val, double* returned_min_val, bool verbose);

//function to find the minimum via brute force with AL
void find_x_min_AL_brute_force(double** Q, int dim, double** A, int rA, double** b, double** lambda, double mu, double** returned_min_x, double** returned_c, double* retured_min_val);

//function to calculate x^tQx
double calculate_xQx(double** Q, double** x, int dim);

//function to test the algorithm at a certain dimension
int test_at_dimension(int N,  
    int M, 
    int MAXITER, 
    int N_AL_ATTEMPTS, 
    double initial_mu, 
    double initial_lambda, 
    double rho, 
    void (*fill_Q)(double**, int, float, float),
    double lb_Q,
    double ub_Q,
    void (*fill_A)(double**, int, int, float, float),
    double lb_A,
    double ub_A,
    void (*fill_b)(double**, int, float, float),
    double lb_b,
    double ub_b, 
    std::function<bool(int, int, int, int, double**, double, double**)> al_end_condition, 
    double (*update_mu)(double, double),
    test_results* results, 
    bool verbose, 
    bool strong_verbose);

//function to save in a .csv file the results of the tests
void finalize(std::vector<test_results> results);
void finalize(test_results mean_results);

//al terimination condition: parameters: i, N_AL_ATTEMPTS, lambda, mu, c
bool max_Al_attempts_condition(int i, int N_AL_ATTEMPTS, int N, int M, double** lambda, double mu, double** c){
    return i < N_AL_ATTEMPTS;
}

bool max_mu_condition(int i, int MAXITER, int N, int M, double** lambda, double mu, double** c){
    return mu < MAX_MU;
}

bool max_lambda_condition(int i, int MAXITER, int N, int M, double** lambda, double mu, double** c){
    for(int i = 0; i < M; i++){
        if(lambda[i][0] >= MAX_LAMBDA){
            return false;
        }
    }
    return true;
}

double update_mu_exp(double mu, double rho){
    return mu * rho;
}

double update_mu_lin(double mu, double rho){
    return mu + rho;
}



int main(int argc, char** argv) {

    //printf("check 0\n");

    //default values
    int MIN_N = 1;
    int MAX_N = 20;
    int MIN_M = 1;
    int MAX_M = 20;
    int MAXITER = 1000;
    int N_AL_ATTEMPTS = 1000;
    double initial_mu = 0.1;
    double initial_lambda = 0.1;
    double rho = 1.1;
    float PARAM_1_Q = -10;
    float PARAM_1_A = -10;
    float PARAM_1_b = -10;
    float PARAM_2_Q = 10;
    float PARAM_2_A = 10;
    float PARAM_2_b = 10;
    bool verbose = false;
    bool strong_verbose = false;
    bool only_final_report = false;


    char fill_distributions_names_str[fill_distributions_end][20] = {"uniform", "MMF", "PCR", "PCRlinear"};
    enum QAb {Q, A, b, QAb_end};
    unsigned int selected_fill_distributions[QAb_end] = {uniform, uniform, uniform};
    char stop_conditions_names_str[stop_conditions_end][20] = {"max_Al_attempts", "max_mu", "max_lambda"};
    int stop_condition_counter = 0;
    bool selected_stop_conditions[3] = {false, false, false};
    bool(*stop_conditions[3])(int, int, int, int, double**, double, double**) = {max_Al_attempts_condition, &max_mu_condition, &max_lambda_condition};
   
    auto end_condition_mix = [&selected_stop_conditions, &stop_conditions](int i, int MAXITER, int N, int M, double** lambda, double mu, double** c) -> bool{
        bool return_val = true;
        for(int j = 0; j < stop_conditions_end; j++){
            if(selected_stop_conditions[j]){
                return_val = return_val && stop_conditions[j](i, MAXITER, N, M, lambda, mu, c);
            }
        }
        return return_val;
    };

    std::function<bool(int, int, int, int, double**, double, double**)> al_end_condition = end_condition_mix;
    void (*fill_Q)(double**, int, float, float) = fill_Q_matrix_uniform;
    void (*fill_A)(double**, int, int, float, float) = fill_A_matrix_uniform;
    void (*fill_b)(double**, int, float, float) = fill_b_vector_uniform;
    double (*update_mu)(double, double) = update_mu_exp;

    bool computer_test = false;

    int opt;
    while ((opt = getopt(argc, argv, "lm:M:N:u:l:i:a:r:n:F:o:P:C:R:b:e::vsdfch::")) != -1) {
        switch (opt) {
            case 'm':
                if(optarg[0] == 'N') MIN_N = atoi(optarg+2);
                else if(optarg[0] == 'M') MIN_M = atoi(optarg+2);
                else if(optarg[0] == 'u') initial_mu = atof(optarg+2); //>= 1 ? atof(optarg+2) : printf("WARNING: mu must be >= 1. Default value will be used.\n");
                else if(optarg[0] == 'l') initial_lambda = atof(optarg+2);
                else if(optarg[0] == 'Q') PARAM_1_Q = atof(optarg+2);
                else if(optarg[0] == 'A') PARAM_1_A = atof(optarg+2);
                else if(optarg[0] == 'b') PARAM_1_b = atof(optarg+2); 
                break;
            case 'M':
                if(optarg[0] == 'N') MAX_N = atoi(optarg+2);
                else if(optarg[0] == 'M' && optarg[1] == 'F') {
                    PARAM_1_A = atof(optarg+3);
                    PARAM_2_A = 1;
                    fill_Q = fill_Q_MMF;
                    fill_A = fill_A_MMF;
                    fill_b = fill_b_MMF;
                    selected_fill_distributions[Q] = selected_fill_distributions[A] = selected_fill_distributions[b] = MMF;
                    Q_DIAG = true;
                    
                } else if(optarg[0] == 'M') MAX_M = atoi(optarg+2);
                else if(optarg[0] == 'u'){
                    MAX_MU = atof(optarg+2);
                    stop_condition_counter++;
                    selected_stop_conditions[max_mu] = true;
                }
                else if(optarg[0] == 'l'){
                    MAX_LAMBDA = atof(optarg+2);
                    stop_condition_counter++;
                    selected_stop_conditions[max_lambda] = true;
                }
                else if(optarg[0] == 'Q') PARAM_2_Q = atof(optarg+2);
                else if(optarg[0] == 'A') PARAM_2_A = atof(optarg+2);
                else if(optarg[0] == 'b') PARAM_2_b = atof(optarg+2);
                break;
            case 'r':
                //rho = atof(optarg) > 1 ? atof(optarg) : printf("WARNING: rho must be > 1. Default value will be used.\n");
                rho = atof(optarg);
                break;
            case 'i':
                MAXITER = atoi(optarg);
                break;
            case 'a':
                N_AL_ATTEMPTS = atoi(optarg);
                stop_condition_counter++;
                selected_stop_conditions[max_Al_attempts] = true;
                break;
            case 'n':
                if(optarg[0] == 'd'){
                    if(optarg[1] == 'Q'){
                        fill_Q = fill_Q_matrix_uniform;
                        selected_fill_distributions[Q] = uniform;
                    }else if(optarg[1] == 'A'){
                        fill_A = fill_A_matrix_uniform;
                        selected_fill_distributions[A] = uniform;
                    } else if(optarg[1] == 'b'){
                        fill_b = fill_b_vector_uniform;
                        selected_fill_distributions[b] = uniform;
                    } else {
                        fill_Q = fill_Q_matrix_uniform;
                        fill_A = fill_A_matrix_uniform;
                        fill_b = fill_b_vector_uniform;
                        selected_fill_distributions[Q] = selected_fill_distributions[A] = selected_fill_distributions[b] = uniform;
                    }
                }
                break;
            case 'P':
                if(optarg[0] == 'C' && optarg[1] == 'R')
                    if(optarg[2] == 'b'){
                        Q_DIAG = true;
                        PCR_PROBLEM = true;
                        std::string arg(optarg+4);
                        size_t delimiter_pos = arg.find(",");
                        if (delimiter_pos != std::string::npos) {
                            PARAM_1_A = std::stof(arg.substr(0, delimiter_pos));
                            PARAM_1_b = -std::stof(arg.substr(delimiter_pos + 1));
                            PARAM_2_A = PARAM_1_b;
                            fill_Q = fill_Q_MMF;
                            fill_A = fill_A_MMF;
                            fill_b = fill_b_PCR;
                            selected_fill_distributions[Q] = MMF; selected_fill_distributions[A] = MMF, selected_fill_distributions[b] = PCR;
                        } else {
                            printf("Invalid argument for -PCRb option\n");
                            exit(EXIT_FAILURE);
                        }
                        break;
                    } else if(optarg[2] == 'l'){
                        Q_DIAG = true;
                        PCR_PROBLEM = true;
                        std::string arg(optarg+4);
                        size_t delimiter_pos_1 = arg.find(",");
                        size_t delimiter_pos_2 = arg.find(",", delimiter_pos_1 + 1);
                        if (delimiter_pos_1 != std::string::npos) {
                            PARAM_1_A = std::stof(arg.substr(0, delimiter_pos_1));
                            PARAM_1_b = -std::stof(arg.substr(delimiter_pos_1 + 1, delimiter_pos_2));
                            PARAM_2_A = PARAM_1_b;
                            delimiter_pos_1 = arg.find(",", delimiter_pos_2 + 1);
                            PARAM_1_Q = std::stof(arg.substr(delimiter_pos_2 + 1, delimiter_pos_1));
                            PARAM_2_Q = std::stof(arg.substr(delimiter_pos_1 + 1));
                            fill_Q = fill_Q_PCRL;
                            fill_A = fill_A_MMF;
                            fill_b = fill_b_PCR;
                            selected_fill_distributions[Q] = PCRL, selected_fill_distributions[A] = MMF; selected_fill_distributions[b] = PCR;
                        } else {
                            printf("Invalid argument for -PCRl option\n");
                            exit(EXIT_FAILURE);
                        }
                        break;
                    } else if(optarg[2] == 'q'){
                        std::string arg(optarg+4);
                        size_t delimiter_pos = arg.find(",");
                        size_t delimiter_pos_1 = arg.find(",");
                        size_t delimiter_pos_2 = arg.find(",", delimiter_pos_1 + 1);
                        if (delimiter_pos != std::string::npos) {
                            PARAM_1_A = std::stof(arg.substr(0, delimiter_pos_1));
                            PARAM_1_b = -std::stof(arg.substr(delimiter_pos_1 + 1, delimiter_pos_2));
                            PARAM_2_A = PARAM_1_b;
                            delimiter_pos_1 = arg.find(",", delimiter_pos_2 + 1);
                            PARAM_1_Q = std::stof(arg.substr(delimiter_pos_2 + 1, delimiter_pos_1));
                            PARAM_2_Q = std::stof(arg.substr(delimiter_pos_1 + 1));
                            fill_Q = fill_Q_matrix_uniform;
                            fill_A = fill_A_MMF;
                            fill_b = fill_b_PCR;
                            selected_fill_distributions[Q] = uniform, selected_fill_distributions[A] = MMF, selected_fill_distributions[b] = PCR;
                        } else {
                            printf("Invalid argument for -PCRQ option\n");
                            exit(EXIT_FAILURE);
                        }
                        break;
                    }

            case 'l':
                update_mu = update_mu_lin;
                break;
            case 's':
                strong_verbose = true;
                break;
            case 'v':
                verbose = true;
                break;
            case 'f':
                only_final_report = true;
                verbose = false;
                strong_verbose = false;
                break;
            case 'o':
                if(optarg){
                    strcpy(name_suffix, optarg);
                }
                break;
            case 'c':
                computer_test = true;
                break;
            case 'h':
                fill_A = fill_A_manual;
                fill_Q = fill_Q_MMF;
                fill_b = fill_b_manual;
                break;
            default:
                printf("Usage: %s [-mN MIN_N] [-MN MAX_N] [-mM MIN_M] [-MM MAX_M] [-mu MIN_MU[ [-Mu MAX_MU] [-ml MIN_LAMBDA] [-Ml MAX_LAMBDA] [-mQ PARAM_1_Q] [-MQ PARAM_2_Q] [-mA PARAM_1_A] [-MA PARAM_2_A] [-mb PARAM_1_b] [-Mb PARAM_2_b] [-r RHO] [-i MAXITER] [-a N_AL_ATTEMPTS] [-v](VERBOSE) [-s](STRONG_VERBOSE) [-f](ONLY_FINAL_REPORT) \n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    MAX_N = MAX_N < MIN_N ? MIN_N : MAX_N;
    MAX_M = MAX_M < MIN_M ? MIN_M : MAX_M;

    if(PCR_PROBLEM && MIN_N < -PARAM_1_b){
        printf("\nWARNING: in PCR problem, N must be >= b, otherwise problems cannot be solved.\nMIN_N will be set accordingly\n\n");
        MIN_N = -PARAM_1_b;
    }

    if(update_mu == update_mu_exp && rho <= 1){
        printf("\nWARNING: if mu is an exponential function in AL_attempts, then rho must be > 1. Default value will be used.");
        rho = 1.1;
    }

    if(stop_condition_counter == 0){
        stop_condition_counter = 1;
        selected_stop_conditions[max_Al_attempts] = true;
    }

    //show test params and save on a file
    std::time_t t = std::time(nullptr);
    char mbstr[100];
    std::strftime(mbstr, sizeof(mbstr), "%Y%m%d_%H%M%S", std::localtime(&t));

    std::stringstream filename;
    filename << results_path << "results_" << mbstr;
    if(strlen(name_suffix) > 0){
        filename << "__" << name_suffix;
    }
    filename << ".txt";

    FILE* file = fopen(filename.str().c_str(), "w");
    print_file_stdout(file, "Test parameters:\n");
    print_file_stdout(file, "%d <= N <= %d\n", MIN_N, MAX_N);
    print_file_stdout(file, "%d <= M <= %d\n", MIN_M, MAX_M);
    print_file_stdout(file, "# problems for each combination of N and M = %d\n", MAXITER);
    print_file_stdout(file, "Initial mu = %.1f\n", initial_mu);
    print_file_stdout(file, "rho = %.1f\n", rho);
    print_file_stdout(file, "Initial lambda = %.1f\n", initial_lambda);
    print_file_stdout(file, "Maximum number of AL attempts (if 'PARAM_2_Al_attempts' selected) = %d\n", N_AL_ATTEMPTS);
    print_file_stdout(file, "Maximum value of mu (if 'max_mu' selected) = "); MAX_MU == DBL_MAX ? print_file_stdout(file, "MAX DOUBLE\n") : print_file_stdout(file, "%.1f\n", MAX_MU);
    print_file_stdout(file, "Maximum value of lambda (if 'max_lambda' selected) = "); MAX_LAMBDA == DBL_MAX ? print_file_stdout(file, "MAX DOUBLE\n") : print_file_stdout(file, "%.1f\n", MAX_LAMBDA);
    print_file_stdout(file, "Stop conditions:\n");
    for(int i=0; i<stop_conditions_end; i++){
        if(selected_stop_conditions[i]){
            print_file_stdout(file, "\t%d: %s\n", i+1, stop_conditions_names_str[i]);
        }
    }
    print_file_stdout(file, "MATRIX FILL POLICIES:\n");
    switch(selected_fill_distributions[Q]){
        case uniform:
            print_file_stdout(file, "\tQ matrix: uniform distribution between %f and %f\n", PARAM_1_Q, PARAM_2_Q);
            break;
        case MMF:
            print_file_stdout(file, "\tQ matrix: diagonal 1\n");
            break;
        case PCRL:
            print_file_stdout(file, "\tQ matrix: diagonal, uniform distribution between %f and %f\n", PARAM_1_Q, PARAM_2_Q);
            break;
    }
    switch(selected_fill_distributions[A]){
        case uniform:
            print_file_stdout(file, "\tA matrix: uniform distribution between %f and %f\n", PARAM_1_A, PARAM_2_A);
            break;
        case MMF:
            print_file_stdout(file, "\tA matrix: binary, -1 with probability %f\n", PARAM_1_A);
            break;
    }
    switch(selected_fill_distributions[b]){
        case uniform:
            print_file_stdout(file, "\tb vector: uniform distribution between %f and %f\n", PARAM_1_b, PARAM_2_b);
            break;
        case MMF:
            print_file_stdout(file, "\tb vector: -1\n");
            break;
        case PCR:
            print_file_stdout(file, "\tb vector: %f\n", PARAM_1_b);
            break;
    }
    print_file_stdout(file, "Update mu: %s\n", update_mu == update_mu_exp ? "exponential" : "linear");
    print_file_stdout(file, "Verbose = %s\n", verbose ? "true" : "false");
    print_file_stdout(file, "Strong verbose = %s\n", strong_verbose ? "true" : "false");

    if(!computer_test){
        printf("\nPress ENTER to start, c to exit\n");
        char c = getchar();
        if(c == 'c'){
            exit(1);
        }
    }


    
    int max_n = (MAX_N >= MIN_N ? MAX_N : MIN_N);
    int max_m = (MAX_M >= MIN_M ? MAX_M : MIN_M);

    //initialize vector of results to the size of the number of tests
    std::vector<test_results> results = std::vector<test_results>((max_n - MIN_N + 1) * (max_m - MIN_M + 1));
    bool terminate = false;

    //#pragma omp parallel for sharefd(terminate)
    for(int n = MIN_N; n <= max_n; n++){
        //if(terminate) continue;
        #pragma omp parallel for shared(terminate)
        for(int m = MIN_M; m <= max_m; m++){
            if(terminate) continue;

            test_results current_results;

            if( 0 == test_at_dimension(n, m, MAXITER, N_AL_ATTEMPTS, initial_mu, initial_lambda, rho, fill_Q, PARAM_1_Q, PARAM_2_Q, fill_A, PARAM_1_A, PARAM_2_A, fill_b, PARAM_1_b, PARAM_2_b, al_end_condition, update_mu, &current_results,  verbose, strong_verbose)){
                printf("Some error occured\n");
                finalize(results);
                terminate = true;
            }
            if(!only_final_report){
                printf("\nN = %d\t M = %d\n", current_results.N, current_results.M);
                printf("\tcorrect ratio = %.1f%%    unfinished ratio = %.1f%%    wrong ratio = %.1f%%    normalized mean error = %.1f%%\n", current_results.correct_ratio*100, current_results.unfinished_ratio*100, (1 - current_results.correct_ratio - current_results.unfinished_ratio)*100, current_results.normalized_error_mean*100);
                if(current_results.correct_ratio > 0){
                    printf("\ton correct solutions:\n");
                    printf("\t\tmean lambda = %.1f\tmax lambda = %.1f\tmin lambda = %.1f\n", current_results.mean_lambda_on_correct_solutions, current_results.lambda_max_on_correct_solutions, current_results.lambda_min_on_correct_solutions);
                    printf("\t\tmean mu = %.1f\tmean AL attempts = %.1f\n", current_results.mean_mu_on_correct_solutions, current_results.mean_al_attempts_on_correct_solutions);
                }
                if(current_results.unfinished_ratio > 0){
                    printf("\ton unfinished solutions:\n");
                    printf("\t\tmean lambda = %.1f\tmax lambda = %.1f\tmin lambda = %.1f\n", current_results.mean_lambda_on_unfinished_solutions, current_results.lambda_max_on_unfinished_solutions, current_results.lambda_min_on_unfinished_solutions);
                    printf("\t\tmean mu = %.1f\tmean AL attempts = %.1f\n", current_results.mean_mu_on_unfinished_solutions, current_results.mean_al_attempts_on_unfinished_solutions);
                }
                if(current_results.correct_ratio < 1-current_results.unfinished_ratio){
                    printf("\ton wrong solutions:\n");
                    printf("\t\tmean lambda = %.1f\tmax lambda = %.1f\tmin lambda = %.1f\n", current_results.mean_lambda_on_wrong_solutions, current_results.lambda_max_on_wrong_solutions, current_results.lambda_min_on_wrong_solutions);
                    printf("\t\tmean mu = %.1f\tmean AL attempts = %.1f\n", current_results.mean_mu_on_wrong_solutions, current_results.mean_al_attempts_on_wrong_solutions);
                }
            }
            results[(n-MIN_N)*max_m + m - MIN_M] = current_results;    

        }
    }


    test_results summary = {};
    summary.lambda_min_on_correct_solutions = DBL_MAX;
    summary.lambda_min_on_unfinished_solutions = DBL_MAX;
    summary.lambda_min_on_wrong_solutions = DBL_MAX;
    summary.lambda_max_on_correct_solutions = DBL_MIN;
    summary.lambda_max_on_unfinished_solutions = DBL_MIN;
    summary.lambda_max_on_wrong_solutions = DBL_MIN;
    int tot_tests = results.size();
    printf("tot_tests = %d\n", tot_tests);
    for(int i = 0; i<tot_tests; i++){
        summary.correct_ratio += results[i].correct_ratio/tot_tests;
        summary.unfinished_ratio += results[i].unfinished_ratio/tot_tests;
        summary.normalized_error_mean += results[i].normalized_error_mean/tot_tests;

        summary.mean_al_attempts_on_correct_solutions += results[i].mean_al_attempts_on_correct_solutions/tot_tests;
        summary.mean_al_attempts_on_unfinished_solutions += results[i].mean_al_attempts_on_unfinished_solutions/tot_tests;
        summary.mean_al_attempts_on_wrong_solutions += results[i].mean_al_attempts_on_wrong_solutions/tot_tests;

        summary.mean_lambda_on_correct_solutions += results[i].mean_lambda_on_correct_solutions/tot_tests;
        summary.mean_lambda_on_unfinished_solutions += results[i].mean_lambda_on_unfinished_solutions/tot_tests;
        summary.mean_lambda_on_wrong_solutions += results[i].mean_lambda_on_wrong_solutions/tot_tests;

        summary.mean_mu_on_correct_solutions += results[i].mean_mu_on_correct_solutions/tot_tests;
        summary.mean_mu_on_unfinished_solutions += results[i].mean_mu_on_unfinished_solutions/tot_tests;
        summary.mean_mu_on_wrong_solutions += results[i].mean_mu_on_wrong_solutions/tot_tests;

        summary.lambda_min_on_correct_solutions = results[i].lambda_min_on_correct_solutions < summary.lambda_min_on_correct_solutions ? results[i].lambda_min_on_correct_solutions : summary.lambda_min_on_correct_solutions;
        summary.lambda_min_on_unfinished_solutions = results[i].lambda_min_on_unfinished_solutions < summary.lambda_min_on_unfinished_solutions ? results[i].lambda_min_on_unfinished_solutions : summary.lambda_min_on_unfinished_solutions;
        summary.lambda_min_on_wrong_solutions = results[i].lambda_min_on_wrong_solutions < summary.lambda_min_on_wrong_solutions ? results[i].lambda_min_on_wrong_solutions : summary.lambda_min_on_wrong_solutions;

        summary.lambda_max_on_correct_solutions = results[i].lambda_max_on_correct_solutions > summary.lambda_max_on_correct_solutions ? results[i].lambda_max_on_correct_solutions : summary.lambda_max_on_correct_solutions;
        summary.lambda_max_on_unfinished_solutions = results[i].lambda_max_on_unfinished_solutions > summary.lambda_max_on_unfinished_solutions ? results[i].lambda_max_on_unfinished_solutions : summary.lambda_max_on_unfinished_solutions;
        summary.lambda_max_on_wrong_solutions = results[i].lambda_max_on_wrong_solutions > summary.lambda_max_on_wrong_solutions ? results[i].lambda_max_on_wrong_solutions : summary.lambda_max_on_wrong_solutions;
    }

    printf("\n\nFINAL REPORT (MEAN VALUES)------------------------------------------------------------------------------------------------------------------\n");
        printf("correct ratio = %.1f%%    unfinished ratio = %.1f%%    wrong ration = %.1f%%    normalized mean error = %.1f%%\n", summary.correct_ratio*100, summary.unfinished_ratio*100, (1 - summary.correct_ratio - summary.unfinished_ratio)*100, summary.normalized_error_mean*100);
    if(summary.correct_ratio > 0){
        printf("on correct solutions:\n");
        printf("\tmean lambda = %.1f\tmax lambda = %.1f\tmin lambda = %.1f\n", summary.mean_lambda_on_correct_solutions, summary.lambda_max_on_correct_solutions, summary.lambda_min_on_correct_solutions);
        printf("\tmean mu = %.1f\tmean AL attempts = %.1f\n", summary.mean_mu_on_correct_solutions, summary.mean_al_attempts_on_correct_solutions);
    }
    if(summary.unfinished_ratio > 0){
        printf("on unfinished solutions:\n");
        printf("\tmean lambda = %.1f\tmax lambda = %.1f\tmin lambda = %.1f\n", summary.mean_lambda_on_unfinished_solutions, summary.lambda_max_on_unfinished_solutions, summary.lambda_min_on_unfinished_solutions);
        printf("\tmean mu = %.1f\tmean AL attempts = %.1f\n", summary.mean_mu_on_unfinished_solutions, summary.mean_al_attempts_on_unfinished_solutions);
    }
    if(summary.correct_ratio < 1 - summary.unfinished_ratio){
        printf("on wrong solutions:\n");
        printf("\tmean lambda = %.1f\tmax lambda = %.1f\tmin lambda = %.1f\n", summary.mean_lambda_on_wrong_solutions, summary.lambda_max_on_wrong_solutions, summary.lambda_min_on_wrong_solutions);
        printf("\tmean mu = %.1f\tmean AL attempts = %.1f\n", summary.mean_mu_on_wrong_solutions, summary.mean_al_attempts_on_wrong_solutions);
    }

    finalize(results);
    finalize(summary);

}

void fill_lamnda(double** lambda, int M, float initial_lambda, float noise_amplitude){
    for(int i = 0; i < M; i++){
        lambda[i][0] = initial_lambda + (double)rand()/(double)RAND_MAX * noise_amplitude;
    }
}

void fill_Q_matrix_uniform(double** matrix, int dim, float lb, float ub){
    double range = ub - lb;
    for(int i = 0; i < dim; i++){
        for(int j = i; j < dim; j++){
            matrix[i][j] = lb + (double)rand()/(double)RAND_MAX * range;
        }
    }

    //set 0 to the lower triangular part
    for(int i = 0; i < dim; i++){
        for(int j = 0; j < i; j++){
            matrix[i][j] = 0;
        }
    }
}

void fill_b_vector_uniform(double** vector, int len, float lb, float ub){
    double range = ub - lb;
    for(int i = 0; i < len; i++){
        vector[i][0] = lb + (double)rand()/(double)RAND_MAX * range;
    }
}

void fill_A_matrix_uniform(double** matrix, int rm, int cm, float lb, float ub){
    double range = ub - lb;
    for(int i = 0; i < rm; i++){
        for(int j = 0; j < cm; j++){
            matrix[i][j] = lb + (double)rand()/(double)RAND_MAX * range;
        }
    }
}

void fill_Q_MMF(double** matrix, int dim, float not_used_1, float not_used_2){
    // Fill the matrix with zeros
    for(int i = 0; i < dim; i++){
        for(int j = 0; j < dim; j++){
            matrix[i][j] = 0;
        }
    }

    // Put 1s on the diagonal
    for(int i = 0; i < dim; i++){
        matrix[i][i] = 1;
    }
}


void fill_A_MMF(double** matrix, int rm, int cm, float one_probability, float b){
    for(int i = 0; i < rm; i++){
        for(int j = 0; j < cm; j++){
            matrix[i][j] = 0;
        }
    }


    int int_b = abs((int)b);
    //printf("int_b = %d\n", int_b);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis1(0, cm-1);
    std::unordered_set<int> idxs_to_fill;

    for(int i=0; i<rm; i++){
        idxs_to_fill.clear();


        while (idxs_to_fill.size() < int_b) {
            int n = dis1(gen);
            //printf("n = %d\n", n);  
            idxs_to_fill.insert(n);
        }

        std::unordered_set<int>::iterator it = idxs_to_fill.begin();

        for(int j=0; j<int_b; j++){
            matrix[i][*it] = -1;
            ++it;
        }
    }

    //for(int i = 0; i<rm; i++){
    //    for(int j=0; j<cm; j++){
    //        printf("%.0f ", matrix[i][j]);
    //    }
    //    printf("\n");
    //}
    int n_missing_ones = (int)(rm * cm * one_probability) - rm*int_b;
    if(n_missing_ones < 0){
        printf("ERROR: probability to low to ensure a feasible problem. Given N = %d, M = %d, b = %d, one probability must be >= %f\n", rm, cm, int_b, (float)(rm*int_b)/(rm*cm));
        exit(0);
    }


    int n_empty_spaces = rm * (cm - int_b);
    //printf("n_missing_ones: %d\n", n_missing_ones);
    //printf("n_empty_spaces: %d\n", n_empty_spaces);


    int empty_spaces_idxs[n_empty_spaces];
    int last_esi = 0;
    for(int i = 0; i<rm; i++){
        for(int j=0; j<cm; j++){
            //printf("last_esi = %d\n", last_esi);
            if(matrix[i][j] == 0){
                empty_spaces_idxs[last_esi] = i * cm + j;
                last_esi++;
            }
        }
    }

    idxs_to_fill.clear();
    std::uniform_int_distribution<> dis2(0, n_empty_spaces-1);

    while (idxs_to_fill.size() < n_missing_ones) {
        idxs_to_fill.insert(dis2(gen));
    }

    //printf("empty spaces indexes:\n");
    //for(int i = 0; i<n_empty_spaces; i++){
    //    printf("%d ", empty_spaces_idxs[i]);
    //}

    std::unordered_set<int>::iterator it = idxs_to_fill.begin();
    //printf("idxs_to_fill.size() = %d\n", idxs_to_fill.size());
    for(int i=0; i<n_missing_ones; i++){
        int int_idx = empty_spaces_idxs[*it];
        //printf("i = %d, int_idx = %d, row = %d, col = %d\n", i, int_idx, int_idx / cm, int_idx % cm);
        matrix[int_idx / cm][int_idx % cm] = -1;
        ++it;
        //printf("cehck\n");
    }
    //printf("fill_A done\n");
}

void fill_b_MMF(double** vector, int len, float not_used_1, float not_used_2){
    for(int i = 0; i < len; i++){
        vector[i][0] = -1;
    }
}

void fill_Q_PCRL(double** matrix, int dim, float lb, float ub){
    // Fill the matrix with zeros
    for(int i = 0; i < dim; i++){
        for(int j = 0; j < dim; j++){
            matrix[i][j] = 0;
        }
    }

    // Put 1s on the diagonal
    double range = ub - lb;
    for(int i = 0; i < dim; i++){
        matrix[i][i] = lb + (double)rand()/(double)RAND_MAX * range;    
    }
}

void fill_b_PCR(double** vector, int len, float b, float not_used_2){
    for(int i = 0; i < len; i++){
        vector[i][0] = b;
    }
}

void multiply_matrices(double** A, double** B, double** C, int rA, int cA, int cB, bool A_diagonal = false, bool A_diagonal_1 = false, bool B_diagonal = false, bool B_diagonal_1 = false){
    if(A == C || B == C){
        printf("Do not use A or B as C\n");
        exit(0);
    }

    bool A_is_matrix = rA > 1 && cA > 1;
    bool B_is_matrix = cA > 1 && cB > 1;

    if( (A_diagonal_1 && !A_diagonal) || (B_diagonal_1 && !B_diagonal) || (A_diagonal && !A_is_matrix) || (B_diagonal && !B_is_matrix) || (A_diagonal && rA != cA) || (B_diagonal && cA != cB) ){
        printf("Error: in diagoal/diagonal_1 properties\n");
        exit(0);
    }
    
    if(A_diagonal_1 && !B_is_matrix){
        for(int i = 0; i < rA; i++){
            C[i][0] = B[i][0];
        }
    } else if(A_diagonal && !B_is_matrix){
        for(int i = 0; i < rA; i++){
            C[i][0] = A[i][i] * B[i][0];
        }
    } else if(B_diagonal_1 && !A_is_matrix){
        for(int i = 0; i < cB; i++){
            C[i][0] = A[0][i];
        }
    } else if(B_diagonal && !A_is_matrix){
        for(int i = 0; i < cB; i++){
            C[i][0] = A[0][i] * B[i][i];
        }
    } else {
        for(int i = 0; i < rA; i++){
            for(int j = 0; j < cB; j++){
                C[i][j] = 0;
                for(int k = 0; k < cA; k++){
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }
}

void sum_matrices(double** A, double** B, double** C, int rA, int cA, int add_sub){
    for(int i = 0; i < rA; i++){
        for(int j = 0; j < cA; j++){
            if(add_sub){
                C[i][j] = A[i][j] + B[i][j];
            } else {
                C[i][j] = A[i][j] - B[i][j];
            }
        }
    }
}

double norm(double** vector, int len){
    double norm = 0;
    for(int i = 0; i < len; i++){
        norm += vector[i][0] * vector[i][0];
    }
    return std::sqrt(norm);
}

void copy_vector(double** a, double** b, int len){
    for(int i = 0; i < len; i++){
        b[i][0] = a[i][0];
    }
}


void transpose_vector(double** a, double** b, int len){
    if(a == b){
        printf("Do not use a as b\n");
        exit(0);
    }
    for(int i = 0; i < len; i++){
        b[0][i] = a[i][0];
    }
}

bool equal_vec(double** a, double** b, int len){
    for(int i = 0; i < len; i++){
        if(a[i][0] != b[i][0]){
            return false;
        }
    }
    return true;
}

int check_c(double** c, int rA){
    for(int i = 0; i < rA; i++){
        if(c[i][0] > 0){
            return 0;
        }
    }
    return 1;
}

double calculate_xQx(double** Q, double** x, int dim){
    double** tmp = new double*[dim];
    double** x_t = new double*[1];
    double** result = new double*[1];
    for(int i = 0; i < dim; i++){
        tmp[i] = new double[1];
    }
    x_t[0] = new double[dim];
    result[0] = new double[1];

    double res = 0;

    if(Q_DIAG){
        for(int i = 0; i < dim; i++){
            res += x[i][0] == 1 ? Q[i][i] : 0;
        }
        return res;
    }
    multiply_matrices(Q, x, tmp, dim, dim, 1);
    transpose_vector(x, x_t, dim);
    multiply_matrices(x_t, tmp, result, 1, dim, 1);

    res = result[0][0];
    // Deallocate
    delete [] x_t[0];
    delete [] result[0];

    for(int i = 0; i < dim; ++i) {
        delete [] tmp[i];
    }

    delete [] tmp;
    delete [] x_t;
    delete [] result;

    return res;

}

bool find_x_min_brute_force(double** Q, int dim, double** A, int rA, double** b, double** returned_min_x, double* returned_max_val, double* returned_min_val, bool verbose){

    // Allocate
    double** min_val = new double*[1];
    double** max_val = new double*[1];
    double** current_val = new double*[1];
    double** x = new double*[dim];
    double** x_t = new double*[1];
    double** tmp = new double*[dim];
    double** global_min_x = new double*[dim];
    double** c_x = new double*[rA];

    min_val[0] = new double[1];
    max_val[0] = new double[1];
    current_val[0] = new double[1];
    x_t[0] = new double[dim];
    bool at_lest_one_feasible_solution = false;
    int n_of_minimums = 0;
    

    for(int i = 0; i < dim; ++i){
        x[i] = new double[1];
        tmp[i] = new double[1];
        global_min_x[i] = new double[1];
    }

    for(int i = 0; i < rA; ++i) {
        c_x[i] = new double[1];
    }

    //start brute forcing
    bool init = false;
    for(int i=0; i < pow(2,dim); i++){
        
        //convert i to binary
        for(int j = 0; j < dim; j++){
            x[j][0] = (i >> j) & 0b1;
        }

        multiply_matrices(A, x, c_x, rA, dim, 1);
        sum_matrices(c_x, b, c_x, rA, 1, sub);

        if(!check_c(c_x, rA)){
            continue;
        }
        else at_lest_one_feasible_solution = true;

        //calculate x^tQx
        if(Q_DIAG){
            current_val[0][0] = 0;
            for(int i = 0; i < dim; i++){
                current_val[0][0] += x[i][0] == 1 ? Q[i][i] : 0;
            }
        } else {
            multiply_matrices(Q, x, tmp, dim, dim, 1);
            transpose_vector(x, x_t, dim);
            multiply_matrices(x_t, tmp, current_val, 1, dim, 1);
        }

        if(!init){
            init = true;
            min_val[0][0] = current_val[0][0];
            max_val[0][0] = current_val[0][0];
            copy_vector(x, global_min_x, dim);
            n_of_minimums = 1;
        } else {
            if(current_val[0][0] == min_val[0][0])
                n_of_minimums++;
            else if(current_val[0][0] < min_val[0][0]){
                min_val[0][0] = current_val[0][0];
                n_of_minimums = 1;
                copy_vector(x, global_min_x, dim);
            } else if(current_val[0][0] > max_val[0][0]){
                max_val[0][0] = current_val[0][0];
            }
        }
    }

    if(at_lest_one_feasible_solution){
        copy_vector(global_min_x, returned_min_x, dim);
        *returned_max_val = max_val[0][0];
        *returned_min_val = min_val[0][0];
    } else {
        *returned_max_val = 0;
        *returned_min_val = 0;
    }

    if(verbose){
        if(at_lest_one_feasible_solution){
            printf("Feasible minimum found in x = [ ");
            for(int i = 0; i < dim; i++){
                printf("%.0f ", global_min_x[i][0]);
            }
            printf("] with value %.1f\n", min_val[0][0]);
            printf("Number of minimums found: %d\n", n_of_minimums);
        }
        else
            printf("No feasible solution found.\n");
    }
    
    // Deallocate
    delete [] min_val[0];
    delete [] max_val[0];
    delete [] current_val[0];
    delete [] x_t[0];

    for(int i = 0; i < dim; ++i) {
        delete [] x[i];
        delete [] tmp[i];
        delete [] global_min_x[i];
    }

    for(int i = 0; i < rA; ++i) {
        delete [] c_x[i];
    }

    delete [] min_val;
    delete [] max_val;
    delete [] current_val;
    delete [] x;
    delete [] x_t;
    delete [] tmp;
    delete [] global_min_x;
    delete [] c_x;
    
    return at_lest_one_feasible_solution;
}

void find_x_min_AL_brute_force(double** Q, int dim, double** A, int rA, double** b, double** lambda, double mu, double** returned_min_x, double** returned_c, double* retured_min_val){

    //finding  minimal x for x^tQx + lambda^Tc(x) + mu/2*||c(x)^2|| via brute forcing x

    // Allocate
    double** min_val = new double*[1];
    double** current_val = new double*[1];
    double** x = new double*[dim];
    double** x_t = new double*[1];
    double** tmp = new double*[dim];
    double** global_min_x = new double*[dim];
    double** c_x = new double*[rA];
    double** c_x_to_return = new double*[rA];
    double** lambda_c = new double*[1];
    double** lambda_t = new double*[1];

    min_val[0] = new double[1];
    current_val[0] = new double[1];
    lambda_c[0] = new double[1];
    x_t[0] = new double[dim];
    lambda_t[0] = new double[rA];

    for(int i = 0; i < dim; ++i) {
        x[i] = new double[1];
        tmp[i] = new double[1];
        global_min_x[i] = new double[1];
    }

    for(int i = 0; i < rA; ++i) {
        c_x[i] = new double[1];
        c_x_to_return[i] = new double[1];
    }
    
    double mu_half_norm;

    for(int i=0; i < pow(2, dim); i++){
        //init variables

        //convert i to binary
        for(int j = 0; j < dim; j++){
            x[j][0] = (i >> j) & 1;
        }

        //calculate x^tQx
        if(Q_DIAG){
            current_val[0][0] = 0;
            for(int i = 0; i < dim; i++){
                current_val[0][0] += x[i][0] == 1 ? Q[i][i] : 0;
            }
        } else {
            multiply_matrices(Q, x, tmp, dim, dim, 1);
            transpose_vector(x, x_t, dim);
            multiply_matrices(x_t, tmp, current_val, 1, dim, 1);
        }
        

        //compute c(x)
        multiply_matrices(A, x, c_x, rA, dim, 1);
        sum_matrices(c_x, b, c_x, rA, 1, sub);

        //compute lambda^Tc(x)
        transpose_vector(lambda, lambda_t, rA);
        multiply_matrices(lambda_t, c_x, lambda_c, 1, rA, 1);
        sum_matrices(lambda_c, current_val, current_val, 1, 1, add);

        //compute mu/2*||c(x)^2||
        mu_half_norm = mu/2 * norm(c_x, rA);
        current_val[0][0] += mu_half_norm;

        if(i==0){
            min_val[0][0] = current_val[0][0];
            copy_vector(x, global_min_x, dim);
            copy_vector(c_x, c_x_to_return, rA);
        } else if(current_val[0][0] < min_val[0][0]){
            min_val[0][0] = current_val[0][0];
            copy_vector(x, global_min_x, dim);
            copy_vector(c_x, c_x_to_return, rA);
        }
    }

    copy_vector(global_min_x, returned_min_x, dim);
    copy_vector(c_x_to_return, returned_c, rA);
    if(Q_DIAG){
            current_val[0][0] = 0;
            for(int i = 0; i < dim; i++){
                current_val[0][0] += global_min_x[i][0] == 1 ? Q[i][i] : 0;
            }
        } else {
            multiply_matrices(Q, global_min_x, tmp, dim, dim, 1);
            transpose_vector(global_min_x, x_t, dim);
            multiply_matrices(x_t, tmp, current_val, 1, dim, 1);
        }
    *retured_min_val = current_val[0][0];

    /// Deallocate
    delete [] min_val[0];
    delete [] current_val[0];
    delete [] x_t[0];
    delete [] lambda_c[0];
    delete [] lambda_t[0];

    for(int i = 0; i < dim; ++i) {
        delete [] x[i];
        delete [] tmp[i];
        delete [] global_min_x[i];
    }

    for(int i = 0; i < rA; ++i) {
        delete [] c_x[i];
    }

    delete [] min_val;
    delete [] current_val;
    delete [] x;
    delete [] x_t;
    delete [] tmp;
    delete [] global_min_x;
    delete [] c_x;
    delete [] lambda_c;
    delete [] lambda_t;
}

int test_at_dimension(int N, int M, int MAXITER, int N_AL_ATTEMPTS, double initial_mu, double initial_lambda, double rho, void (*fill_Q)(double**, int, float, float), double lb_Q, double ub_Q, void (*fill_A)(double**, int, int, float, float), double lb_A,double ub_A, void (*fill_b)(double**, int, float, float), double lb_b, double ub_b, std::function<bool(int, int, int, int, double**, double, double**)> al_end_condition, double (*update_mu)(double, double), test_results* results, bool verbose, bool strong_verbose){
    printf("N = %d\tM = %d\n", N, M);
    
    auto start = std::chrono::high_resolution_clock::now();
    const int progressBarWidth = 100;
    srand(time(0));

    // Allocate
    double** Q = new double*[N];
    double** A = new double*[M];
    double** b = new double*[M];
    double** lambda = new double*[M];
    double** old_lambda = new double*[M];
    double** expected_min_x = new double*[N];
    double** min_x = new double*[N];
    double** c = new double*[M];

    for(int i = 0; i < N; ++i) {
        Q[i] = new double[N];
        expected_min_x[i] = new double[1];
        min_x[i] = new double[1];
    }

    for(int i = 0; i < M; ++i) {
        A[i] = new double[N];
        b[i] = new double[1];
        lambda[i] = new double[1];
        old_lambda[i] = new double[1];
        c[i] = new double[1];
    }

    double true_max_val, true_min_val, al_min_val;

    double mu;
    double old_mu;
    double mean_lambda_on_correct_solutions       = 0,    mean_mu_on_correct_solutions      = 0;
    double mean_lambda_on_unfinished_solutions    = 0,    mean_mu_on_unfinished_solutions   = 0;
    double mean_lambda_on_wrong_solutions         = 0,    mean_mu_on_wrong_solutions        = 0;
    double lambda_min_on_correct_solutions        = DBL_MAX,  lambda_max_on_correct_solutions       = DBL_MIN;     
    double lambda_min_on_unfinished_solutions     = DBL_MAX,  lambda_max_on_unfinished_solutions    = DBL_MIN; 
    double lambda_min_on_wrong_solutions          = DBL_MAX,  lambda_max_on_wrong_solutions         = DBL_MIN; 
    double mean_al_attempts_on_correct_solutions     = 0;
    double mean_al_attempts_on_wrong_solutions       = 0;
    double mean_al_attempts_on_unfinished_solutions  = 0;   

    bool correct, unfinished, wrong;

    int correct_counter = 0;
    int unfinished_counter = 0;
    double normalized_error_mean = 0;

    for(int iter = 0; iter < MAXITER; iter++) {
        correct = unfinished = wrong = 0;

        fill_Q(Q, N, lb_Q, ub_Q);
        fill_A(A, M, N, lb_A, ub_A);
        fill_b(b, M, lb_b, ub_b);

        if(verbose || strong_verbose){
            printf("-------------------------------------------------------------\n");
            //print Q, A, b
                printf("Q = \n");
                for(int i = 0; i < N; i++){
                    for(int j = 0; j < N; j++){
                        printf("%.1f\t", Q[i][j]);
                    }
                    printf("\n");
                }
                printf("\nA = \n");
                for(int i = 0; i < M; i++){
                    for(int j = 0; j < N; j++){
                        printf("%.0f\t", A[i][j]);
                    }
                    printf("\n");
                }
                printf("\nb^T =\t\t[\t");
                for(int i = 0; i < M; i++){
                    printf("%.1f\t", b[i][0]);
                }
                printf("]\n");
        }
        mu = initial_mu;

        fill_lamnda(lambda, M, initial_lambda, 0);

        double* A_lin_per_col = new double[M * N];
        for(int j = 0; j < N; j++){
            for(int i = 0; i < M; i++){
                A_lin_per_col[i*N + j] = A[i][j];
            }
        }

        //linarizzo Q considerando che è triangolare superiore con diagonal principale
        double* Q_lin_per_righe = new double[N * (N+1) / 2];
        int pos = 0;
        for(int i = 0; i < N; i++){
            for(int j = i; j < N; j++){
                Q_lin_per_righe[pos++] = Q[i][j];
            }
        }

        double* b_lin = new double[M];
        for(int i = 0; i < M; i++){
            b_lin[i] = b[i][0];
        }

    
        A_Type*       A_gpu;
        Q_Type*       Q_gpu;
        b_Type*       b_gpu;
        bool*         feasible_gpu;
        fx_Type*      fx_gpu; 
        int*          x_min_gpu;
        double*       fx_min_gpu;

        CHECK(cudaMalloc(&A_gpu, M * N * sizeof(A_Type)));
        CHECK(cudaMalloc(&Q_gpu, N * N * sizeof(Q_Type)));
        CHECK(cudaMalloc(&b_gpu, M * sizeof(b_Type)));
        CHECK(cudaMalloc(&feasible_gpu, pow(2,N) * sizeof(bool)));
        CHECK(cudaMalloc(&fx_gpu, pow(2,N) * sizeof(fx_Type)));
        CHECK(cudaMalloc(&x_min_gpu, sizeof(int)));
        CHECK(cudaMalloc(&fx_min_gpu, sizeof(double)));

        CHECK(cudaMemcpy(A_gpu, A_lin_per_col, M * N * sizeof(A_Type), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(Q_gpu, Q_lin_per_righe, N * N * sizeof(Q_Type), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(b_gpu, b_lin, M * sizeof(b_Type), cudaMemcpyHostToDevice));


        dim3 threads_per_block(1024);
	    dim3 blocks_per_grid(pow(2,N-10));          ///RICORDATI DI PARAMETRIZZARE QUESTA ROBA PER N DIVERSI

        brute_force<<<blocks_per_grid, threads_per_block>>>(Q_gpu, A_gpu, b_gpu, N, M, feasible_gpu, fx_gpu);
	    CHECK_KERNELCALL();
	    CHECK(cudaDeviceSynchronize());


        reduce_argmin_feasible<<<blocks_per_grid, threads_per_block>>>(fx_gpu, feasible_gpu, fx_min_gpu, x_min_gpu);

        /*//NB: im skipping the problem if there is no feasible solution. It would be interesting to check if AL realize it.
        if(!find_x_min_brute_force(Q, N, A, M, b, expected_min_x, &true_max_val, &true_min_val, strong_verbose)){
            iter--;
            continue;
        }*/

        int i = 0;
        bool ok;
        bool al_condition;
        do{
            if(strong_verbose){
                printf("AL attempt %d\tmu = %.5f\tlambda^T = [ ", i, mu);
            }
            ok = true;

            find_x_min_AL_brute_force(Q, N, A, M, b, lambda, mu, min_x, c, &al_min_val);
            
            if(strong_verbose){
                for(int i = 0; i < M; i++){
                    printf("%.5f ", lambda[i][0]);
                }
                printf("]\tc_x_opt^T = [ ");
                for(int i = 0; i < M; i++){
                    printf("%.5f ", c[i][0]);
                }
                printf("]\tx_opt = [ ");
                for(int i = 0; i < N; i++){
                    printf("%.0f ", min_x[i][0]);
                }
                printf("]\tmin_val = %.1f\n", al_min_val);
            }

            copy_vector(lambda, old_lambda, M);
            old_mu = mu;
            for(int j=0; j<M; j++){
                if(c[j][0] > 0){
                    lambda[j][0] = lambda[j][0] + mu * c[j][0];               //ORIGINALEEEEEE
                    //lambda[i][0] = lambda[i][0] + rho * c[i][0];

                    ok = false;
                }
            }

            i++;
            //mu = mu * rho;
            //mu = mu + rho;
            mu = update_mu(mu, rho);
            al_condition = al_end_condition(i, N_AL_ATTEMPTS, N, M, lambda, mu, c);
        } while (!ok && al_condition);
        //} while (!check_c(c, M) && al_condition);


        correct = al_condition && ok && calculate_xQx(Q, min_x, N) == true_min_val;
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
            normalized_error_mean += true_max_val-true_min_val != 0 ? (calculate_xQx(Q, min_x, N) - true_min_val) / (true_max_val-true_min_val) : 1;
            //It DOESN'T make sesnse that the error is negative. true_min_val is the minimum feasible value of the function, if AL exit the loop beleiving that a lower minimum (that could exists) fulfils the constraint, there is a problem while checking c(x)
            if(normalized_error_mean < 0){
                printf("ERROR!\ntrue max val : %.1f\t true min val: %.1f\t xQx: %.1f\n", true_max_val, true_min_val, calculate_xQx(Q, min_x, N));
                printf("Q:\n");
                for(int i = 0; i < N; i++){
                    for(int j = 0; j < N; j++){
                        printf("%.1f ", Q[i][j]);
                    }
                    printf("\n");
                }
                printf("\n");
                printf("A = \n");
                for(int i = 0; i < M; i++){
                    for(int j = 0; j < N; j++){
                        printf("%.1f ", A[i][j]);
                    }
                    printf("\n");
                }
                printf("b = \n");
                for(int i = 0; i < M; i++){
                    printf("%.1f ", b[i][0]);
                }
                printf("\n");
                printf("c = \n");
                for(int i = 0; i < M; i++){
                    printf("%.1f ", c[i][0]);
                }
                printf("\n");
                exit(0);

            }
        } else {
            printf("ERROR: something went wrong\n");
            return 0;
        }
       
        // Print progress bar
        /*if(!verbose && !strong_verbose){
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
                printf("Problem solved in %d iterations\n", i);
            } else{
                printf("Problem not solved in %d iterations\n", i);
            }

            if(!strong_verbose){
                printf("c_x^T =\t\t[\t");
                for(int i = 0; i < M; i++){
                    printf("%.1f\t", c[i][0]);
                }
                printf("]\nlambda^T =\t[\t");
                for(int i = 0; i < M; i++){
                    printf("%.1f\t", lambda[i][0]);
                }
                printf("]\nmu =\t\t%.1f\n\n", mu);
            }
        }*/


        if(correct){
            mean_al_attempts_on_correct_solutions += i;
            mean_mu_on_correct_solutions += old_mu;
            for(int j = 0; j < M; j++){
                mean_lambda_on_correct_solutions += old_lambda[j][0]/M;
                if(lambda[j][0] < lambda_min_on_correct_solutions)
                    lambda_min_on_correct_solutions = lambda[j][0];
                if(lambda[j][0] > lambda_max_on_correct_solutions)
                    lambda_max_on_correct_solutions = lambda[j][0];
            }
        }
        else if(unfinished){
            mean_al_attempts_on_unfinished_solutions += i;
            mean_mu_on_unfinished_solutions += old_mu;
            for(int j = 0; j < M; j++){
                mean_lambda_on_unfinished_solutions += old_lambda[j][0]/M;
                if(lambda[j][0] < lambda_min_on_unfinished_solutions)
                    lambda_min_on_unfinished_solutions = lambda[j][0];
                if(lambda[j][0] > lambda_max_on_unfinished_solutions)
                    lambda_max_on_unfinished_solutions = lambda[j][0];
            }
        }
        else if(wrong){
            mean_al_attempts_on_wrong_solutions += i;
            mean_mu_on_wrong_solutions += old_mu;
            for(int j = 0; j < M; j++){
                mean_lambda_on_wrong_solutions += old_lambda[j][0]/M;
                if(lambda[j][0] < lambda_min_on_wrong_solutions)
                    lambda_min_on_wrong_solutions = lambda[j][0];
                if(lambda[j][0] > lambda_max_on_wrong_solutions)
                    lambda_max_on_wrong_solutions = lambda[j][0];
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


    // Deallocate
    for(int i = 0; i < N; ++i) {
        delete [] Q[i];
        delete [] expected_min_x[i];
        delete [] min_x[i];
    }

    for(int i = 0; i < M; ++i) {
        delete [] A[i];
        delete [] b[i];
        delete [] lambda[i];
        delete [] old_lambda[i];
        delete [] c[i];
    }

    delete [] Q;
    delete [] A;
    delete [] b;
    delete [] lambda;
    delete [] old_lambda;
    delete [] expected_min_x;
    delete [] min_x;
    delete [] c;


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

