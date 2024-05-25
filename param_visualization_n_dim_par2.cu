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

#include "AL_test.cu"


/*--------------------------------------- DEFINES ------------------------------------------------- */

#define CUDA_DEVICE_INDEX 0
#define MAX_N_WITHOUT_COARSENING 29

/*--------------------------------------- GLOBAL VARIABLES ---------------------------------------- */

double MAX_MU = max_val<mu_Type>();
double MAX_LAMBDA = max_val<lambda_Type>();

char name_suffix[20] = "";  
char results_path[100] = "./results/";


bool Q_DIAG = false;
bool Q_ID = false;
bool PCR_PROBLEM = false;


/*--------------------------------------- MAIN ---------------------------------------------------- */

int main(int argc, char** argv) {

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, CUDA_DEVICE_INDEX);

    //default values
    int MIN_N = 1;
    int MAX_N = 20;
    int MIN_M = 1;
    int MAX_M = 20;
    int MAXITER = 1000;
    int N_AL_ATTEMPTS = 1000;
    mu_Type initial_mu = 0.1;
    lambda_Type initial_lambda = 0.1;
    mu_Type rho = 1.1;
    Q_Type PARAM_1_Q = -10;
    Q_Type PARAM_2_Q = 10;
    float PARAM_1_A = -10;
    b_Type PARAM_1_b = -10;
    bool verbose = false;
    bool strong_verbose = false;
    bool only_final_report = false;

    enum fill_policys {identity, diagonal, upper_triangular, manual, negative_binary, all_equals, not_set, fill_policys_end};

    unsigned char Q_fill_policy = not_set;
    unsigned char A_fill_policy = not_set;    
    unsigned char b_fill_policy = not_set;

    char stop_conditions_names_str[stop_conditions_end][20] = {"max_Al_attempts", "max_mu", "max_lambda"};
    int stop_condition_counter = 0;
    bool selected_stop_conditions[3] = {false, false, false};
    bool(*stop_conditions[3])(const int i, const int N_AL_ATTEMPTS, const dim_Type N, const dim_Type M, const lambda_Type* __restrict__ lambda, const mu_Type mu, const b_Type* __restrict__ c) = {max_Al_attempts_condition, &max_mu_condition, &max_lambda_condition};
   
    auto end_condition_mix = [&selected_stop_conditions, &stop_conditions](const int i, const int N_AL_ATTEMPTS, const dim_Type N, const dim_Type M, const lambda_Type* __restrict__ lambda, const mu_Type mu, const b_Type* __restrict__ c) -> bool{
        bool return_val = true;
        for(int j = 0; j < stop_conditions_end; j++){
            if(selected_stop_conditions[j]){
                return_val = return_val && stop_conditions[j](i, N_AL_ATTEMPTS, N, M, lambda, mu, c);
            }
        }
        return return_val;
    };

    std::function<bool(const int i, const int N_AL_ATTEMPTS, const dim_Type N, const dim_Type M, const lambda_Type* __restrict__ lambda, const mu_Type mu, const b_Type* __restrict__ c)> al_end_condition = end_condition_mix;
    void (*fill_Q)(Q_Type* Q, const dim_Type N, const Q_Type lowerbound_or_unused, const Q_Type upperbound_or_unused) = fill_Q_id_lin;
    void (*fill_A)(A_Type* A, const dim_Type M, const dim_Type N, const float one_probability, const b_Type b) = fill_A_neg_binary_lin;
    void (*fill_b)(b_Type* b, const dim_Type M, const b_Type b_val) = fill_b_vector_lin;
    mu_Type (*update_mu)(const mu_Type mu, const mu_Type rho) = update_mu_exp;

    bool computer_test = false;

/*--------------------------------------- INPUT PARAMS RPOCESSING --------------------------------- */


    int opt;
    while ((opt = getopt(argc, argv, "lm:M:N:u:l:i:a:r:n:F:o:P:C:R:b:e::vsdfch::")) != -1) {
        switch (opt) {
            case 'm':
                if      (optarg[0] == 'N') MIN_N            = atoi(optarg+2);
                else if (optarg[0] == 'M') MIN_M            = atoi(optarg+2);
                else if (optarg[0] == 'u') initial_mu       = atof(optarg+2); //>= 1 ? atof(optarg+2) : printf("WARNING: mu must be >= 1. Default value will be used.\n");
                else if (optarg[0] == 'l') initial_lambda   = atof(optarg+2);
                break;
            case 'M':
                if      (optarg[0] == 'N') MAX_N = atoi(optarg+2);
                else if (optarg[0] == 'M') MAX_M = atoi(optarg+2);
                else if (optarg[0] == 'M' && optarg[1] == 'F') {            //MMF
                    PARAM_1_A = atof(optarg+3);     // -1 probability
                    PARAM_1_b = 1;                  // b
                    fill_Q = fill_Q_id_lin;
                    fill_A = fill_A_neg_binary_lin;
                    fill_b = fill_b_vector_lin;
                    Q_fill_policy = identity;
                    A_fill_policy = negative_binary;
                    b_fill_policy = all_equals;
                    Q_DIAG = true;
                    Q_ID = true;
                } else if(optarg[0] == 'u'){
                    MAX_MU = atof(optarg+2);
                    stop_condition_counter++;
                    selected_stop_conditions[max_mu] = true;
                }
                else if(optarg[0] == 'l'){
                    MAX_LAMBDA = atof(optarg+2);
                    stop_condition_counter++;
                    selected_stop_conditions[max_lambda] = true;
                }
                break;
            case 'r':
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
            case 'P':
                if(optarg[0] == 'C' && optarg[1] == 'R')
                    if(optarg[2] == 'b'){                       //PCRb
                        Q_DIAG = true;
                        Q_ID = true;
                        PCR_PROBLEM = true;
                        std::string arg(optarg+4);
                        size_t delimiter_pos = arg.find(",");
                        if (delimiter_pos != std::string::npos) {
                            PARAM_1_A = std::stof(arg.substr(0, delimiter_pos));
                            PARAM_1_b = (b_Type)(-std::stof(arg.substr(delimiter_pos + 1)));
                            fill_Q = fill_Q_id_lin;
                            fill_A = fill_A_neg_binary_lin;
                            fill_b = fill_b_vector_lin;
                            Q_fill_policy = identity;
                            A_fill_policy = negative_binary;
                            b_fill_policy = all_equals;
                        } else {
                            printf("Invalid argument for -PCRb option\n");
                            exit(EXIT_FAILURE);
                        }
                        break;
                    } else if(optarg[2] == 'l'){            //PCRl
                        Q_DIAG = true;
                        Q_ID = false;
                        PCR_PROBLEM = true;
                        std::string arg(optarg+4);
                        size_t delimiter_pos_1 = arg.find(",");
                        size_t delimiter_pos_2 = arg.find(",", delimiter_pos_1 + 1);
                        if (delimiter_pos_1 != std::string::npos) {
                            PARAM_1_A = std::stof(arg.substr(0, delimiter_pos_1));
                            PARAM_1_b = -std::stof(arg.substr(delimiter_pos_1 + 1, delimiter_pos_2));
                            delimiter_pos_1 = arg.find(",", delimiter_pos_2 + 1);
                            PARAM_1_Q = std::stof(arg.substr(delimiter_pos_2 + 1, delimiter_pos_1));
                            PARAM_2_Q = std::stof(arg.substr(delimiter_pos_1 + 1));
                            fill_Q = fill_Q_diag_lin;
                            fill_A = fill_A_neg_binary_lin;
                            fill_b = fill_b_vector_lin;
                            Q_fill_policy = diagonal;
                            A_fill_policy = negative_binary;
                            b_fill_policy = all_equals;
                        } else {
                            printf("Invalid argument for -PCRl option\n");
                            exit(EXIT_FAILURE);
                        }
                        break;
                    } else if(optarg[2] == 'q'){            //PCRq
                        std::string arg(optarg+4);
                        size_t delimiter_pos = arg.find(",");
                        size_t delimiter_pos_1 = arg.find(",");
                        size_t delimiter_pos_2 = arg.find(",", delimiter_pos_1 + 1);
                        if (delimiter_pos != std::string::npos) {
                            PARAM_1_A = std::stof(arg.substr(0, delimiter_pos_1));
                            PARAM_1_b = -std::stof(arg.substr(delimiter_pos_1 + 1, delimiter_pos_2));
                            delimiter_pos_1 = arg.find(",", delimiter_pos_2 + 1);
                            PARAM_1_Q = std::stof(arg.substr(delimiter_pos_2 + 1, delimiter_pos_1));
                            PARAM_2_Q = std::stof(arg.substr(delimiter_pos_1 + 1));
                            fill_Q = fill_Q_upper_trianular_lin;
                            fill_A = fill_A_neg_binary_lin;
                            fill_b = fill_b_vector_lin;
                            Q_fill_policy = upper_triangular;
                            A_fill_policy = negative_binary;
                            b_fill_policy = all_equals;
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
                fill_A = fill_A_manual_lin;
                fill_b = fill_b_manual_lin;
                break;
            default:
                printf("Si un giorno la faccio la lista dei parametri :)\n");
                exit(EXIT_FAILURE);
        }
    }


/*--------------------------------------- CHECKs -------------------------------------------------- */


    if(MIN_N < -PARAM_1_b){
        printf("ERROR: unfeasible problem. N must be >= b\n");
        exit(EXIT_FAILURE);
    }

    if(initial_mu < 1 &&  update_mu == update_mu_exp){
        printf("WARNING: mu must be >= 1 when using exponential update of mu. Default value 1 will be used.\n");
        initial_mu = 1;
    }
    if(rho <= 1 && update_mu == update_mu_exp){
        printf("WARNING: rho must be > 1 when using exponential update of mu. Default value 1.1 will be used.\n");
        initial_mu = 1;
    }


    if(PARAM_1_A < (float)(-PARAM_1_b) / MIN_N){
        PARAM_1_A = (float)(-PARAM_1_b) / MIN_N;
        printf("ERROR: probability to low to ensure a feasible problem. -1's probability will be set to %.1f to satisfy unequalities.\n", PARAM_1_A);        
    }


    MAX_N = MAX_N < MIN_N ? MIN_N : MAX_N;
    MAX_M = MAX_M < MIN_M ? MIN_M : MAX_M;

    if(PCR_PROBLEM && MIN_N < -PARAM_1_b){
        printf("\nWARNING: in PCR problem, N must be >= b, otherwise problems cannot be solved.\nMIN_N will be set accordingly\n\n");
        MIN_N = -PARAM_1_b;
    }

    if( MAX_M * MAX_N * sizeof(A_Type) + MAX_M * sizeof(b_Type) + (Q_DIAG ? MAX_N : MAX_N * (MAX_N + 1) / 2) * sizeof(Q_Type) > prop.totalConstMem){
        printf("ERROR: shared memory size exceeded. Please reduce either N or M.\n");
        exit(EXIT_FAILURE);
    }


    if(stop_condition_counter == 0){
        stop_condition_counter = 1;
        selected_stop_conditions[max_Al_attempts] = true;
    }


/*--------------------------------------- INPUT PARAMS VISUALIZZATION ----------------------------- */

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
    print_file_stdout(file, "b = %f\n", PARAM_1_b);
    print_file_stdout(file, "# problems for each combination of N and M = %d\n", MAXITER);
    print_file_stdout(file, "Initial mu = %f\n", initial_mu);
    print_file_stdout(file, "rho = %f\n", rho);
    print_file_stdout(file, "Initial lambda = %.1f\n", initial_lambda);
    print_file_stdout(file, "Maximum number of AL attempts (if 'PARAM_2_Al_attempts' selected) = %d\n", N_AL_ATTEMPTS);
    print_file_stdout(file, "Maximum value of mu (if 'max_mu' selected) = "); MAX_MU == max_val<mu_Type>() ? print_file_stdout(file, "MAX DOUBLE\n") : print_file_stdout(file, "%.1f\n", MAX_MU);
    print_file_stdout(file, "Maximum value of lambda (if 'max_lambda' selected) = "); MAX_LAMBDA == max_val<lambda_Type>() ? print_file_stdout(file, "MAX DOUBLE\n") : print_file_stdout(file, "%.1f\n", MAX_LAMBDA);
    print_file_stdout(file, "Stop conditions:\n");
    for(int i=0; i<stop_conditions_end; i++){
        if(selected_stop_conditions[i]){
            print_file_stdout(file, "\t%d: %s\n", i+1, stop_conditions_names_str[i]);
        }
    }
    print_file_stdout(file, "MATRIX FILL POLICIES:\n");
    switch(Q_fill_policy){
        case identity:
            print_file_stdout(file, "\tQ matrix: identity matrix\n");
            break;
        case diagonal:
            print_file_stdout(file, "\tQ matrix: diagonal, uniform distribution between %f and %f\n", PARAM_1_Q, PARAM_2_Q);
            break;
        case upper_triangular:
            print_file_stdout(file, "\tQ matrix: upper triangular, uniform distribution between %f and %f\n", PARAM_1_Q, PARAM_2_Q);
            break;
        case manual:
            print_file_stdout(file, "\tQ matrix: manual\n");
            break;
    }
    switch(A_fill_policy){
        case negative_binary:
            print_file_stdout(file, "\tA matrix: binary, -1 with probability %f\n", PARAM_1_A);
            break;
        case manual:
            print_file_stdout(file, "\tA matrix: manual\n");
            break;
    }
    switch(b_fill_policy){
        case all_equals:
            print_file_stdout(file, "\tb vector: all elements equal to %f\n", PARAM_1_b);
            break;
        case manual:
            print_file_stdout(file, "\tb vector: manual\n");
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


/*--------------------------------------- TEST ---------------------------------------------------- */


    int max_n = (MAX_N >= MIN_N ? MAX_N : MIN_N);
    int max_m = (MAX_M >= MIN_M ? MAX_M : MIN_M);

    //initialize vector of results to the size of the number of tests
    std::vector<test_results> results = std::vector<test_results>((max_n - MIN_N + 1) * (max_m - MIN_M + 1));

    for(dim_Type n = MIN_N; n <= max_n; n++){
        for(dim_Type m = MIN_M; m <= max_m; m++){

            test_results current_results;
            if(n <= MAX_N_WITHOUT_COARSENING){
                if( 0 == test_at_dimension_coarsening(0, n, m, MAXITER, N_AL_ATTEMPTS, initial_mu, initial_lambda, rho, fill_Q, PARAM_1_Q, PARAM_2_Q, fill_A, PARAM_1_A, fill_b, PARAM_1_b, al_end_condition, update_mu, &current_results, verbose, strong_verbose)){
                    finalize(results);
                }
            } else {
                if( 0 == test_at_dimension_coarsening(n - MAX_N_WITHOUT_COARSENING + 1, n, m, MAXITER, N_AL_ATTEMPTS, initial_mu, initial_lambda, rho, fill_Q, PARAM_1_Q, PARAM_2_Q, fill_A, PARAM_1_A, fill_b, PARAM_1_b, al_end_condition, update_mu, &current_results, verbose, strong_verbose)){
                    finalize(results);
                }
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


/*--------------------------------------- SUMMARY DATA -------------------------------------------- */


    test_results summary = {};
    summary.lambda_min_on_correct_solutions     = max_val<lambda_Type>();
    summary.lambda_min_on_unfinished_solutions  = max_val<lambda_Type>();
    summary.lambda_min_on_wrong_solutions       = max_val<lambda_Type>();
    summary.lambda_max_on_correct_solutions     = -max_val<lambda_Type>();
    summary.lambda_max_on_unfinished_solutions  = -max_val<lambda_Type>();
    summary.lambda_max_on_wrong_solutions       = -max_val<lambda_Type>();
    int tot_tests = results.size();
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


/*--------------------------------------- FILE SAVING --------------------------------------------- */


    //finalize(results);
    //finalize(summary);

}




