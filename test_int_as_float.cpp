#include <random>
#include <iostream>
#include <float.h>


int main(){
    unsigned long long int a = 1;
    printf("%b\n", -1);

    // Create a random device and generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Define the range of the random numbers
    std::uniform_real_distribution<double> distr(-1000, 1000);

    for(int i = 0; i<0; i++){
    // Extract two random integer values
        double* num1 = new double;
        *num1 = distr(gen);
        double* num2 = new double;
        *num2 = distr(gen);

        double aux;
        *num1 < *num2 ? aux = *num2, *num2 = *num1, *num1 = aux : 1==1;
        
        bool same_sign = (*num1 < 0 && *num2 < 0) || (*num1 >= 0 && *num2 >= 0);

        unsigned long long int num1_int = *(unsigned long long*)num1;
        unsigned long long int num2_int = *(unsigned long long*)num2;

        std::cout << *num1 << " < " << *num2 << std::endl;
        std::cout << num1_int << " < " << num2_int << std::endl;

        if((*(unsigned long long*)num1 < *(unsigned long long*)num2 && same_sign) || (*(unsigned long long*)num1 > *(unsigned long long*)num2 && !same_sign))
            std::cout << "ERROR";
        else {}
            //std::cout << "OK";

        std::cout << std::endl;

    }



    return 0;
}