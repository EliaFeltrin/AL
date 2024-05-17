#include <random>
#include <iostream>
#include <cuda_runtime.h>

int main(){
    // Create a random device and generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Define the range of the random numbers
    std::uniform_int_distribution<> distr(1, 100);

    // Extract two random integer values
    int num1 = distr(gen);
    int num2 = distr(gen);

    int aux;
    num1 < num2 ? aux = num2, num2 = num1, num1 = aux : num1 = num1;

    if(__int_as_float(num1) < __int_as_float(num2)){
        std::cout << "error" << std::endl;
    }

    return 0;
}