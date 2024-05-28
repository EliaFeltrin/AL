#include <iostream>
#include <cmath>


int main(){
    uint32_t a = 0xFFFFFFFF;

    std::cout << a << std::endl;

    std::cout << (a >> 32) << std::endl;

    a = 0b1111;

    std::cout << a << std::endl;

    std::cout << (a >> 4) << std::endl;
}