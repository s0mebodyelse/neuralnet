#ifndef VECTOROPS_H
#define VECTOROPS_H

/*
*   Vector operations bases on std::vector<T>
*   Matrix operations based on std::vector<std::vector<T>>
*/

#include <vector>
#include <chrono>
#include <random>

template <typename T>
void uniform_random_initialization (
    std::vector<std::vector<T>> &A,
    const std::pair<size_t, size_t> &shape,
    const T &low, const T &high
){
    A.clear();  
    /* Uniform distribution in range [low, high] */
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<T> distribution(low, high);
    for (size_t i = 0; i < shape.first; i++) {  
        std::vector<T> row;  
        row.resize(shape.second);
        for (auto &r : row) {             
            r = distribution(generator);  
        }
        A.push_back(row);  
    }
    return;
}

#endif