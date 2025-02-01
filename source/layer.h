#ifndef LAYER_H
#define LAYER_H

#include "vectorops.h"

#include <vector>
#include <string>

/*
* Layer class
* Stores a vector of neurons aka weights
*/
template <typename T>
class Layer {
    public:
        Layer(const int numNeurons, const std::string activationFunction, const std::pair<int, int> shape);
        ~Layer();

    private:
        int m_neurons;
        std::string m_activation;
        std::vector<std::vector<T>> m_weights;
};

template <typename T>
Layer<T>::Layer(const int numNeurons, const std::string activationFunction, const std::pair<int, int> shape):
    m_neurons(numNeurons),  m_activation(activationFunction)
{
    /* init weights */
    uniform_random_initialization<T>(m_weights, shape, -1, 1);
}

template<typename T>
Layer<T>::~Layer() {}

#endif