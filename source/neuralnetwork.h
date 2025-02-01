#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <string>

#include "layer.h"

template <typename T>
class NeuralNetwork {
    public:
        NeuralNetwork(const std::vector<std::pair<int, std::string>> &shape, float learningRate);
        ~NeuralNetwork();

        void train(std::vector<T> input);
        void query(std::vector<T> input);

        void printweights(); 

    private:
        std::vector<Layer<T>> m_layers;
        float m_learningRate;
};

template <typename T>
NeuralNetwork<T>::NeuralNetwork(const std::vector<std::pair<int, std::string>> &shape, float learningRate):
    m_learningRate(learningRate)
{
    /* first layer has no activation */
    if (shape[0].second != "none") {
        std::cerr << "First layer must have no activation" << std::endl;
        throw std::invalid_argument("First layer must have no activation");
    }

    /* atleast two layers are needed */
    if (shape.size() < 2) {
        std::cerr << "Atleast two layers are needed" << std::endl;
        throw std::invalid_argument("Atleast two layers are needed");
    }

    /* init first layer */
    m_layers.push_back(Layer<T>(
        shape[0].first,
        shape[0].second,
        {shape[0].first, shape[0].first}
    ));

    /* init rest of the network */
    for (size_t i = 0; i < shape.size(); i++) {
        m_layers.push_back(Layer<T>(
            shape[i].first,
            shape[i].second,
            {shape[i - 1].first, shape[i].first}
        ));
    }
}

template<typename T>
NeuralNetwork<T>::~NeuralNetwork() {}

template<typename T>
void NeuralNetwork<T>::printweights() {
    for (auto &layer : m_layers) {
        for (auto &row : layer.m_weights) {
            for (auto &r : row) {
                std::cout << r << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

#endif