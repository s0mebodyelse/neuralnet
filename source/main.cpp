#include <iostream>

#include "neuralnetwork.h"

int main(int argc, const char *argv[]) {
    std::cout << "Hello World!" << std::endl;

    NeuralNetwork nn = NeuralNetwork<double>({{2, "none"}, {3, "sigmoid"}, {1, "sigmoid"}}, 0.3);
    nn.printweights();

    return 0;
}