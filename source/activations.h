#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <cmath>

namespace activations {
    template <typename T>
    T sigmoid(T x) {
        return 1 / (1 + exp(-x));
    }

    template <typename T>
    T relu(T x) {
        return x > 0 ? x : 0;
    }

    template <typename T>
    T tanh(T x) {
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    }
}

#endif