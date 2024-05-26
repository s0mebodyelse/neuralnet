#include "neuralnetwork.h"

Neuralnetwork::Neuralnetwork(
    int inputnodes, 
    int hiddennodes,
    int outputnodes,
    double learningrate 
):  
    inodes(inputnodes),
    hnodes(hiddennodes), 
    onodes(outputnodes),
    learningrate(learningrate)
{
    init_weights();
}

void Neuralnetwork::train(const std::vector<double> &inputs, const std::vector<double> &targets) {
    try {
        /* feed forward the signal */    
        std::vector<double> hidden_outputs = multiply_2dim_times_1dim_array(
            wih,
            inputs
        );

        /* apply sigmoid function */
        std::transform(hidden_outputs.begin(), hidden_outputs.end(), hidden_outputs.begin(), sigmoid);

        /* calculate final outputs */
        std::vector<double> final_outputs = multiply_2dim_times_1dim_array(
            who,
            hidden_outputs
        );
        std::transform(final_outputs.begin(), final_outputs.end(), final_outputs.begin(), sigmoid);

        /* get error from targets */
        std::vector<double> outputs_error = calculate_error(targets, final_outputs);

        /* 
        *   back propagate the error to the hidden layer 
        *   erros_hidden = WhoT * output_error
        */
        std::vector<double> error_hidden;

        /* transform Weights */
        std::vector<std::vector<double>> transposed_who = transpose_matrix(who);
        error_hidden = multiply_2dim_times_1dim_array(transposed_who, outputs_error);

        /* update weights between hidden and final output */
        who = update_weights(who, outputs_error, hidden_outputs, final_outputs);    

        /* update weights between input and hidden */
        wih = update_weights(wih, error_hidden, inputs, hidden_outputs);    
    } catch (std::exception &err) {
        throw std::runtime_error("Error training network");
    }
} 

/* takes input and returns the output of the network */
std::vector<double> Neuralnetwork::query(const std::vector<double> &inputs) {
    std::vector<double> final_outputs;

    try {
        /* weights times input, 2dim * 1dim vector */    
        std::vector<double> hidden_outputs = multiply_2dim_times_1dim_array(
            wih,
            inputs
        );

        /* apply sigmoid function */
        std::transform(hidden_outputs.begin(), hidden_outputs.end(), hidden_outputs.begin(), sigmoid);

        /* hidden outputs is then used a input to output layer */
        final_outputs = multiply_2dim_times_1dim_array(
            who,
            hidden_outputs
        );
        
        std::transform(final_outputs.begin(), final_outputs.end(), final_outputs.begin(), sigmoid);

    } catch (std::exception &err) {
        throw std::runtime_error("Error querying network");
    }

    return final_outputs;
} 

void Neuralnetwork::print_weights() {
    std::cout << "WIH: " << std::endl;
    for(const auto &i: wih) {
        for (const auto &j: i) {
            std::cout << std::fixed;
            std::cout << std::setprecision(4);
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "WHO: " << std::endl;
    for(const auto &i: who) {
        for (const auto &j: i) {
            std::cout << std::fixed;
            std::cout << std::setprecision(4);
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }
}

double Neuralnetwork::sigmoid(double x) {
    double result = 0.0;
    result = 1 / (1 + exp(-x));
    return result;
}

/* init weights with random numbers */
void Neuralnetwork::init_weights() {
    std::random_device dev;
    /* seed using random device, 1 used for testing */
    std::mt19937 rng(dev());
    /* random number distribution */
    std::normal_distribution<double> unif(0.0, std::pow(hnodes, -0.5));

    /* init weights between hidden and input layer */
    for (int i = 0; i < hnodes; ++i) {
        std::vector<double> random_weights;
        for (int j = 0; j < inodes; ++j) {
            double ran_num = unif(rng);
            random_weights.push_back(ran_num);
            if (ran_num > 1.0) {
                std::cout << "Error initiating weight, above 1.0: " << ran_num << std::endl;
            }
        }

        wih.push_back(random_weights);
    }

    std::normal_distribution<double> normd(0.0, std::pow(inodes, -0.5));
    /* init weights between hidden layer and output layer */
    for (int i = 0; i < onodes; ++i) {
        std::vector<double> random_weights;
        for (int j = 0; j < hnodes; ++j) {
            double ran_num = normd(rng);
            random_weights.push_back(ran_num);
            if (ran_num > 1.0) {
                std::cout << "Error initiating weight, above 1.0: " << ran_num << std::endl;
            }
        }

        who.push_back(random_weights);
    }
}

std::vector<double> Neuralnetwork::calculate_error(
    const std::vector<double> &target, 
    const std::vector<double> &actual
) {

    if (actual.size() != target.size()) {
        throw std::runtime_error("Cant calculate the error, size of output is unequal to size of target");
    }

    std::vector<double> error;
    error.reserve(target.size());

    try {
        for (std::size_t i = 0; i < target.size(); ++i) {
            double err = -(target.at(i) - actual.at(i));
            error.push_back(err);
        }
    } catch (const std::out_of_range &err) {
        throw std::runtime_error("Out of range Error calculating error");
    }
    
    return error;
}

std::vector<std::vector<double>> Neuralnetwork::update_weights(
    std::vector<std::vector<double>> &weights_jk,
    const std::vector<double> &error,
    const std::vector<double> &output_j,
    const std::vector<double> &output_k
) {
    /* check dimensions */
    if (weights_jk.empty() || weights_jk[0].empty() || error.empty() || output_j.empty() || output_k.empty()) {
        throw std::invalid_argument("Empty Parameter during updating weights");
    }

    std::size_t weights_rows = weights_jk.size();
    std::size_t weights_cols = weights_jk[0].size();

    if (error.size() != weights_rows || output_j.size() != weights_cols || output_k.size() != weights_rows) {
        throw std::invalid_argument("Dimensions dont fit to update the weights");
    }

    std::vector<std::vector<double>> updated_weights(weights_rows, std::vector<double>(weights_cols));

    /* k= rows, j = columns */
    for (int k = 0; k < weights_rows; ++k) {
        for (int j = 0; j < weights_cols; ++j) {
            /* calculate the need change in the weight */
            double change_w = error[k] * output_k[k] * (1.0 - (output_k[k])) * output_j[j];
            /* calculate the new weight */
            updated_weights[k][j] = weights_jk[k][j] - (learningrate * change_w);
        }
    }

    return updated_weights;
}

/* Matrix times vector function */
std::vector<double> Neuralnetwork::multiply_2dim_times_1dim_array(
    const std::vector<std::vector<double>> &matrix,
    const std::vector<double> &vector
) {
    /* check dimension and compatibility */
    if (matrix.empty() || vector.empty() || matrix[0].size() != vector.size()) {
        throw std::invalid_argument("Matrix and Vector are not compatible for multiplication");
    }

    /* dimension of the matrix */
    std::size_t m_rows = matrix.size();
    std::size_t m_cols = matrix.at(0).size();

    /* result vector */
    std::vector<double> dot_product(m_rows, 0.0);

    /* Sum(Mij * Vj) */
    for (std::size_t i = 0; i < m_rows; ++i) {
        for (int j = 0; j < m_cols; ++j) {
            dot_product[i] += matrix[i][j] * vector[j];
        } 
    }

    return dot_product;
}

std::vector<std::vector<double>> Neuralnetwork::transpose_matrix(
    const std::vector<std::vector<double>> &matrix
) {
    /* check dimensions */
    if (matrix.empty() || matrix[0].empty()) {
        throw std::invalid_argument("Error transposing Matrix: Matrix is empty");
    }

    std::size_t m_rows = matrix.size();
    std::size_t m_cols = matrix[0].size();
    /* create new empty transpose matrix */
    std::vector<std::vector<double>> transposed_matrix(m_cols, std::vector<double>(m_rows));
      
    for (std::size_t i = 0; i < m_rows; ++i) {
        for (int j = 0; j < m_cols; ++j) {
            transposed_matrix[j][i] = matrix[i][j];
        }
    }

    return transposed_matrix;
}


void Neuralnetwork::print_matrix(const std::vector<std::vector<double>> &matrix) {
    for (const auto &i: matrix) {
        for (const auto &j: i) {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }
}

void Neuralnetwork::print_vector(const std::vector<double> &vector) {
    for (const auto &i: vector) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}