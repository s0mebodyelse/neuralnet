#include "neuralnetwork.h"

Neuralnetwork::Neuralnetwork(
    std::vector<int> neurons,
    double learningrate,
    int thread_num
): learningrate(learningrate), thread_num(thread_num) {
    /* create and init weights, first layer is skipped */
    for (std::size_t i = 1; i < neurons.size(); ++i) {
        /* create empty vector */
        std::vector<std::vector<double>> weight;
        std::pair<std::size_t, std::size_t> shape{neurons.at(i), neurons.at(i - 1)};
        uniform_random_initialization(weight, shape, -1.0, 1.0);
        weights.push_back(weight);
    }
}

void Neuralnetwork::train(const std::vector<double> &inputs, const std::vector<double> &targets) {
    std::vector<std::vector<double>> outputs;
    std::vector<double> error;
    std::vector<double> input = inputs;

    /* feed forward signal and save the output of every layer */
    for (auto &weight: weights) {
        std::vector<double> output = multiply_2dim_times_1dim_array(weight, input);
        std::transform(output.begin(), output.end(), output.begin(), sigmoid);
        input = output;
        outputs.push_back(output);
    }

    /* 
    *   backpropagate the error, begining with the weights between Output layer and last hidden layer
    *   On a 3 Layer Network the next loop will only make 1 iteration
    *   first calculate the final output error, the hidden error, then input error
    */
    error = calculate_error(targets, outputs.at(weights.size() - 1));
    for (std::size_t i = weights.size() - 1; i > 0; --i) {
        /* then update the weights based on the error */
        weights.at(i) = update_weights(weights.at(i), error, outputs.at(i - 1), outputs.at(i));

        /* calculate error for the every hidden layer and update the weights in the next iteration */
        std::vector<std::vector<double>> weightT = transpose_matrix(weights.at(i));
        error = multiply_2dim_times_1dim_array(weightT, error);
    }

    /* 
    *   Now every hidden layer is updated 
    *   last step is to update weights between input and first hidden layer 
    */
    weights.at(0) = update_weights(weights[0], error, inputs, outputs[0]);
}

std::vector<double> Neuralnetwork::query(const std::vector<double> &inputs) {
    std::vector<double> output;
    std::vector<double> input = inputs;

    for (auto &weight: weights) {
        /* get output */
        output = multiply_2dim_times_1dim_array(weight, input);
        /* apply activation */
        std::transform(output.begin(), output.end(), output.begin(), sigmoid);
        input = output;
    }

    return output;
}
        

double Neuralnetwork::sigmoid(double x) {
    double result = 0.0;
    result = 1 / (1 + exp(-x));
    return result;
}

template<typename T>
void Neuralnetwork::uniform_random_initialization(
    std::vector<std::vector<T>> &weights,
    const std::pair<std::size_t, std::size_t> &shape,
    const T &low, const T &high
) {
    weights.clear();

    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<T> distribution(low, high);

    for (size_t i = 0; i < shape.first; i++) { 
        std::vector<T> row;  
        row.resize(shape.second);
        for (auto &r : row) {            
            r = distribution(generator);  
        }
        weights.push_back(row);  
    }
    return;
}

std::vector<double> Neuralnetwork::calculate_error(
    const std::vector<double> &target, 
    const std::vector<double> &actual
) {
    const auto func_start{std::chrono::steady_clock::now()};
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
    
    const auto func_end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> func_seconds{
        func_end - func_start };
    countFunctionCall("calculate_error", func_seconds);

    return error;
}

std::vector<std::vector<double>> Neuralnetwork::update_weights(
    std::vector<std::vector<double>> &weights_jk,
    const std::vector<double> &error,
    const std::vector<double> &output_j,
    const std::vector<double> &output_k
) {
    const auto func_start{std::chrono::steady_clock::now()};
    /* check dimensions */
    if (weights_jk.empty() || weights_jk[0].empty() || error.empty() || output_j.empty() || output_k.empty()) {
        throw std::invalid_argument("Empty Parameter during updating weights");
    }

    std::size_t weights_rows = weights_jk.size();
    std::size_t weights_cols = weights_jk[0].size();

    if (error.size() != weights_rows || output_j.size() != weights_cols || output_k.size() != weights_rows) {
        std::cerr << "Error: " << error.size() << " Weights rows: " << weights_rows << std::endl;
        std::cerr << "output_j: " << output_j.size() << " Weights cols: " << weights_cols << std::endl;
        std::cerr << "output_k: " << output_k.size() << " Weights rows: " << weights_rows << std::endl;
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

    const auto func_end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> func_seconds{
        func_end - func_start };
    countFunctionCall("update_weights", func_seconds);

    return updated_weights;
}

/* Matrix Vector product */
std::vector<double> Neuralnetwork::multiply_2dim_times_1dim_array_old(
    const std::vector<std::vector<double>> &matrix,
    const std::vector<double> &vector
) {
    const auto func_start{std::chrono::steady_clock::now()};

    /* check dimension and compatibility */
    if (matrix.empty() || vector.empty() || matrix[0].size() != vector.size()) {
        std::cerr << "Matrix and Vector are not compatible for multiplication" << std::endl;
        std::exit(EXIT_FAILURE);
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

    const auto func_end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> func_seconds{
        func_end - func_start };
    countFunctionCall("multiply_2dim_times_1dim_array_old", func_seconds);

    return dot_product;
}

/* Matrix Vector product using threads */
std::vector<double> Neuralnetwork::multiply_2dim_times_1dim_array(
    const std::vector<std::vector<double>> &matrix,
    const std::vector<double> &vector
) {
    const auto func_start{std::chrono::steady_clock::now()};

    if (thread_num == 0) {
        return multiply_2dim_times_1dim_array_old(matrix, vector);
    }

    /* check dimension and compatibility */
    if (matrix.empty() || vector.empty() || 
            matrix[0].size() != vector.size()
    ) {
        std::cerr << "Matrix and Vector are not compatible for multiplication" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    /* dimension of the matrix */
    std::size_t m_rows = matrix.size();
    std::size_t m_cols = matrix.at(0).size();

    /* result vector */
    std::vector<double> dot_product(m_rows, 0.0);
    
    /* split the work by the matrix rows */
    int work_per_thread = m_rows / thread_num;

    for (std::size_t i = 0; i < thread_num; ++i) {
        /* define the start and end index for every thread */
        int start_index = i * work_per_thread;
        int end_index = (i == thread_num - 1) ? m_rows: 
            (i + 1) * work_per_thread;

        /* run threads */
        threads.emplace_back([
            &matrix, &vector, &dot_product, &m_cols,
            start_index, end_index
        ]() {
            /* Sum(Mij * Vj) */
            for (std::size_t i = start_index; i < end_index; ++i) {
                for (int j = 0; j < m_cols; ++j) {
                    dot_product[i] += matrix[i][j] * vector[j];
                } 
            }
        });
    }

    for (std::thread &thread: threads) {
        thread.join();
    }
    /* clear threads for the next call */
    threads.clear();

    const auto func_end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> func_seconds{
        func_end - func_start };
    countFunctionCall("multiply_2dim_times_1dim_array", func_seconds);

    return dot_product;
}

std::vector<std::vector<double>> Neuralnetwork::transpose_matrix(
    const std::vector<std::vector<double>> &matrix
) {
    const auto func_start{std::chrono::steady_clock::now()};
    
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

    const auto func_end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> func_seconds{
        func_end - func_start };
    countFunctionCall("transpose_matrix", func_seconds);

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

void Neuralnetwork::printPerfmon() {
    std::cout << "Performance Analysis" << std::endl;

   for (auto &func: perfmon) {
       std::cout << func.first << " called ";
       std::cout << func.second.first << " times ";
       std::cout << func.second.second << " total seconds";
       std::cout << std::endl;
   } 
}

void Neuralnetwork::countFunctionCall(std::string func_name,
        std::chrono::duration<double> time) {
    int counter = 1;

    if (perfmon.contains(func_name)) {
        /* just update if the entry already exists */
        perfmon.at(func_name).first++;
        perfmon.at(func_name).second += time;
        return;
    }
    /* insert new entry with counter 1 and time */
    std::pair<int, std::chrono::duration<double>> p{counter, time};
    perfmon.insert(std::make_pair(func_name, p));
}
