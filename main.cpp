#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <exception>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <iterator>

#include "neuralnetwork.h"

/* scales integer input to doubles between 0.01 and 1.0 */
double scale_data(int input) {
    double scaled_input = (input / 255.0) * 0.98 + 0.01;
    return scaled_input;
}

std::vector<std::vector<double>> read_csv_data(std::string filepath) {
    std::vector<std::vector<double>> data;

    if (!std::filesystem::exists(filepath)) {
        throw std::runtime_error("csv file not found: " + filepath);
    }

    std::ifstream infile{filepath};
    std::string line;

    while (std::getline(infile, line)) {
        std::vector<double> csv_line;
        std::istringstream iss{line};

        int j = 0;
        /* read every value in the line */
        for (int i; iss >> i;) {
            /* dont scale the first value */
            if (j == 0) {
                csv_line.push_back(i);
            } else {
                csv_line.push_back(scale_data(i));
            }

            if (iss.peek() == ',') {
                iss.ignore();
            }
            ++j;
        }
        data.push_back(csv_line);
    }

    return data;
}

template<typename T>
void print_data(const std::vector<std::vector<T>> data) {
    for (const auto &i: data) {
        std::cout << "Line: ";
        for (const auto &j: i) {
            std::cout << j << " ";
        }
        std::cout << std::endl << std::endl;
    }
}

template<typename T>
void print_vector(const std::vector<T> data) {
    for (const auto &i: data) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

/* gets the input data from the csv data */
std::vector<double> get_input(const std::vector<double> &training_data) {
    std::vector<double> input;
    try {
        auto it = std::next(training_data.begin(), 1);
        input.insert(input.end(), it, training_data.end());
    } catch (const std::out_of_range &err) {
        std::cout << "Out of range err: " << err.what() << std::endl;
        throw std::runtime_error("Err get input");
    } 

    return input;
}

/* gets the target of a specific training data entry */
std::vector<double> get_targets(const std::vector<double> &training_data, int onodes) {
    std::vector<double> targets(onodes, 0.01);

    /* set the target */
    try {
        targets.at(training_data.at(0)) = 0.99;
    } catch (const std::out_of_range &err) {
        std::cout << "Out of Range: " << err.what() << " tried accessing training data: " << training_data.at(0) << std::endl;
        throw std::runtime_error("Err");
    }
    
    return targets;
}

template<typename T>
void print_test_result(const std::vector<T> data) {
    int outcome = 0;
    for (const auto &i: data) {
        std::cout << std::fixed;
        std::cout << std::setprecision(4);
        std::cout << outcome << " " << i << std::endl;
        ++outcome;
    }
}

template<typename T>
bool check_result(int expected_value, std::vector<T> result) {
    int index = std::distance(result.begin(), std::max_element(result.begin(), result.end()));

    if (index == expected_value) {
        return true;
    }

    return false;
} 

int main(int argc, const char *argv[]) {
    int epochs = atoi(argv[1]);
    double learning_rate = atof(argv[2]);

    const std::vector<int> neurons = {784, 300, 10};

    /* Path to training and test data */
    std::string training_data_file = "./mnist/mnist_train.csv";
    std::string test_data_file = "./mnist/mnist_test.csv";

    Neuralnetwork neuralnet{neurons, learning_rate};

    /* read the training and test data */
    std::cout << "Reading CSV Files" << std::endl;
    std::vector<std::vector<double>> training_data = read_csv_data(training_data_file);
    std::vector<std::vector<double>> test_data = read_csv_data(test_data_file);

    /* train on the training data */
    std::cout << "Training Network with " << epochs << " epochs" << std::endl;
    const auto start{std::chrono::steady_clock::now()};
    for (int i = 0; i < epochs; ++i) {
        for (const auto &data_set: training_data) {
            std::vector<double> targets = get_targets(data_set, neurons.at(neurons.size() - 1));
            std::vector<double> inputs = get_input(data_set);
            
            neuralnet.train(inputs, targets); 
        }
    }
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    std::cout << "Training done, duraction: " << elapsed_seconds << std::endl;

    /* query the network using the test data */
    std::cout << "Testing Network" << std::endl;
    std::vector<int> scoreboard;

    for (const auto &test_set: test_data) {
        int expected_value = test_set.at(0);
        std::vector<double> result = neuralnet.query(get_input(test_set));

        if (check_result(expected_value, result)) {
            scoreboard.push_back(1);
        } else {
            scoreboard.push_back(0);
        }
    }  

    double performance = std::reduce(scoreboard.begin(), scoreboard.end());
    std::cout << "Network Performance: ";
    std::cout << std::fixed;
    std::cout << std::setprecision(4);
    std::cout << performance / scoreboard.size() << std::endl;

    std::exit(EXIT_SUCCESS);
}