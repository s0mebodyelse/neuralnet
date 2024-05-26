#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <exception>
#include <iomanip>

#include "neuralnetwork.h"

/* scales integer input to doubles between 0.01 and 1.0 */
double scale_data(int input) {
    double scaled_input = (input / 255.0) * 0.98 + 0.01;
    return scaled_input;
}

std::vector<std::vector<double>> read_csv_data(std::string filepath) {
    std::vector<std::vector<double>> data;

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
    
    auto it = std::next(training_data.begin(), 1);
    input.insert(input.end(), it, training_data.end());

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

int main(int argc, const char *argv[]) {
    /* number of nodes in the layers */
    int input_nodes = 784;
    int hidden_nodes = 100;
    int output_nodes = 10;
    
    int rounds = atoi(argv[1]);
    int test_data_set = atoi(argv[2]);

    /* Path to training and test data */
    std::string training_data_file = "./mnist_dataset/mnist_train.csv";
    std::string test_data_file = "./mnist_dataset/mnist_test.csv";

    /* initialize the neuralnetwork */
    Neuralnetwork neuralnet{input_nodes, hidden_nodes, output_nodes, 0.3};

    /* read the training and test data */
    std::vector<std::vector<double>> training_data = read_csv_data(training_data_file);
    std::vector<std::vector<double>> test_data = read_csv_data(test_data_file);

    /* train on the training data */
    std::cout << "Training Network on " << rounds << " Datasets" << std::endl;
    int i = 0;
    for (const auto &data_set: training_data) {
        if (i == rounds) {
            break;
        }
        std::vector<double> targets = get_targets(data_set, output_nodes);
        std::vector<double> inputs = get_input(data_set);

        neuralnet.train(inputs, targets); 
        ++i;
    }

    /* query the network using the test data */
    std::vector<double> test_input = get_input(test_data.at(test_data_set));
    std::cout << "Querying Neuralnetwork" << std::endl;
    std::cout << "Expected Value : " << test_data.at(test_data_set).at(0) << std::endl;
    std::vector<double> test_result = neuralnet.query(test_input);

    print_test_result(test_result);

    return 0;
}