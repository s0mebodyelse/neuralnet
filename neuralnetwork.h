#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <sstream>
#include <iomanip>

class Neuralnetwork {
    public:
        Neuralnetwork(
            int inputnodes, 
            int hiddennodes,
            int outputnodes,
            double learningrate
        );

        void train(const std::vector<double> &inputs, const std::vector<double> &targets);
        std::vector<double> query(const std::vector<double> &inputs);

        void print_weights();

        static double sigmoid(double x);

    private:
        int inodes;
        int hnodes;
        int onodes;
        double learningrate;

        /* weights between nodes */
        std::vector<std::vector<double>> wih;
        std::vector<std::vector<double>> who;

        /* initializes the weights with random numbers betwenn 0.0 and 1.0 */        
        void init_weights();

        /* 
        *   E = -(Tk - Ok)
        *   E is the error
        *   TK is the the target at index k 
        *   Ok is the actual output at index k
        */
        std::vector<double> calculate_error(
            const std::vector<double> &target,
            const std::vector<double> &actual
        );

        /* 
        *    Calculating the Change of the weights, aka the learning, between the layer J and K
        *
        *    dWjk = Ek * Ok * (1 - Ok) * Oj 
        * 
        *    dWjk = the change of the weights connecting Node J (previous layer) with Node K (preceeding layer)
        *    Ek = Error of Node K, could be back propagated error or actual error if k is output layer
        *    Ok = Sigomid(Sum(Wjk * Oj)), aka output of layer k
        *    Oj = Output Node J, needs to be transposed 
        *     
        *    When we update W11, we calculate the change dWjk and then subtract the Change to current weight, a can be used as a moderator (learning rate)
        *    newWjk = oldWjk - a * dWjk
        * 
        *    a = learning rate, constant to moderate the learning, could be 0.1
        *
        *   This calculation is run in a loop over every weight, connecting the two layers    
        */
        std::vector<std::vector<double>> update_weights(
            std::vector<std::vector<double>> &weights_jk,
            const std::vector<double> &error,
            const std::vector<double> &output_j,
            const std::vector<double> &output_k
        );


        /* Matrix multiplication helper functions */
        std::vector<double> multiply_2dim_times_1dim_array(
            const std::vector<std::vector<double>> &matrix,
            const std::vector<double> &vector
        );

        std::vector<std::vector<double>> transpose_matrix(
            const std::vector<std::vector<double>> &matrix
        );

        void print_matrix(const std::vector<std::vector<double>> &matrix);
        void print_vector(const std::vector<double> &vector);
};