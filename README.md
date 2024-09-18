# neuralnet
Neuralnet from the book "Make your own Neuralnetwork" in c++

## Build
clang++ -std=c++20 main.cpp neuralnetwork.cpp -o nn -Wall

## Run
./nn <number of epochs> <index of test data to query the network with>  

## Example Outputs
Training Network with 100 epochs
Training done, duraction: 16.1129s
Querying Neuralnetwork
Expected Value : 1
0 0.0227
1 0.9891 <-
2 0.0077
3 0.0113
4 0.0032
5 0.0072
6 0.0080
7 0.0122
8 0.0102
9 0.0168