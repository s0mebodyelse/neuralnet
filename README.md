# neuralnet
Neuralnet from the book "Make your own Neuralnetwork" in c++

## Build
clang++ -std=c++20 main.cpp neuralnetwork.cpp -o nn -Wall

## Run
./nn <number of epochs> <learning_rate>  
 ./nn 1 0.3

# Example output
./nn 6 0.1 
Reading CSV Files  
Training Network with 6 epochs  
Training done, duraction: 83.44s  
Testing Network  
Performance: 0.9615  

## Example Outputs mnist_train
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

## Example Outputs mnist full
Training Network with 1 epochs
Training done, duraction: 157.798s
Querying Neuralnetwork
Expected Value : 2
0 0.0000
1 0.0056
2 0.9844 <-
3 0.0081
4 0.0000
5 0.0032
6 0.0009
7 0.0000
8 0.0000
9 0.0005


### Optimized build -O3
Training Network with 1 epochs
Training done, duraction: 13.8999s
Querying Neuralnetwork
Expected Value : 2
0 0.0023
1 0.0006
2 0.9963 <-
3 0.0015
4 0.0000
5 0.0004
6 0.0041
7 0.0004
8 0.0000
9 0.0000