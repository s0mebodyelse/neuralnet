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
Performance: 0.9615%   
