# neuralnet
Neuralnet from the book "Make your own Neuralnetwork" in c++

## Build
clang++ -std=c++20 main.cpp neuralnetwork.cpp -o nn -Wall

## Run
-> if number of threads = 0, no threads will be used

./nn <number of epochs> <learning_rate> <num_of_threads>
 ./nn 1 0.3 2

# Example output on the full MNIST Dataset
/nn 5 0.3
Reading CSV Files
Training Network with 5 epochs
Epoch 0 done, duraction: 13.5947s
Epoch 1 done, duraction: 13.5288s
Epoch 2 done, duraction: 13.4242s
Epoch 3 done, duraction: 13.4218s
Epoch 4 done, duraction: 13.4603s
Training done, duraction: 67.4299s
Testing Network
Network Performance (percentage of correct output): 0.9396
