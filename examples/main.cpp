#include <iostream>
#include "mlp.cpp"
#include "mnist.hpp"
#include "cross_entropy.hpp"
using namespace nn;

int main() {

    MNIST mnist_dataset;

    // load MNIST data
    Tensor<> images = mnist_dataset.readImages("../data/mnist/train-images.idx3-ubyte");
    Tensor<int> original_labels = mnist_dataset.readLabels("../data/mnist/train-labels.idx1-ubyte");

    // convert the label to one hot vectors
    Tensor<int> labels({original_labels.shapes()[0], 10}, 0);
    for(int i = 0; i < original_labels.shapes()[0]; i++) {
        labels[i, original_labels[i]] = 1;
    }

    cout << "images dimensions : " << images.ndim() << endl;

    cout << "images shapes : " << images.shapes()[0] << " x " << images.shapes()[1] << endl;
    cout << "labels shapes : " << labels.shapes()[0] << " x " << labels.shapes()[1] << endl;

    // Now images are normalized with mean=0.1307 and std=0.3081
    // Each pixel value is transformed using: (x/255 - mean) / std

    // Initialize the model
    MLP model = MLP({784, 128, 64, 10});

    // Define the loss function
    CrossEntropyLoss loss = CrossEntropyLoss();

    // Define the hyperparameters
    double learning_rate = 0.01;
    double epoch = 10;
    double batch_size = 64;

    // Train the model
    for(int i = 0; i < epoch; i++) {
    }

    return 0;
}
