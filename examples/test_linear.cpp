#include <iostream>
#include "modules/layers/linear.hpp"
#include "modules/losses/mse.hpp"
#include "modules/losses/cross_entropy.hpp"
using namespace nn;

int main() {
        const bool bias = true;

    Linear linear_1(3, 5, bias);

    cout << "Before initialization: " << endl;
    linear_1.getWeights().print();

    Linear linear_2(5, 7, bias);

    Tensor<> specific_weights_1 = {
        {0.1, 0.4, 0.7, 1.0, 1.3},
        {0.2, 0.5, 0.8, 1.1, 1.4},
        {0.3, 0.6, 0.9, 1.2, 1.5}
    };
    
    Tensor<> specific_weights_2 = {
        {0.1, 0.6, 1.1, 1.6, 2.1, 2.6, 3.1},
        {0.2, 0.7, 1.2, 1.7, 2.2, 2.7, 3.2},
        {0.3, 0.8, 1.3, 1.8, 2.3, 2.8, 3.3},
        {0.4, 0.9, 1.4, 1.9, 2.4, 2.9, 3.4},
        {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5}
    };

    Tensor<> specific_bias_1 = {
        0.1,
        0.2,
        0.3,
        0.4,
        0.5
    };

    Tensor<> specific_bias_2 = {
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7
    };

    Tensor<> input = {
        {1.1f, 2.1f, 3.1f},
        {4.1f, 5.1f, 6.1f},
        {7.1f, 8.1f, 9.1f},
        {10.1f, 11.1f, 12.1f}
    };

    cout << "After initialization: " << endl;

    specific_bias_1 = specific_bias_1.transpose();
    specific_bias_2 = specific_bias_2.transpose();

    cout << "bias 1: " << endl;
    specific_bias_1.print();

    cout << "bias 2: " << endl;
    specific_bias_2.print();

    linear_1.setWeights(specific_weights_1);
    linear_2.setWeights(specific_weights_2);

    linear_1.setBiases(specific_bias_1);
    linear_2.setBiases(specific_bias_2);

    cout << endl;

    Tensor<> output_1 = linear_1(input);
    Tensor<> Y_hat = linear_2(output_1);

    cout << "Y_hat: " << endl;
    Y_hat.print();

    cout << endl;

    Tensor<> Y = {
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
        {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0}
    };

    MSE mse;
    CrossEntropyLoss cross_entropy_loss;

    const float mse_loss = mse.forward(Y, Y_hat);
    const float cross_entropy_loss_loss = cross_entropy_loss.forward(Y, Y_hat);

    cout << "Cross Entropy Loss: " << cross_entropy_loss_loss << endl;

    Tensor<> dL_dZ = cross_entropy_loss.backward();
    Tensor<> dL_dY = linear_2.backward(dL_dZ);
    Tensor<> dL_dX = linear_1.backward(dL_dY);




    // ===================softmax=====================

    // Softmax softmax;

    // Tensor<> X_1d = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // Tensor<> Y_hat_softmax_1d = softmax.forward(X_1d);

    // Y_hat_softmax_1d.print();

    // Tensor<> X_2d = {
    //     {1.0f, 2.0f, 3.0f},
    //     {4.0f, 2.0f, 8.0f},
    //     {1.0f, 8.0f, 3.0f}
    // };

    // Tensor<> Y_hat_softmax_2d = softmax.forward(X_2d);

    // Y_hat_softmax_2d.print();

    return 0;
}
