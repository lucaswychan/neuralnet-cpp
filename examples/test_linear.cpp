#include <iostream>
#include "modules/layers/linear.hpp"
#include "modules/losses/mse.hpp"
using namespace nn;

int main() {
    const bool bias = true;

    Linear linear_1(3, 5, bias);

    cout << "Before initialization: " << endl;
    linear_1.getWeights().print();

    Linear linear_2(5, 2, bias);

    Tensor<> specific_weights_1 = {
        {1.1f, 4.1f, 7.1f, 10.1f, 13.1f},
        {2.1f, 5.1f, 8.1f, 11.1f, 14.1f},
        {3.1f, 6.1f, 9.1f, 12.1f, 15.1f}
    };

    Tensor<> input = {
        {1.1f, 2.1f, 3.1f},
        {4.1f, 5.1f, 6.1f},
        {7.1f, 8.1f, 9.1f},
        {10.1f, 11.1f, 12.1f}
    };
    
    Tensor<> specific_weights_2 = {
        {1.1f, 6.1f},  // First column (original first row)
        {2.1f, 7.1f},  // Second column (original first row)
        {3.1f, 8.1f},  // Third column (original first row)
        {4.1f, 9.1f},  // Fourth column (original first row)
        {5.1f, 10.1f}  // Fifth column (original first row)
    };

    Tensor<> specific_bias_1 = {
        2.1f,  // First row
        4.1f,  // Second row
        6.1f,  // Third row
        8.1f,  // Fourth row
        10.1f  // Fifth row
    };

    Tensor<> specific_bias_2 = {
        3.1f,  // First row
        6.1f   // Second row
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
    Tensor<> Y = linear_2(output_1);

    Y.print();

    cout << endl;

    Tensor<> Y_hat = {
        {100.1f, 200.1f},   // First row
        {400.1f, 500.1f},   // Second row
        {700.1f, 800.1f},   // Third row
        {1000.1f, 1100.1f}  // Fourth row
    };

    MSE mse;

    const float mse_loss = mse.forward(Y, Y_hat);

    Tensor<> dL_dZ = mse.backward();
    Tensor<> dL_dY = linear_2.backward(dL_dZ);
    Tensor<> dL_dX = linear_1.backward(dL_dY);

    cout << "MSE Loss: " << mse_loss << endl;

    return 0;
}
