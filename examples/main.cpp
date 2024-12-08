#include <iostream>
#include <vector>
#include "modules/layers/linear.hpp"
#include "modules/losses/mse.hpp"
#include "core/tensor.hpp"
using namespace nn;

int main() {
    // const bool bias = true;

    // Linear linear_1(3, 5, bias);

    // cout << "Before initialization: " << endl;
    // printMatrix(linear_1.getWeights());

    // Linear linear_2(5, 2, bias);

    // vector<vector<float>> specific_weights_1 = {
    //     {1.0f, 4.0f, 7.0f, 10.0f, 13.0f},
    //     {2.0f, 5.0f, 8.0f, 11.0f, 14.0f},
    //     {3.0f, 6.0f, 9.0f, 12.0f, 15.0f}
    // };

    // std::vector<std::vector<float>> input = {
    //     {1.0f, 2.0f, 3.0f},
    //     {4.0f, 5.0f, 6.0f},
    //     {7.0f, 8.0f, 9.0f},
    //     {10.0f, 11.0f, 12.0f}
    // };
    
    // std::vector<std::vector<float>> specific_weights_2 = {
    //     {1.0f, 6.0f},  // First column (original first row)
    //     {2.0f, 7.0f},  // Second column (original first row)
    //     {3.0f, 8.0f},  // Third column (original first row)
    //     {4.0f, 9.0f},  // Fourth column (original first row)
    //     {5.0f, 10.0f}  // Fifth column (original first row)
    // };

    // std::vector<std::vector<float>> specific_bias_1 = {
    //     {2.0f},  // First row
    //     {4.0f},  // Second row
    //     {6.0f},  // Third row
    //     {8.0f},  // Fourth row
    //     {10.0f}  // Fifth row
    // };

    // std::vector<std::vector<float>> specific_bias_2 = {
    //     {3.0f},  // First row
    //     {6.0f}   // Second row
    // };

    // linear_1.setWeights(specific_weights_1);
    // linear_2.setWeights(specific_weights_2);

    // linear_1.setBiases(specific_bias_1);
    // linear_2.setBiases(specific_bias_2);

    // printMatrix(input);

    // cout << endl;

    // vector<vector<float>> output_1 = linear_1(input);
    // vector<vector<float>> Y = linear_2(output_1);

    // printMatrix(Y);

    // cout << endl;

    // std::vector<std::vector<float>> Y_hat = {
    //     {100.0f, 200.0f},   // First row
    //     {400.0f, 500.0f},   // Second row
    //     {700.0f, 800.0f},   // Third row
    //     {1000.0f, 1100.0f}  // Fourth row
    // };

    // MSE mse;

    // const float mse_loss = mse.forward(Y, Y_hat);

    // vector<vector<float>> dL_dZ = mse.backward();
    // vector<vector<float>> dL_dY = linear_2.backward(dL_dZ);
    // vector<vector<float>> dL_dX = linear_1.backward(dL_dY);

    // cout << "MSE Loss: " << mse_loss << endl;

    Tensor<float> tensor_1d {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    Tensor<float> tensor_2d {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}};

    cout << "Tensor 1D: " << endl;
    tensor_1d.print();
    cout << endl;

    cout << "Tensor 2D: " << endl;
    tensor_2d.print();
    cout << endl;

    Tensor<float> transposed_2d = tensor_2d.transpose();

    cout << "Transposed Tensor 2D: " << endl;
    transposed_2d.print();
    cout << endl;

    Tensor<float> mul_tensor = tensor_2d.matmul(transposed_2d);

    cout << "Matrix Multiplication Result: " << endl;
    mul_tensor.print();
    cout << endl;

    return 0;
}
