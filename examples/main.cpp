#include <iostream>
#include <vector>
#include "modules/layers/linear.hpp"
#include "utils/matrix_utils.hpp"
using namespace nn;

int main() {
    Linear linear(3, 4, false);

    vector<vector<float>> input = allocateMatrix(2, 3);
    input[0][0] = 1.0f;
    input[0][1] = 2.0f;
    input[0][2] = 3.0f;
    input[1][0] = 4.0f;
    input[1][1] = 5.0f;
    input[1][2] = 6.0f;

    printMatrix(input);

    cout << endl;

    cout << "Batch size in main: " << sizeof(input) / sizeof(float) << endl;

    vector<vector<float>> weights = allocateMatrix(3, 4);
    weights[0][0] = 1.0f;
    weights[0][1] = 2.0f;
    weights[0][2] = 3.0f;
    weights[0][3] = 4.0f;
    weights[1][0] = 5.0f;
    weights[1][1] = 6.0f;
    weights[1][2] = 7.0f;
    weights[1][3] = 8.0f;
    weights[2][0] = 9.0f;
    weights[2][1] = 10.0f;
    weights[2][2] = 11.0f;
    weights[2][3] = 12.0f;

    cout << "Weights rows: " << sizeof(weights) / sizeof(float) << endl;

    linear.setWeights(weights);

    printMatrix(linear.getWeights());

    vector<vector<float>> output = linear.forward(input);

    printMatrix(output);

    return 0;
}
