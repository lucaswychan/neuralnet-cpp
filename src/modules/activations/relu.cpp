#include "relu.hpp"
using namespace nn;

vector<vector<float>> ReLU::forward(const vector<vector<float>>& input) {
    this->input_cache_ = input;

    vector<vector<float>> output = input;
    const int B = input.size(), N = input[0].size();
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < N; j++) {
            if (input[i][j] < 0) {
                output[i][j] = 0;
            }
        }
    }

    return output;
}

vector<vector<float>> ReLU::backward(const vector<vector<float>>& grad_output) {
    return matrixMultiplication(this->input_cache_, grad_output);
}
