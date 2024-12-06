#include "modules/layers/linear.hpp"
#include "utils/matrix_utils.hpp"
using namespace nn;

Linear::Linear(int in_features, int out_features, bool bias) : in_features_(in_features), out_features_(out_features), bias_(bias) {
    this->weights_ = allocateMatrix(in_features, out_features);
    
    if (bias) {
        this->biases_ = allocateMatrix(out_features, 1);
    }
    else {
        this->biases_ = vector<vector<float>>();
    }

    this->input_cache_ = vector<vector<float>>();
}

vector<vector<float>> Linear::forward(const vector<vector<float>>& input) {
    this->input_cache_ = input;
    int batchSize = input.size();
    cout << "Batch size: " << batchSize << endl;
    const vector<vector<float>>& XW = matrixMultiplication(input, this->weights_);

    if (!this->bias_) {
        return XW;
    }
    return matrixAddition(XW, this->biases_);
}

vector<vector<float>> Linear::backward(const vector<vector<float>>& grad_output) {
    return vector<vector<float>>();
}

void Linear::update_params(const float lr) {
    return;
}