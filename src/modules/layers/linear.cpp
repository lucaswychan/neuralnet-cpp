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
    const vector<vector<float>>& XW = matrixMultiplication(input, this->weights_);

    if (!this->bias_) {
        return XW;
    }

    vector<vector<float>> biases_repeated = allocateMatrix(batchSize, this->out_features_);
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < this->out_features_; j++) {
            biases_repeated[i][j] = this->biases_[j][0];
        }
    }

    return matrixAddition(XW, biases_repeated);
}

vector<vector<float>> Linear::backward(const vector<vector<float>>& grad_output) {
    // grad_output = dL/dy

    // dL/dW = X^T * dL/dy
    this->grad_weights_ = matrixMultiplication(matrixTranspose(this->input_cache_), grad_output);

    // dL/dX = dL/dy * W^T
    vector<vector<float>> grad_input = matrixMultiplication(grad_output, matrixTranspose(this->weights_));

    // dL/db = dL/dy^T * 1_B (1_B is a vector of ones of size batchSize)
    if (this->bias_)
        this->grad_biases_ = matrixMultiplication(matrixTranspose(grad_output), vector<vector<float>>(grad_output.size(), vector<float>(1, 1.0f)));

    return grad_input;
}

void Linear::update_params(const float lr) {
    for (int i = 0; i < this->weights_.size(); i++) {
        for (int j = 0; j < this->weights_[i].size(); j++) {
            this->weights_[i][j] -= lr * this->grad_weights_[i][j];
        }
    }

    for (int i = 0; i < this->biases_.size(); i++) {
        for (int j = 0; j < this->biases_[i].size(); j++) {
            this->biases_[i][j] -= lr * this->grad_biases_[i][j];
        }
    }

    return;
}