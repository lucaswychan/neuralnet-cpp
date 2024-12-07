#include "linear.hpp"
#include <random>
#include <cmath>
using namespace nn;

Linear::Linear(int in_features, int out_features, bool bias) : in_features_(in_features), out_features_(out_features), bias_(bias) {
    this->weights_ = allocateMatrix(in_features, out_features);
    
    if (bias) {
        this->biases_ = allocateMatrix(out_features, 1);
    }
    else {
        this->biases_ = vector<vector<float>>();
    }

    // randomize the weights. The bias is originally 0.
    this->randomizeParams();

    this->grad_weights_ = allocateMatrix(in_features, out_features);
    this->grad_biases_ = allocateMatrix(out_features, 1);

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
    // grad_output = dL/dY

    // dL/dW = X^T * dL/dY
    this->grad_weights_ = matrixMultiplication(matrixTranspose(this->input_cache_), grad_output);

    // dL/dX = dL/dY * W^T
    vector<vector<float>> grad_input = matrixMultiplication(grad_output, matrixTranspose(this->weights_));

    cout << "dL/dW: " << endl;
    printMatrix(this->grad_weights_);
    cout << endl;

    // dL/db = dL/dY^T * 1_B (1_B is a vector of ones of size batchSize)
    // dL/db = dL/dY.sum(axis=0)
    if (this->bias_) 
        this->grad_biases_ = matrixMultiplication(matrixTranspose(grad_output), allocateMatrix(grad_output.size(), 1, 1.0f));

    cout << "dL/db: " << endl;
    printMatrix(this->grad_biases_);
    cout << endl;

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

void Linear::randomizeParams() {
    // Calculate the limit for the uniform distribution
    float limit = sqrt(6.0f / (this->in_features_ + this->out_features_));

    // Set up the random number generator
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-limit, limit);

    // Xavier initialization
    for (int i = 0; i < this->in_features_; i++) {
        for (int j = 0; j < this->out_features_; j++) {
            this->weights_[i][j] = dis(gen);
        }
    }
}