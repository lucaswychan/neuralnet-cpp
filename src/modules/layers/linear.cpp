#include <random>
#include <cmath>
#include "linear.hpp"
using namespace nn;

Linear::Linear(size_t in_features, size_t out_features, bool bias) : in_features_(in_features), out_features_(out_features), bias_(bias) {
    this->weights_ = Tensor<>({in_features, out_features}, 0.0f);
    
    if (bias) {
        this->biases_ = Tensor<>({out_features, 1}, 0.0f);
    }
    else {
        this->biases_ = Tensor<>();
    }

    // randomize the weights. The bias is originally 0.
    this->randomizeParams();

    this->grad_weights_ = Tensor<>({in_features, out_features}, 0.0f);;
    this->grad_biases_ = Tensor<>({out_features, 1}, 0.0f);

    this->input_cache_ = Tensor<>();
}

Tensor<> Linear::forward(const Tensor<>& input) {
    this->input_cache_ = input;
    size_t batchSize = input.shapes()[0];

    const Tensor<>& XW = input.matmul(this->weights_);

    if (!this->bias_) {
        return XW;
    }

    Tensor<> biases_repeated = Tensor<>({batchSize, this->out_features_}, 0.0f);

    for (size_t i = 0; i < batchSize; i++) {
        for (size_t j = 0; j < this->out_features_; j++) {
            biases_repeated[i, j] = this->biases_[j, 0];
        }
    }

    return XW + biases_repeated;
}

Tensor<> Linear::backward(const Tensor<>& grad_output) {
    // dL/dY = grad_output

    // dL/dW = X^T * dL/dY
    this->grad_weights_ = this->input_cache_.transpose().matmul(grad_output);

    // cout << endl << "dL/dW: " << endl;
    // this->grad_weights_.print();
    // cout << endl;

    // dL/dX = dL/dY * W^T
    Tensor<> grad_input = grad_output.matmul(this->weights_.transpose());

    /*
    dL/db = dL/dY^T * 1_B (1_B is a vector of ones of size batchSize)
    dL/db = dL/dY.sum(axis=0)
    */
    if (this->bias_) 
        this->grad_biases_ = grad_output.transpose().matmul(Tensor<>({grad_output.shapes()[0], 1}, 1.0f));

    // cout << endl << "dL/db: " << endl;
    // this->grad_biases_.print();
    // cout << endl;

    return grad_input;
}

void Linear::update_params(const float lr) {

    this->weights_ -= this->grad_weights_ * lr;
    this->biases_ -= this->grad_biases_ * lr;

    return;
}

void Linear::randomizeParams() {
    // Calculate the limit for the uniform distribution
    double limit = sqrt(6.0f / (this->in_features_ + this->out_features_));

    // Set up the random number generator
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-limit, limit);

    // Xavier initialization
    for (size_t i = 0; i < this->in_features_; i++) {
        for (size_t j = 0; j < this->out_features_; j++) {
            this->weights_[i, j] = dis(gen);
        }
    }
}