#include "linear.hpp"
#include <random>
#include <cmath>
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

    cout << "biases_repeated: " << endl;
    biases_repeated.print();
    cout << endl;

    cout << "XW" << endl;
    XW.print();
    cout << endl;

    return XW + biases_repeated;
}

Tensor<> Linear::backward(const Tensor<>& grad_output) {
    // grad_output = dL/dY

    // dL/dW = X^T * dL/dY
    this->grad_weights_ = this->input_cache_.transpose().matmul(grad_output);

    // dL/dX = dL/dY * W^T
    Tensor<> grad_input = grad_output.matmul(this->weights_.transpose());

    cout << "dL/dW: " << endl;
    this->grad_weights_.print();
    cout << endl;

    // dL/db = dL/dY^T * 1_B (1_B is a vector of ones of size batchSize)
    // dL/db = dL/dY.sum(axis=0)
    if (this->bias_) 
        this->grad_biases_ = grad_output.transpose().matmul(Tensor<>({grad_output.shapes()[0], 1}, 1.0f));

    cout << "dL/db: " << endl;
    this->grad_biases_.print();
    cout << endl;

    return grad_input;
}

void Linear::update_params(const float lr) {
    const size_t n = this->weights_.shapes()[0], m = this->weights_.shapes()[1];

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            this->weights_[i, j] -= lr * this->grad_weights_[i, j];
        }
    }

    const size_t k = this->biases_.shapes()[0];

    for (size_t i = 0; i < k; i++) {
        this->biases_[i, 0] -= lr * this->grad_biases_[i, 0];
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
    for (size_t i = 0; i < this->in_features_; i++) {
        for (size_t j = 0; j < this->out_features_; j++) {
            this->weights_[i, j] = dis(gen);
        }
    }
}