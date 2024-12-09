#include <math.h>
#include "softmax.hpp"
using namespace nn;

Tensor<> Softmax::softmax_helper(const Tensor<>& input) {
    Tensor<> result = input.map([](double x) { return exp(x); });
    double sum = result.sum();

    return result * (1 / sum);
}

vector<double> Softmax::softmax_helper(const vector<double>& input) {
    double sum = 0.0f;
    vector<double> result;

    for (size_t i = 0; i < input.size(); i++) {
        sum += exp(input[i]);
    }
    for (size_t i = 0; i < input.size(); i++) {
        result.push_back(exp(input[i]) / sum);
    }

    return result;
}

// Only support 1D and 2D Tensors
Tensor<> Softmax::forward(const Tensor<>& input) {
    this->input_cache_ = input;

    if (input.ndim() == 1) {
        return this->softmax_helper(input);
    }

    vector<vector<double>> result;

    for (size_t i = 0; i < input.shapes()[0]; i++) {
        vector<double>sub_vector;
        for (size_t j = 0; j < input.shapes()[1]; j++) {
            sub_vector.push_back(input[i, j]);
        }
        result.push_back(this->softmax_helper(sub_vector));
    }

    return Tensor<>(result);
}

Tensor<> Softmax::backward(const Tensor<>& grad_output) {
    Tensor<> grad(grad_output.shapes(), 0.0f);

    for (size_t i = 0; i < this->input_cache_.shapes()[0]; i++) {
        for (size_t j = 0; j < this->input_cache_.shapes()[1]; j++) {
            grad[i, j] = grad_output[i, j] * this->input_cache_[i, j];
        }
    }

    return grad;
}