#include "relu.hpp"
using namespace nn;

Tensor<> ReLU::forward(const Tensor<>& input) {
    this->input_cache_ = input;

    return input.filter([](double x) { return x > 0.0f; });
}

Tensor<> ReLU::backward(const Tensor<>& grad_output) {
    Tensor<> grad(grad_output.shapes(), 0.0f);

    for (size_t i = 0; i < this->input_cache_.shapes()[0]; i++) {
        for (size_t j = 0; j < this->input_cache_.shapes()[1]; j++) {
            if (this->input_cache_[i, j] <= 0.0f) {
                grad[i, j] = 0.0f;
            }
        }
    }

    return grad;
}
