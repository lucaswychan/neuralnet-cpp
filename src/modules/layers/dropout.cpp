#include "dropout.hpp"
using namespace nn;

Dropout::Dropout(float p) {
    if (p < 0 || p > 1) {
        throw runtime_error("Dropout probability must be between 0 and 1");
    }

    // probability P[X_i = 0] = p
    // It is inverted dropout
    this->p_ = p;
    this->scale_ = 1.0f / (1 - p);
    this->pmf_ = bernoulli_distribution(1 - p);

    // Set up the random number generator
    random_device rd;
    this->gen_ = mt19937(rd());
}

Tensor<> Dropout::forward(const Tensor<>& input) {
    // no need to cache input. Instead, we have to cache the mask for backprop
    this->mask_cache_ = Tensor<>(input.shapes(), 0.0f);

    if (!this->training) {
        return input;
    }

    Tensor<> result(input.shapes(), 0.0f);

    for (size_t i = 0; i < input.shapes()[0]; i++) {
        for (size_t j = 0; j < input.shapes()[1]; j++) {
            bool is_active = this->pmf_(this->gen_);

            if (is_active) {
                result[i, j] = input[i, j];
                this->mask_cache_[i, j] = 1.0f;
            } else {
                result[i, j] = 0.0f;
            }
        }
    }

    return result * this->scale_;
}

Tensor<> Dropout::backward(const Tensor<>& grad_output) {
    /*
    Z = Dropout(Y) = Y * MASK * 1 / (1 - p)
    Y R^B x M, MASK R^B x M, B is the batch size and M is the output dimension

    dL/dZ = grad_output
    dZ/dY = MASK * 1 / (1 - p)

    dL/dY = dL/dZ * dZ/dY
          = grad_output * MASK * 1 / (1 - p)
    */

    if (!this->training) {
        return grad_output;
    }

    return grad_output * this->mask_cache_ * this->scale_;

}