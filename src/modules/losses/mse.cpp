#include "modules/losses/mse.hpp"
using namespace nn;

MSE::MSE() {}

float MSE::forward(const Tensor<>& Y, const Tensor<>& Y_hat) {
    // Y R^B x M, Y_hat R^B x M , B is the batch size and M is the output dimension

    // 1 / (B * M) * ||(Y - Y_hat)||^2

    this->Y_cache_ = Y;
    this->Y_hat_cache_ = Y_hat;
    
    const size_t B  = Y.shapes()[0], M = Y.shapes()[1];
    if (Y_hat.shapes()[0] != B || Y_hat.shapes()[1] != M) {
        throw runtime_error("Shape mismatch");
    }

    Tensor<> diff = Y - Y_hat;
    diff *= diff;

    float loss = 0.0f;
    for (size_t i = 0; i < B; i++) {
        for (size_t j = 0; j < M; j++) {
            loss += diff[i, j];
        }
    }

    return loss / (B * M);
}

Tensor<> MSE::backward() {
    // dL/dY_hat should have the same shape as Y_hat

    // 2 / (B * M) * (Y - Y_hat)

    const size_t B = this->Y_cache_.shapes()[0], M = this->Y_cache_.shapes()[1];
    const float factor = 2.0f / (B * M);

    Tensor<> diff = this->Y_cache_ - this->Y_hat_cache_;

    Tensor<> grad_output = diff * factor;

    return grad_output;
}