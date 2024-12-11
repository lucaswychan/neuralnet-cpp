#include "modules/losses/mse.hpp"
using namespace nn;

MSE::MSE() {}

double MSE::forward(const Tensor<>& Y_hat, const Tensor<>& Y) {
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

    double loss_without_factor = diff.sum();

    return loss_without_factor / (B * M);
}

Tensor<> MSE::backward() {
    // dL/dY_hat should have the same shape as Y_hat

    // 2 / (B * M) * (Y - Y_hat)

    const size_t B = this->Y_cache_.shapes()[0], M = this->Y_cache_.shapes()[1];
    const double factor = 2.0f / (B * M);

    Tensor<> diff = this->Y_cache_ - this->Y_hat_cache_;

    Tensor<> grad_output = diff * factor;

    return grad_output;
}