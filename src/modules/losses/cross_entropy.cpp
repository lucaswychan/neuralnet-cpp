#include "modules/losses/cross_entropy.hpp"
#include "softmax.hpp"
#include <math.h>
using namespace nn;

CrossEntropyLoss::CrossEntropyLoss() {}

double CrossEntropyLoss::forward(const Tensor<>& Y, const Tensor<>& Y_hat) {
    /*
    L = 1 / B \sum_{i=1}^B \sum_{j=1}^M Y_{ij} * log(softmax(Y_hat_{ij)})

    R^B x M, Y R^B x M

    Y_{ij} is 1 if the correct label is j and 0 otherwise
    */
    this->Y_cache_ = Y;
    this->Y_hat_cache_ = Y_hat;

    const size_t B = Y.shapes()[0];
    const double factor = -1.0f / B;

    // apply softmax to model output
    Tensor<> softmax_Y_hat = Softmax().forward(Y_hat);
    this->softmax_Y_hat_cache_ = softmax_Y_hat;

    // log(softmax(Y_hat_{ij}))
    Tensor<> logit = softmax_Y_hat.map([](double x) { return log(x); });

    // Y_{ij} * log(softmax(Y_hat_{ij}))
    logit *= Y;

    // sum up all the elements
    double loss_without_factor = logit.sum();

    return loss_without_factor * factor;
}

Tensor<> CrossEntropyLoss::backward() {
    /*
    dL/dY_hat should have the same shape as Y_hat

    dL/dY_hat = 1/ B (softmax_Y_hat * \sum_{j=1}^M Y_{ij} - Y)

    softmax_Y_hat R^B x M, Y R^B x M
    */

    const size_t B = this->Y_cache_.shapes()[0];

    // assuming Y_i is a one-hot vector
    Tensor<> grad_output = this->softmax_Y_hat_cache_ - this->Y_cache_;
    grad_output *= 1.0f / B;

    return grad_output;
}