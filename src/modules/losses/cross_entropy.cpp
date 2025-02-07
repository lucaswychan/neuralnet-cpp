#include "cross_entropy.hpp"
#include "softmax.hpp"
#include <math.h>
using namespace nn;

CrossEntropyLoss::CrossEntropyLoss() {
    cout << "Starting CrossEntropyLoss" << endl;
    cout << "CrossEntropyLoss initialized" << endl;
}

double CrossEntropyLoss::forward(const Tensor<>& Y_hat, const Tensor<>& Y) {
    /*
    L = 1 / B \sum_{i=1}^B \sum_{j=1}^M Y_{ij} * log(softmax(Y_hat_{ij)})

    R^B x M, Y R^B x M

    Y_{ij} is 1 if the correct label is j and 0 otherwise
    */

    // We don't have to store the Y_hat as it is not used in the backward pass. Instead, we store the softmax(Y_hat)
    // Note that this->Y_cache_ is just a vector with label, and it is not a matrix with one-hot vectors.
    if (Y.ndim() == 2) {
        // In this case, we assume Y is a matrix of one-hot vectors. So we can just store the index of the correct label
        this->Y_cache_ = Y.argmax().dtype<double>();
    }
    else if (Y.ndim() == 1) {
        this->Y_cache_ = Y;
    }
    else {
        throw std::runtime_error("Currently, Cross Entropy Loss does not support label with more than 2 dimensions.");
    }

    // B = batch size
    const size_t B = this->Y_cache_.shapes()[0];
    const double factor = -1.0f / B;

    // apply softmax to model output
    Tensor<> softmax_Y_hat = this->softmax_(Y_hat);
    this->softmax_Y_hat_cache_ = softmax_Y_hat;

    // sum up all the elements
    double loss_without_factor = 0.0f;

    for (int i = 0; i < B; ++i) {
        // Y_{ij} * log(softmax(Y_hat_{ij}))
        loss_without_factor += log(softmax_Y_hat[i, static_cast<int>(this->Y_cache_[i])]);
    }

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
    Tensor<> grad_output = this->softmax_Y_hat_cache_;

    /*
    softmax(Y_hat) - Y

    Since Y is a matrix of one-hot vectors, only the correct label is 1 and the rest are 0
    */

    for (int i = 0; i < B; ++i) {
        grad_output[i, static_cast<int>(this->Y_cache_[i])] -= 1.0f;
    }

    grad_output *= 1.0f / B;

    return grad_output;
}