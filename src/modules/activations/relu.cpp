#include "relu.hpp"
using namespace nn;

Tensor<> ReLU::forward(const Tensor<>& input) {
    /*
    The forward process of ReLU is very similar to dropout with different criteria to select active units
    */

    // no need to cache input. Instead, we have to cache the mask for backprop
    this->mask_cache_ = Tensor<>(input.shapes(), 0.0f);

    Tensor<> result(input.shapes(), 0.0f);

    for (size_t i = 0; i < input.shapes()[0]; i++) {
        for (size_t j = 0; j < input.shapes()[1]; j++) {

            if (input[i, j] > 0.0f) {
                result[i, j] = input[i, j];
                this->mask_cache_[i, j] = 1.0f;
            } else {
                result[i, j] = 0.0f;
            }
        }
    }

    return result;
}

Tensor<> ReLU::backward(const Tensor<>& grad_output) {
    /*
    Z = ReLU(Y) = max(0, Y)
    Y R^B x M, B is the batch size and M is the output dimension

    dL/dZ = grad_output
    dZ/dY = MASK, where MASK[i, j] = 1 if Y[i, j] > 0 and 0 otherwise

    dL/dY = dL/dZ * dZ/dY
          = grad_output * MASK
    */ 

    return grad_output * this->mask_cache_;
}
