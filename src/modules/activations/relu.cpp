#include "relu.hpp"
using namespace nn;

Tensor<> ReLU::forward(const Tensor<>& input) {
    /*
    The forward process of ReLU is very similar to dropout with different criteria to select active units
    */

    // Initialize the mask cache with the same shape as input, filled with zeros
    this->mask_cache_ = Tensor<>(input.shapes(), 0.0f);

    // Create result tensor with the same shape as input, filled with zeros
    Tensor<> result(input.shapes(), 0.0f);
    
    // Process all elements regardless of tensor dimensions
    for (size_t i = 0; i < input.size(); i++) {
        // Use linear indexing with at() to access tensor elements
        if (input.at(i) > 0.0f) {
            result.at(i) = input.at(i);
            this->mask_cache_.at(i) = 1.0f;
        }
        // No need for else case as tensors are already initialized with zeros
    }

    return result;
}

Tensor<> ReLU::backward(const Tensor<>& grad_output) {
    /*
    Z = ReLU(Y) = max(0, Y)
    Y can be of any dimension

    dL/dZ = grad_output
    dZ/dY = MASK, where MASK[i] = 1 if Y[i] > 0 and 0 otherwise

    dL/dY = dL/dZ * dZ/dY
          = grad_output * MASK
    */ 

    return grad_output * this->mask_cache_;
}
