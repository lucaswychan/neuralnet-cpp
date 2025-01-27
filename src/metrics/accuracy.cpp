#include "accuracy.hpp"

double metrics::accuracy(const Tensor<>& output, const Tensor<>& target) {
    Tensor<size_t> output_argmax = output.argmax();

    Tensor<size_t> target_argmax;

    if (target.ndim() == 2) {
        // since target are a matrix of one hot vectors
        target_argmax = target.argmax();
    }
    else if (target.ndim() == 1) {
        target_argmax = target.dtype<size_t>();
    }
    else {
        throw std::runtime_error("Currently, Accuracy does not support label with more than 2 dimensions.");
    }

    Tensor<int> result = output_argmax.equal(target_argmax);

    return (double)result.sum() / (double)result.shapes()[0];
}