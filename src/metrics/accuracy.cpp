#include "accuracy.hpp"

double metrics::accuracy(const Tensor<>& output, const Tensor<>& target) {
    Tensor<size_t> output_argmax = output.argmax();
    // since target are a matrix of one hot vectors
    Tensor<size_t> target_argmax = target.argmax();

    Tensor<int> result = output_argmax.equal(target_argmax);

    return (double)result.sum() / (double)result.shapes()[0];
}