#pragma once
#include "tensor.hpp"

namespace metrics {
    double accuracy(const Tensor<>& output, const Tensor<>& target);
}