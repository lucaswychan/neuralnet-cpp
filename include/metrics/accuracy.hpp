#pragma once
#include "tensor.hpp"

namespace metrics {
    float accuracy(const Tensor<>& output, const Tensor<>& target);
}