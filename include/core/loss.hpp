#pragma once
#include "tensor.hpp"

namespace nn {
class Loss {
    public:
        virtual double forward(const Tensor<>& Y, const Tensor<>& Y_hat) = 0;
        virtual Tensor<> backward() = 0;
    protected:
        Tensor<> grad_output_;
        Tensor<> Y_cache_;
        Tensor<> Y_hat_cache_;
    };
}

