#pragma once
#include "tensor.hpp"

namespace nn {
class Loss {
    public:
        virtual ~Loss() = default;
        virtual double forward(const Tensor<>& Y_hat, const Tensor<>& Y) = 0;
        virtual Tensor<> backward() = 0;
    protected:
        Tensor<> grad_output_;
        Tensor<> Y_cache_;
        Tensor<> Y_hat_cache_;
    };
}

