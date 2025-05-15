#pragma once
#include "tensor.hpp"

namespace nn {
class Loss {
    public:
        Loss() = default;
        virtual ~Loss() = default;
        virtual float forward(const Tensor<>& Y_hat, const Tensor<>& Y) = 0;
        virtual Tensor<> backward() = 0;
        inline float operator()(const Tensor<>& Y_hat, const Tensor<>& Y) { return this->forward(Y_hat, Y); }


    protected:
        Tensor<> Y_cache_;
        Tensor<> Y_hat_cache_;
    };
}

