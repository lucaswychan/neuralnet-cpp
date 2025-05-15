#pragma once
#include "loss.hpp"

namespace nn {

class MSE : public Loss{
    public:
        MSE();
        virtual float forward(const Tensor<>& Y_hat, const Tensor<>& Y) override;
        virtual Tensor<> backward() override;
    };

}