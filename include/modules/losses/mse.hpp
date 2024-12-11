#pragma once
#include "loss.hpp"

namespace nn {

class MSE : public Loss{
    public:
        MSE();
        virtual double forward(const Tensor<>& Y_hat, const Tensor<>& Y) override;
        virtual Tensor<> backward() override;
    };

}