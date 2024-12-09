#pragma once
#include "loss.hpp"

namespace nn {

class CrossEntropyLoss : Loss {
    public:
        CrossEntropyLoss();
        virtual double forward(const Tensor<>& Y, const Tensor<>& Y_hat) override;
        virtual Tensor<> backward() override;
    private:
        Tensor<> softmax_Y_hat_cache_;
    };

}