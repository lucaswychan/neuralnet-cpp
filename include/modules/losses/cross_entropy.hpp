#pragma once
#include "loss.hpp"

namespace nn {

class CrossEntropyLoss : public Loss {
    public:
        CrossEntropyLoss();
        virtual double forward(const Tensor<>& Y_hat, const Tensor<>& Y) override;
        virtual Tensor<> backward() override;

    private:
        Tensor<> softmax_Y_hat_cache_;
    };

}