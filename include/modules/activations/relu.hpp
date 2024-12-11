#pragma once
#include "module.hpp"

namespace nn {
class ReLU : public Module {
    public:
        ReLU() = default;
        virtual Tensor<> forward(const Tensor<>& input) override;
        virtual Tensor<> backward(const Tensor<>& grad_output) override;
    };
}