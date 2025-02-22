#pragma once
#include "module.hpp"
#include "conv2d_utils.hpp"

namespace nn
{
    class MaxPool2d : public Module
    {
    public:
        MaxPool2d(var_pair kernel_size, var_pair stride, var_pair padding, const string &padding_mode = "zeros");

        virtual Tensor<> forward(const Tensor<> &input) override;
        virtual Tensor<> backward(const Tensor<> &grad_output) override;
        virtual void update_params(const float lr) override;

    private:
        size_tp2 kernel_size_;
        size_tp2 stride_;
        size_tp2 padding_;
        PaddingMode padding_mode_;
        Padding padding_module_;
        vector<size_t> original_input_shape_;
        Tensor<> grad_input_;
    };
}