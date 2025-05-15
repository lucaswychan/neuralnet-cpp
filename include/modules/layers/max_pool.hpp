#pragma once
#include "module.hpp"
#include "conv2d_utils.hpp"

namespace nn
{
    class MaxPool2d : public Module
    {
    public:
        MaxPool2d(var_pair kernel_size, var_pair stride = (size_t)1, var_pair padding = (size_t)0, const string &padding_mode = "zeros");

        virtual Tensor<> forward(const Tensor<> &input) override;
        virtual Tensor<> backward(const Tensor<> &grad_output) override;

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