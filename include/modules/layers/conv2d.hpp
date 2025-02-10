#pragma once
#include <utility>
#include "module.hpp"
#include "conv2d_utils.hpp"
using namespace nn;

namespace nn
{

    class Conv2d : public Module
    {
    public:
        Conv2d(int64_t in_channels,
               int64_t out_channels,
               var_pair kernel_size,
               var_pair stride = 1,
               var_pair padding = 0,
               var_pair dilation = 1,
               const string &padding_mode = "zeros",
               bool bias = true);

        virtual Tensor<> forward(const Tensor<> &input) override;
        virtual Tensor<> backward(const Tensor<> &grad_output) override;
        virtual void update_params(const float lr) override;

    private:
        int64_t in_channels_;
        int64_t out_channels_;
        int2 kernel_size_;
        int2 stride_;
        int2 padding_;
        int2 dilation_;
        bool use_bias_;
        PaddingMode padding_mode_;
        Padding padding_module_;
        vector<size_t> original_input_shape_;
        Tensor<> weight_;
        Tensor<> bias_;
        Tensor<> grad_weight_;
        Tensor<> grad_bias_;
    };
}