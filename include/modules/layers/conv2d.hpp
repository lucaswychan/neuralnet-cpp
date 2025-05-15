#pragma once
#include <utility>
#include "module.hpp"
#include "conv2d_utils.hpp"

namespace nn
{

    class Conv2d : public Module
    {
    public:
        Conv2d(size_t in_channels,
               size_t out_channels,
               var_pair kernel_size,
               var_pair stride = (size_t)1,
               var_pair padding = (size_t)0,
               var_pair dilation = (size_t)1,
               const string &padding_mode = "zeros",
               bool bias = true);

        virtual Tensor<> forward(const Tensor<> &input) override;
        virtual Tensor<> backward(const Tensor<> &grad_output) override;
        virtual void update_params(const float lr) override;

        void reset_parameters();

        // Setters
        inline void set_weight(const Tensor<> &target_weight) { this->weight_ = target_weight; }
        inline void set_bias(const Tensor<> &target_bias) { this->bias_ = target_bias; }

        // Getters
        inline const Tensor<> &get_weight() const { return this->weight_; }
        inline const Tensor<> &get_bias() const { return this->bias_; }

    private:
        size_t in_channels_;
        size_t out_channels_;
        size_tp2 kernel_size_;
        size_tp2 stride_;
        size_tp2 padding_;
        size_tp2 dilation_;
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