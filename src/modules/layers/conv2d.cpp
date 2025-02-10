#include "conv2d.hpp"
using namespace nn;

Conv2d::Conv2d(int64_t in_channels,
               int64_t out_channels,
               var_pair kernel_size,
               var_pair stride,
               var_pair padding,
               var_pair dilation,
               const string &padding_mode,
               bool bias)
{
    this->in_channels_ = in_channels;
    this->out_channels_ = out_channels;
    this->use_bias_ = bias;

    // Helper lambda to process variant parameters
    auto process_variant = [](auto &&arg) -> int2
    {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, int64_t>)
        {
            if (arg < 0)
            {
                throw std::invalid_argument("Negative kernel size, stride, padding, or dilation is not supported");
            }
            return {arg, arg};
        }
        else
        {
            static_assert(std::is_same_v<T, std::pair<int64_t, int64_t>>, "Unexpected type in variant");
            if (arg.first < 0 || arg.second < 0)
            {
                throw std::invalid_argument("Negative kernel size, stride, padding, or dilation is not supported");
            }
            return arg;
        }
    };

    this->kernel_size_ = std::visit(process_variant, kernel_size);
    this->stride_ = std::visit(process_variant, stride);
    this->padding_ = std::visit(process_variant, padding);
    this->dilation_ = std::visit(process_variant, dilation);

    unordered_map<string, PaddingMode> all_padding_modes = {{"zeros", PaddingMode::ZEROS}, {"reflect", PaddingMode::REFLECT}, {"replicate", PaddingMode::REPLICATE}};

    if (all_padding_modes.find(padding_mode) == all_padding_modes.end())
    {
        throw std::invalid_argument("Padding mode must be one of 'zeros', 'reflect', or 'replicate'");
    }

    this->padding_mode_ = all_padding_modes[padding_mode];
    this->padding_module_ = Padding(this->padding_, this->padding_mode_);

    vector<size_t> weight_shape = {(size_t)this->out_channels_, (size_t)this->in_channels_, (size_t)this->kernel_size_.first, (size_t)this->kernel_size_.second};

    this->weight_ = Tensor<>(weight_shape, 0.0);

    if (this->use_bias_)
    {
        vector<size_t> bias_shape = {(size_t)this->out_channels_};
        this->bias_ = Tensor<>(bias_shape, 0.0);
    }
}

Tensor<> Conv2d::forward(const Tensor<> &input)
{
    Tensor<> input_data = input;
    this->original_input_shape_ = input.shapes();

    vector<size_t> output_shape = calculate_output_shape(input.shapes(), this->out_channels_, this->kernel_size_, this->stride_, this->padding_, this->dilation_);

    if (this->padding_.first > 0 && this->padding_.second > 0)
    {
        input_data = this->padding_module_.pad(input_data, this->padding_);
    }

    // this input is the padded version of the original input
    this->input_cache_ = input_data;

    return convolution(this->stride_, this->dilation_, output_shape, input_data, this->weight_, this->bias_, this->use_bias_);
}

Tensor<> Conv2d::backward(const Tensor<> &grad_output)
{
    // dL_dY = grad_output

    // dL_dW = conv(input_data, dL_dY)
    Tensor<> permuted_input = this->input_cache_.permute({1, 0, 2, 3});
    Tensor<> permuted_grad_output = grad_output.permute({1, 0, 2, 3});

    this->grad_weight_ = convolution(this->dilation_, this->stride_, this->weight_.shapes(), permuted_input, permuted_grad_output, Tensor<>(), false);

    this->grad_weight_ = this->grad_weight_.permute({1, 0, 2, 3});

    // dL_dB = sum(dL_dY, dims=(0, 2, 3))
    if (this->use_bias_)
    {
        this->grad_bias_ = Tensor<>({(size_t)this->out_channels_}, 0.0);
        for (size_t i = 0; i < grad_output.shapes()[0]; i++)
        {
            for (size_t j = 0; j < grad_output.shapes()[1]; j++)
            {
                for (size_t k = 0; k < grad_output.shapes()[2]; k++)
                {
                    for (size_t l = 0; l < grad_output.shapes()[3]; l++)
                    {
                        this->grad_bias_[j] += grad_output[i, j, k, l];
                    }
                }
            }
        }
    }

    // dL_dX = fullconv(dL_dY, W)
    Tensor<> flipped_weight = flip_vertical_and_horizontal(this->weight_);
    Tensor<> permuted_flipped_weight = flipped_weight.permute({1, 0, 2, 3});

    Tensor<> copy_grad_output = grad_output;

    if (this->stride_.first > 1 || this->stride_.second > 1)
    {
        copy_grad_output = dilate_input(copy_grad_output, this->stride_);
    }

    const size_t H_further_pad = (this->kernel_size_.first - 1) * this->dilation_.first - this->padding_.first;
    const size_t W_further_pad = (this->kernel_size_.second - 1) * this->dilation_.second - this->padding_.second;

    if (H_further_pad > 0 && W_further_pad > 0)
    {
        copy_grad_output = this->padding_module_.pad(copy_grad_output, {H_further_pad, W_further_pad});
    }
    else if (H_further_pad < 0 && W_further_pad < 0)
    {
        permuted_flipped_weight = this->padding_module_.pad(permuted_flipped_weight, {-H_further_pad, -W_further_pad});
    }
    else
    {
        throw std::invalid_argument("The further padding for dL/dX is not correct");
    }

    Tensor<> grad_input = convolution({1, 1}, this->dilation_, this->original_input_shape_, copy_grad_output, permuted_flipped_weight, Tensor<>(), false);

    return grad_input;
}

void Conv2d::update_params(const float lr)
{
    this->weight_ -= this->grad_weight_ * lr;

    if (this->use_bias_)
    {
        this->bias_ -= this->grad_bias_ * lr;
    }

    return;
}
