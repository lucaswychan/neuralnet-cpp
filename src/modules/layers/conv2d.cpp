#include "conv2d.hpp"
using namespace nn;

Conv2d::Conv2d(int64_t in_channels,
               int64_t out_channels,
               var_pair kernel_size,
               var_pair stride,
               var_pair padding,
               var_pair dilation,
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
    this->original_input_shape_ = input.shapes();
}

Tensor<> Conv2d::backward(const Tensor<> &grad_output)
{
}

void Conv2d::update_params(const float lr)
{
}

Tensor<> Conv2d::convolution(const int2 &stride, const int2 &dilation, const vector<size_t> &output_shape, const Tensor<> &input, const Tensor<> &kernel, const Tensor<> &bias, bool use_bias)
{
    const vector<size_t> &input_shape = input.shapes();
    const vector<size_t> &kernel_shape = kernel.shapes();

    const size_t B = output_shape[0];
    const size_t C_out = output_shape[1];
    const size_t H_out = output_shape[2];
    const size_t W_out = output_shape[3];

    const size_t C_in = input_shape[1];
    const size_t H_in = input_shape[2];
    const size_t W_in = input_shape[3];

    const size_t K_H = kernel_shape[2];
    const size_t K_W = kernel_shape[3];

    Tensor<> output(output_shape, 0.0);

    /*
    The logic behind is that
    Let's us first focus on the first kernel among all out_channel kernels

    Each input channel of the data is convolved with the same channel of the kernel, and the result is added to the output
    Meaning that each input data channel only corresponds to the same channel of the kernel

    For example, the channel 1 of the input data is convolved with the channel 1 of the kernel, but it will not be convolved with the channel 2 of the kernel

    After each input data channel convolving with the same channel of the kernel, element-wise addition is performed among all the convolved result with the first kernel

    Now we get a single output channel

    We repeat this process for all the out_channel channels

    And finally we will get an output with out_channel channels
    */

    for (size_t b = 0; b < B; ++b)
    {
        for (size_t c = 0; c < C_out; ++c)
        {
            for (size_t h = 0; h < H_out; ++h)
            {
                for (size_t w = 0; w < W_out; ++w)
                {
                    size_t h_start = h * stride.first;
                    size_t w_start = w * stride.second;

                    for (size_t ic = 0; ic < C_in; ++ic)
                    {
                        for (size_t kh = 0; kh < K_H; ++kh)
                        {
                            for (size_t kw = 0; kw < K_W; ++kw)
                            {
                                size_t h_in = h_start + kh * dilation.first;
                                size_t w_in = w_start + kw * dilation.second;

                                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in)
                                {
                                    output[b, c, h, w] += input[b, ic, h_in, w_in] * kernel[c, ic, kh, kw];
                                }
                            }
                        }
                    }

                    if (use_bias)
                    {
                        output[b, c, h, w] += bias[c];
                    }
                }
            }
        }
    }

    return output;
}

std::tuple<int64_t, int64_t, int64_t, int64_t> Conv2d::calculate_output_shape(const Tensor<> &input)
{
}
