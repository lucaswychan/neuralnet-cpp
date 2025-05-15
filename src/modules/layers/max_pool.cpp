#include "max_pool.hpp"
using namespace nn;

MaxPool2d::MaxPool2d(var_pair kernel_size, var_pair stride, var_pair padding, const string &padding_mode)
{
    // Helper lambda to process variant parameters
    auto process_variant = [](auto &&arg) -> size_tp2
    {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, size_t>)
        {
            if (arg < 0)
            {
                throw std::invalid_argument("Negative kernel size, stride, padding, or dilation is not supported");
            }
            return {arg, arg};
        }
        else
        {
            static_assert(std::is_same_v<T, size_tp2>, "Unexpected type in variant");
            return arg;
        }
    };

    // Set kernel size, stride, and padding
    this->kernel_size_ = std::visit(process_variant, kernel_size);
    this->stride_ = std::visit(process_variant, stride);
    this->padding_ = std::visit(process_variant, padding);

    // Check if padding mode is valid
    unordered_map<string, PaddingMode> all_padding_modes = {{"zeros", PaddingMode::ZEROS}, {"reflect", PaddingMode::REFLECT}, {"replicate", PaddingMode::REPLICATE}};

    if (all_padding_modes.find(padding_mode) == all_padding_modes.end())
    {
        throw std::invalid_argument("Padding mode must be one of 'zeros', 'reflect', or 'replicate'");
    }

    // Set padding mode
    this->padding_mode_ = all_padding_modes[padding_mode];
    this->padding_module_ = Padding(this->padding_, this->padding_mode_);
}

Tensor<> MaxPool2d::forward(const Tensor<> &input)
{
    this->original_input_shape_ = input.shapes();

    // return output;
}

Tensor<> MaxPool2d::backward(const Tensor<> &grad_output)
{
    // return grad_input;
}