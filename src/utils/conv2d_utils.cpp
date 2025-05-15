#include "conv2d_utils.hpp"

Tensor<> Padding::pad(const Tensor<> &input, const size_tp2 &padding) const
{
    switch (this->padding_mode_)
    {
    case PaddingMode::ZEROS:
        return this->zero_pad(input, padding);
        break;
    default:
        throw std::invalid_argument("Invalid padding mode");
    }
}

Tensor<> Padding::zero_pad(const Tensor<> &input, const size_tp2 &padding) const
{
    const vector<size_t> &input_shape = input.shapes();

    if (input_shape.size() != 4)
    {
        throw std::invalid_argument("Input shape must be 4D");
    }

    const size_t B = input_shape[0];
    const size_t C = input_shape[1];
    const size_t H = input_shape[2];
    const size_t W = input_shape[3];

    const size_t padded_H = H + padding.first * 2;
    const size_t padded_W = W + padding.second * 2;

    Tensor<> padded_output({B, C, padded_H, padded_W}, 0.0);

    for (size_t b = 0; b < B; ++b)
    {
        for (size_t c = 0; c < C; ++c)
        {
            for (size_t h = 0; h < H; ++h)
            {
                for (size_t w = 0; w < W; ++w)
                {
                    padded_output[b, c, h + padding.first, w + padding.second] = input[b, c, h, w];
                }
            }
        }
    }

    return padded_output;
}

Tensor<> convolution(const size_tp2 &stride, const size_tp2 &dilation, const vector<size_t> &output_shape, const Tensor<> &input, const Tensor<> &kernel, const Tensor<> &bias, bool use_bias)
{
    const vector<size_t> &input_shape = input.shapes();
    const vector<size_t> &kernel_shape = kernel.shapes();

    if (output_shape.size() != 4)
    {
        throw std::invalid_argument("Output shape must be 4D");
    }
    if (input_shape.size() != 4)
    {
        throw std::invalid_argument("Input shape must be 4D");
    }
    if (kernel_shape.size() != 4)
    {
        throw std::invalid_argument("Kernel shape must be 4D");
    }

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

/**
 * Calculate the output shape of a 2D convolutional layer.
 *
 * @param input_shape The shape of the input tensor, which is a 4D tensor with shape (B, C_in, H_in, W_in).
 * @param out_channel The number of output channels.
 * @param kernel_size The size of the kernel, which is a 2D integer pair.
 * @param stride The stride of the convolution, which is a 2D integer pair.
 * @param padding The padding of the convolution, which is a 2D integer pair.
 * @param dilation The dilation of the convolution, which is a 2D integer pair.
 * @return The output shape, which is a 4D vector with shape (B, out_channel, H_out, W_out).
 *
 * @throws std::invalid_argument if input_shape is not 4D or if the output shape is invalid.
 */
const vector<size_t> calculate_output_shape(const vector<size_t> &input_shape, const int64_t out_channel, const size_tp2 &kernel_size, const size_tp2 &stride, const size_tp2 &padding, const size_tp2 &dilation)
{
    if (input_shape.size() != 4)
    {
        throw std::invalid_argument("Input shape must be 4D");
    }

    const size_t B = input_shape[0];
    const size_t H_in = input_shape[2];
    const size_t W_in = input_shape[3];

    cout << "Btach Size : " << B << endl;
    cout << "H_in : " << H_in << endl;
    cout << "W_in : " << W_in << endl;
    cout << "Out Channel : " << out_channel << endl;
    cout << "Kernel Size : " << kernel_size.first << ", " << kernel_size.second << endl;
    cout << "Stride : " << stride.first << ", " << stride.second << endl;
    cout << "Padding : " << padding.first << ", " << padding.second << endl;
    cout << "Dilation : " << dilation.first << ", " << dilation.second << endl;

    const int64_t H_out = (H_in + 2 * padding.first - dilation.first * (kernel_size.first - 1) - 1) / stride.first + 1;
    const int64_t W_out = (W_in + 2 * padding.second - dilation.second * (kernel_size.second - 1) - 1) / stride.second + 1;

    if (H_out <= 0 || W_out <= 0)
    {
        throw std::invalid_argument("Invalid output shape");
    }

    return {B, (size_t)out_channel, (size_t)H_out, (size_t)W_out};
}

Tensor<> flip_vertical_and_horizontal(const Tensor<> &input)
{
    if (input.ndim() != 4)
    {
        throw std::invalid_argument("Input shape must be 4D");
    }

    Tensor<> output = input;

    const size_t B = input.shapes()[0];
    const size_t C = input.shapes()[1];
    const size_t H = input.shapes()[2];
    const size_t W = input.shapes()[3];

    float cache;

    for (size_t b = 0; b < B; ++b)
    {
        for (size_t c = 0; c < C; ++c)
        {
            for (size_t h = 0; h < H / 2; ++h)
            {
                for (size_t w = 0; w < W; ++w)
                {
                    cache = output[b, c, h, w];
                    output[b, c, h, w] = output[b, c, H - h - 1, w];
                    output[b, c, H - h - 1, w] = cache;
                }
            }
            for (size_t h = 0; h < H; ++h)
            {
                for (size_t w = 0; w < W / 2; ++w)
                {
                    cache = output[b, c, h, w];
                    output[b, c, h, w] = output[b, c, h, W - w - 1];
                    output[b, c, h, W - w - 1] = cache;
                }
            }
        }
    }

    return output;
}
Tensor<> dilate_input(const Tensor<> &input, const size_tp2 &dilation)
{
    if (input.ndim() != 4)
    {
        throw std::invalid_argument("Input shape must be 4D");
    }

    const size_t B = input.shapes()[0];
    const size_t C = input.shapes()[1];
    const size_t H = input.shapes()[2];
    const size_t W = input.shapes()[3];

    const size_t H_dilated = H + (H - 1) * (dilation.first - 1);
    const size_t W_dilated = W + (W - 1) * (dilation.second - 1);

    Tensor<> dilated_input({B, C, H_dilated, W_dilated}, 0.0);

    for (size_t b = 0; b < B; ++b)
    {
        for (size_t c = 0; c < C; ++c)
        {
            for (size_t h = 0; h < H; ++h)
            {
                for (size_t w = 0; w < W; ++w)
                {
                    dilated_input[b, c, h * dilation.first, w * dilation.second] = input[b, c, h, w];
                }
            }
        }
    }

    return dilated_input;
}