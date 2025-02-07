#include "conv2d.hpp"
using namespace nn;

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, int dilation, bool bias) {
    this->in_channels_ = in_channels;
    this->out_channels_ = out_channels;
    this->kernel_size_ = kernel_size;
    this->stride_ = stride;
    this->padding_ = padding;
    this->dilation_ = dilation;
    this->bias_ = bias;
}
