#include "tensor.hpp"
using namespace std;

using int2 = std::pair<int64_t, int64_t>;
using var_pair = std::variant<int64_t, int2>;

enum class PaddingMode
{
    ZEROS,
    REFLECT,
    REPLICATE
};

class Padding
{
public:
    Padding() = default;
    Padding(int2 padding, PaddingMode padding_mode) : padding_(padding), padding_mode_(padding_mode) {}
    Tensor<> pad(const Tensor<> &input, const int2 &padding) const;
    Tensor<> zero_pad(const Tensor<> &input, const int2 &padding) const;

private:
    int2 padding_;
    PaddingMode padding_mode_;
};

Tensor<>
convolution(const int2 &stride, const int2 &dilation, const vector<size_t> &output_shape, const Tensor<> &input, const Tensor<> &kernel, const Tensor<> &bias, bool use_bias);

const vector<size_t> calculate_output_shape(const vector<size_t> &input_shape, const int64_t out_channel, const int2 &kernel_size, const int2 &stride, const int2 &padding, const int2 &dilation);

Tensor<> flip_vertical_and_horizontal(const Tensor<> &input);

Tensor<> dilate_input(const Tensor<> &input, const int2 &dilation);