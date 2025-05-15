#include "tensor.hpp"
using namespace std;

using size_tp2 = std::pair<size_t, size_t>;
using var_pair = std::variant<size_t, size_tp2>;

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
    Padding(size_tp2 padding, PaddingMode padding_mode) : padding_(padding), padding_mode_(padding_mode) {}
    Tensor<> pad(const Tensor<> &input, const size_tp2 &padding) const;
    Tensor<> zero_pad(const Tensor<> &input, const size_tp2 &padding) const;

private:
    size_tp2 padding_;
    PaddingMode padding_mode_;
};

Tensor<>
convolution(const size_tp2 &stride, const size_tp2 &dilation, const vector<size_t> &output_shape, const Tensor<> &input, const Tensor<> &kernel, const Tensor<> &bias, bool use_bias);

const vector<size_t> calculate_output_shape(const vector<size_t> &input_shape, const int64_t out_channel, const size_tp2 &kernel_size, const size_tp2 &stride, const size_tp2 &padding, const size_tp2 &dilation);

Tensor<> flip_vertical_and_horizontal(const Tensor<> &input);

Tensor<> dilate_input(const Tensor<> &input, const size_tp2 &dilation);

// // Helper lambda to process variant parameters
// template <typename U>
// size_tp2 process_variant(U &&arg)
// {
//     using T = std::decay_t<U>;

//     if constexpr (std::is_same_v<T, size_t>)
//     {
//         if (arg < 0)
//         {
//             throw std::invalid_argument("Negative kernel size, stride, padding, or dilation is not supported");
//         }
//         return {arg, arg};
//     }
//     else
//     {
//         static_assert(std::is_same_v<T, size_tp2>, "Unexpected type in variant");
//         return arg;
//     }
// }