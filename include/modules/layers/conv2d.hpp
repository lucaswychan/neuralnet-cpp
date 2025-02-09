#include <utility>
#include "module.hpp"
using namespace nn;

using int2 = std::pair<int64_t, int64_t>;
using var_pair = std::variant<int64_t, int2>;

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
               bool bias = true);

        virtual Tensor<> forward(const Tensor<> &input) override;
        virtual Tensor<> backward(const Tensor<> &grad_output) override;
        virtual void update_params(const float lr) override;

        Tensor<> convolution(const int2 &stride, const int2 &dilation, const vector<size_t> &output_shape, const Tensor<> &input, const Tensor<> &kernel, const Tensor<> &bias, bool use_bias);

        std::tuple<int64_t, int64_t, int64_t, int64_t> calculate_output_shape(const Tensor<> &input);

    private:
        int64_t in_channels_;
        int64_t out_channels_;
        int2 kernel_size_;
        int2 stride_;
        int2 padding_;
        int2 dilation_;
        bool use_bias_;
        vector<size_t> original_input_shape_;
        Tensor<> weight_;
        Tensor<> bias_;
        Tensor<> grad_weights_;
        Tensor<> grad_biases_;
    };
}