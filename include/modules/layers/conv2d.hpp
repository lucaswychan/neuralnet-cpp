#include "module.hpp"
using namespace nn;

namespace nn {

class Conv2d : public Module {
    public:
        Conv2d(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, int dilation = 1, bool bias = true);
        virtual Tensor<> forward(const Tensor<>& input) override;
        virtual Tensor<> backward(const Tensor<>& grad_output) override;
        virtual void update_params(const float lr) override;

        Tensor<> convolution(const Tensor<>& input, const Tensor<> filter);
    
    private:
        int64_t in_channels_;
        int64_t out_channels_;
        tuple<int64_t, int64_t> kernel_size_;
        tuple<int64_t, int64_t> stride_;
        tuple<int64_t, int64_t> padding_;
        tuple<int64_t, int64_t> dilation_;
        bool bias_;
        Tensor<> weights_;
        Tensor<> biases_;
        Tensor<> grad_weights_;
        Tensor<> grad_biases_;
};

}