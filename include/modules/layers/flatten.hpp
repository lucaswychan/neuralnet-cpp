#pragma once
#include "module.hpp"

namespace nn
{

    class Flatten : public Module
    {
    public:
        Flatten(int64_t start_dim = 1, int64_t end_dim = -1);

        virtual Tensor<> forward(const Tensor<> &input) override;
        virtual Tensor<> backward(const Tensor<> &grad_output) override;
        virtual void update_params(const float lr) override;

    private:
        int64_t start_dim_;
        int64_t end_dim_;
        vector<size_t> original_shape_;
    };

}