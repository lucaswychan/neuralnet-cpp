#pragma once
#include "module.hpp"

namespace nn
{

    class Linear : public Module
    {
    public:
        Linear(size_t in_features, size_t out_features, bool bias);

        virtual Tensor<> forward(const Tensor<> &input) override;
        virtual Tensor<> backward(const Tensor<> &grad_output) override;
        virtual void update_params(const float lr) override;

        void reset_parameters();

        // setters
        inline void set_weight(const Tensor<> &target_weight) { this->weight_ = target_weight; };
        inline void set_bias(const Tensor<> &target_bias) { this->bias_ = target_bias; }

        // getters
        inline const Tensor<> &get_weight() const { return this->weight_; }
        inline const Tensor<> &get_bias() const { return this->bias_; }

    private:
        size_t in_features_;
        size_t out_features_;
        bool use_bias_;
        Tensor<> weight_;
        Tensor<> bias_;
        Tensor<> grad_weight_;
        Tensor<> grad_bias_;
    };

}