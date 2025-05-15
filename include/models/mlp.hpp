#pragma once
#include "module.hpp"
#include "sequential.hpp"
using namespace nn;


class MLP : public Module {
    public:
        MLP(size_t in_channels, initializer_list<size_t> hidden_channels, bool bias = true, float dropout = 0.0);

        virtual Tensor<> forward(const Tensor<>& input) override;
        virtual Tensor<> backward(const Tensor<>& grad_output) override;
        virtual void update_params(const float lr) override;
        virtual Module& train(const bool mode = true) override;
        virtual Module& eval() override;

        inline const Sequential& get_layers() const { return this->layers_; }

    private:
        Sequential layers_; // we use a sequential container to store the layers
};