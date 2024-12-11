#pragma once
#include "module.hpp"
using namespace nn;

class MLP : public Module {
    public:
        MLP(vector<size_t> layer_sizes);
        MLP(initializer_list<size_t> layer_sizes);
        ~MLP();

        virtual Tensor<> forward(const Tensor<>& input) override;
        virtual Tensor<> backward(const Tensor<>& grad_output) override;
        virtual void update_params(const float lr) override;

    private:
        vector<Module*> layers_;
        int num_layers_;
};