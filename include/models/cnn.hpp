#pragma once
#include "module.hpp"
using namespace nn;

class CNN : public Module
{
public:
    CNN(vector<size_t> layer_sizes, double dropout_p = 0.5);
    CNN(initializer_list<size_t> layer_sizes, double dropout_p = 0.5);
    ~CNN();

    virtual Tensor<> forward(const Tensor<> &input) override;
    virtual Tensor<> backward(const Tensor<> &grad_output) override;
    virtual void update_params(const float lr) override;

private:
    vector<Module *> layers_;
    int num_layers_;
};
