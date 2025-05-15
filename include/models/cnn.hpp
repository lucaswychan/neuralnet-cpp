#pragma once
#include "module.hpp"
using namespace nn;

class CNN : public Module
{
public:
    CNN(vector<size_t> layer_sizes, float dropout_p = 0.5);
    CNN(initializer_list<size_t> layer_sizes, float dropout_p = 0.5);
    ~CNN();

    virtual Tensor<> forward(const Tensor<> &input) override;
    virtual Tensor<> backward(const Tensor<> &grad_output) override;
    virtual void update_params(const float lr) override;

protected:
    /**
     * Applies a function to all child modules.
     * This implementation iterates through all layers and applies the function to each.
     * 
     * @param fn A function that takes a Module reference and returns void.
     */
    virtual void apply_to_children(const function<void(Module&)>& fn) override;

private:
    vector<Module *> layers_;
    int num_layers_;
};
