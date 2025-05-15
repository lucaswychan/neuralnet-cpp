#include "mlp.hpp"
#include "linear.hpp"
#include "relu.hpp"
#include "dropout.hpp"

MLP::MLP(size_t in_channels, initializer_list<size_t> hidden_channels, bool bias, float dropout)
{
    vector<Module *> layers;
    size_t i = 0;
    for (size_t hidden_channel : hidden_channels)
    {
        layers.push_back(new Linear(in_channels, hidden_channel, bias));
        in_channels = hidden_channel; // update the input channels for the next layer

        if (i < hidden_channels.size() - 1)
        {
            layers.push_back(new ReLU());
            if (dropout > 0.0f) {
                layers.push_back(new Dropout(dropout));
            }
        }
        i++;
    }

    // Move the vector directly to the Sequential constructor
    this->layers_ = Sequential(std::move(layers));
}

Tensor<> MLP::forward(const Tensor<> &input)
{
    return this->layers_.forward(input);
}

Tensor<> MLP::backward(const Tensor<> &grad_output)
{
    return this->layers_.backward(grad_output);
}

void MLP::update_params(const float lr)
{
    this->layers_.update_params(lr);
}

Module& MLP::train(const bool mode) {
    this->layers_.train(mode);
    return *this;
}

Module& MLP::eval() {
    this->layers_.eval();
    return *this;
}