#include "mlp.hpp"
#include "linear.hpp"
#include "relu.hpp"

MLP::MLP(vector<size_t> layer_sizes) {
    this->num_layers_ = layer_sizes.size();

    for (size_t i = 0; i < this->num_layers_  - 1; i++) {
        this->layers_.push_back(new Linear(layer_sizes[i], layer_sizes[i + 1], true));
        if (i < this->num_layers_ - 2) {
            this->layers_.push_back(new ReLU());
        }
    }
}

MLP::MLP(initializer_list<size_t> layer_sizes) : MLP(vector<size_t>(layer_sizes)) {}

MLP::~MLP() {
    for (Module* layer : this->layers_) {
        delete layer;
    }
}

Tensor<> MLP::forward(const Tensor<>& input) {
    Tensor<> x = input;

    for (Module* layer : this->layers_) {
        x = layer->forward(x);
    }

    return x;
}

Tensor<> MLP::backward(const Tensor<>& grad_output) {
    Tensor<> grad = grad_output;

    for (int i = this->layers_.size() - 1; i >= 0; i--) {
        grad = this->layers_[i]->backward(grad);
    }

    return grad;
}

void MLP::update_params(const float lr) {
    for (Module* layer : this->layers_) {
        layer->update_params(lr);
    }

    return;
}