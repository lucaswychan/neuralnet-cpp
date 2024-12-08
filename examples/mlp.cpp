#include "module.hpp"
#include "linear.hpp"
#include "relu.hpp"
using namespace nn;

class MLP : public Module {
    public:
        MLP(vector<int> layer_sizes) {
            this->num_layers_ = layer_sizes.size();

            for (int i = 0; i < this->num_layers_  - 1; i++) {
                this->layers_.push_back(new Linear(layer_sizes[i], layer_sizes[i + 1], true));
                this->layers_.push_back(new ReLU());
            }
        }

        virtual Tensor<> forward(const Tensor<>& input) override{
            Tensor<> x = input;

            for (Module* layer : this->layers_) {
                x = layer->forward(x);
            }

            return x;
        }

        virtual Tensor<> backward(const Tensor<>& grad_output) override {
            Tensor<> grad = grad_output;

            for (int i = this->num_layers_ - 1; i >= 0; i--) {
                grad = this->layers_[i]->backward(grad);
            }

            return grad;
        }
    private:
        vector<Module*> layers_;
        int num_layers_;
};