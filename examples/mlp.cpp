#include "module.hpp"
#include "linear.hpp"
using namespace nn;

class MLP : public Module {
    public:
        MLP(vector<int> layer_sizes) {
            this->num_layers_ = layer_sizes.size();
            this->layers_.resize(this->num_layers_ );

            for (int i = 0; i < this->num_layers_  - 1; i++) {
                this->layers_[i] = new Linear(layer_sizes[i], layer_sizes[i + 1], true);
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