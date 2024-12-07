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

        virtual vector<vector<float>> forward(const vector<vector<float>>& input) override{
            
        }
    private:
        vector<Module*> layers_;
        int num_layers_;
};