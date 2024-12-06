#include "module.hpp"
using namespace std;

class Linear : Module{
public:
    Linear(int in_features, int out_features);
    
    float** forward(const float** input) override;
    float** backward(const float** grad_output) override;
    void update_params(const float lr) override;

private:
    int in_features;
    int out_features;
    float** weights;
    float* biases;
};  