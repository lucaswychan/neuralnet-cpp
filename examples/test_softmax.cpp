#include <linear.hpp>
#include <softmax.hpp>
#include <cross_entropy.hpp>
using namespace nn;

int main() {
    const bool bias = true;

    Linear linear_1(3, 5, bias);
    Linear linear_2(5, 7, bias);
    // Dropout dropout(0.3);

    Tensor<> specific_weights_1 = {
        {0.1, 0.4, 0.7, 1.0, 1.3},
        {0.2, 0.5, 0.8, 1.1, 1.4},
        {0.3, 0.6, 0.9, 1.2, 1.5}
    };
    
    Tensor<> specific_weights_2 = {
        {0.1, 0.6, 1.1, 1.6, 2.1, 2.6, 3.1},
        {0.2, 0.7, 1.2, 1.7, 2.2, 2.7, 3.2},
        {0.3, 0.8, 1.3, 1.8, 2.3, 2.8, 3.3},
        {0.4, 0.9, 1.4, 1.9, 2.4, 2.9, 3.4},
        {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5}
    };

    Tensor<> specific_bias_1 = {
        0.1,
        0.2,
        0.3,
        0.4,
        0.5
    };

    Tensor<> specific_bias_2 = {
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7
    };

    Tensor<> input = {
        {1.1f, 2.1f, 3.1f},
        {4.1f, 5.1f, 6.1f},
        {7.1f, 8.1f, 9.1f},
        {10.1f, 11.1f, 12.1f}
    };

    Tensor<> label = {6, 4, 3, 2};

    specific_bias_1 = specific_bias_1.transpose();
    specific_bias_2 = specific_bias_2.transpose();

    linear_1.set_weights(specific_weights_1);
    linear_2.set_weights(specific_weights_2);

    linear_1.set_biases(specific_bias_1);
    linear_2.set_biases(specific_bias_2);

    cout << endl;

    Softmax softmax = Softmax();
    CrossEntropyLoss criterion = CrossEntropyLoss();

    Tensor<> output_1 = linear_1.forward(input);
    Tensor<> output_softmax = softmax.forward(output_1);
    Tensor<> output_2 = linear_2.forward(output_softmax);

    double cross_entropy_loss = criterion.forward(output_2, label);

    cout << "cross entropy loss: " << cross_entropy_loss << endl;


    return 0;
}