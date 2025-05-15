#include "conv2d.hpp"
#include "flatten.hpp"
#include "linear.hpp"
#include "cross_entropy.hpp"
using namespace nn;

int main()
{
    size_t batch_size = 2;
    size_t input_data_size = 15;

    size_t in_channels = 4;
    size_t out_channels = 8;
    size_t weight_size = 3;
    size_t padding = 3;
    size_t stride = 2;
    size_t dilation = 2;
    string padding_mode = "zeros";
    bool use_bias = true;

    size_t out_channels_2 = 7;
    size_t weight_size_2 = 3;
    size_t padding_2 = 3;
    size_t stride_2 = 2;
    size_t dilation_2 = 4;
    string padding_mode_2 = "zeros";
    bool use_bias_2 = true;

    size_t in_features = out_channels_2 * input_data_size * input_data_size;
    size_t out_features = 10;

    Tensor<> test_weight = Tensor<>({out_channels, in_channels, weight_size, weight_size}, 0.0f);
    Tensor<> test_bias = Tensor<>({out_channels}, 0.0f);

    float val = 0.01;
    for (size_t i = 0; i < out_channels; i++)
    {
        for (size_t j = 0; j < in_channels; j++)
        {
            for (size_t k = 0; k < weight_size; k++)
            {
                for (size_t l = 0; l < weight_size; l++)
                {
                    test_weight[i, j, k, l] = val;
                    val += 0.01;
                }
            }
        }
    }

    val = 0.01;
    for (size_t i = 0; i < out_channels; i++)
    {
        test_bias[i] = val;
        val += 0.01;
    }

    Tensor<> test_weight_2 = Tensor<>({out_channels_2, out_channels, weight_size_2, weight_size_2}, 0.0f);
    Tensor<> test_bias_2 = Tensor<>({out_channels_2}, 0.0f);

    val = 0.01;
    for (size_t i = 0; i < out_channels_2; i++)
    {
        for (size_t j = 0; j < out_channels; j++)
        {
            for (size_t k = 0; k < weight_size_2; k++)
            {
                for (size_t l = 0; l < weight_size_2; l++)
                {
                    test_weight_2[i, j, k, l] = val;
                    val += 0.01;
                }
            }
        }
    }

    val = 0.01;
    for (size_t i = 0; i < out_channels_2; i++)
    {
        test_bias_2[i] = val;
        val += 0.01;
    }

    Tensor<> test_input = Tensor<>({batch_size, in_channels, input_data_size, input_data_size}, 0.0f);

    val = 0.01;
    for (size_t i = 0; i < batch_size; i++)
    {
        for (size_t j = 0; j < in_channels; j++)
        {
            for (size_t k = 0; k < input_data_size; k++)
            {
                for (size_t l = 0; l < input_data_size; l++)
                {
                    test_input[i, j, k, l] = val;
                    val += 0.01;
                }
            }
        }
    }

    // cout << "Test input: " << endl;
    // test_input.print();
    // cout << endl;

    // cout << "Test weight: " << endl;
    // test_weight.print();
    // cout << endl;

    // cout << "Test bias: " << endl;
    // test_bias.print();
    // cout << endl;

    // cout << "Test weight 2: " << endl;
    // test_weight_2.print();
    // cout << endl;

    // cout << "Test bias 2: " << endl;
    // test_bias_2.print();
    // cout << endl;

    Conv2d conv2d_1(in_channels, out_channels, weight_size, padding, stride, dilation, padding_mode, use_bias);
    Conv2d conv2d_2(out_channels, out_channels_2, weight_size_2, padding_2, stride_2, dilation_2, padding_mode_2, use_bias_2);
    Flatten flatten;
    CrossEntropyLoss cross_entropy;

    conv2d_1.set_weight(test_weight);
    conv2d_1.set_bias(test_bias);

    conv2d_2.set_weight(test_weight_2);
    conv2d_2.set_bias(test_bias_2);

    // cout << "Test input: " << endl;
    // test_input.print();
    // cout << endl;

    Tensor<> output = conv2d_1(test_input);
    Tensor<> output_2 = conv2d_2(output);

    cout << "Output: " << endl;
    output.print();
    cout << endl;

    cout << "Output 2: " << endl;
    output_2.print();
    cout << endl;

    cout << "output shape : ";
    for (int i = 0; i < output.ndim(); i++)
    {
        cout << output.shapes()[i] << " ";
    }
    cout << endl;

    cout << "output_2 shape : ";
    for (int i = 0; i < output_2.ndim(); i++)
    {
        cout << output_2.shapes()[i] << " ";
    }
    cout << endl;

    Tensor<> flattened_output_2 = flatten(output_2);

    cout << "Flattened output 2: " << endl;
    flattened_output_2.print();
    cout << endl;

    Linear linear(flattened_output_2.shapes()[1], out_features, false);

    Tensor<> test_linear_weight({flattened_output_2.shapes()[1], out_features}, 0.0);

    val = 0.01;
    for (size_t i = 0; i < flattened_output_2.shapes()[1]; ++i)
    {
        for (size_t j = 0; j < out_features; ++j)
        {
            test_linear_weight[i, j] = val;
            val += 0.01;
        }
    }

    cout << "linear in features: " << flattened_output_2.shapes()[1] << endl;
    cout << "linear out features: " << out_features << endl;

    linear.set_weight(test_linear_weight);

    cout << "Test linear weight: " << endl;
    test_linear_weight.print();
    cout << endl;

    Tensor<> output_3 = linear(flattened_output_2);

    cout << "Output 3: " << endl;
    output_3.print();
    cout << endl;

    Tensor<> labels({batch_size}, 0);

    val = 1;
    for (size_t i = 0; i < batch_size; i++)
    {
        labels[i] = val;
        val++;
    }

    // output.print();
    // output_2.print();
    // output_3.print();

    output_3 /= 1e6;

    float loss = cross_entropy(output_3, labels);

    cout << "Loss: " << loss << endl;

    Tensor<> dL_dV_2 = cross_entropy.backward();
    Tensor<> dL_dV_1 = linear.backward(dL_dV_2);
    Tensor<> dL_dZ = flatten.backward(dL_dV_1);

    cout << "dL/dZ: " << endl;
    dL_dZ.print();
    cout << endl;

    Tensor<> dL_dY = conv2d_2.backward(dL_dZ);
    Tensor<> dL_dX = conv2d_1.backward(dL_dY);

    cout << "dL_dY: " << endl;
    dL_dY.print();
    cout << endl;

    cout << "dL_dX: " << endl;
    dL_dX.print();
    cout << endl;

    return 0;
}