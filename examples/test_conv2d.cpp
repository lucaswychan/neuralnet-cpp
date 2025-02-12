#include "conv2d.hpp"
using namespace nn;

int main()
{
    size_t batch_size = 1;
    size_t in_channels = 4;
    size_t out_channels = 8;
    size_t weight_size = 3;
    size_t input_data_size = 15;
    size_t padding = 3;
    size_t stride = 2;
    size_t dilation = 2;
    string padding_mode = "zeros";
    bool use_bias = true;

    size_t out_channels_2 = 7;
    size_t weight_size_2 = 3;
    size_t stride_2 = 2;
    size_t padding_2 = 3;
    size_t dilation_2 = 4;
    string padding_mode_2 = "zeros";
    bool use_bias_2 = true;

    Tensor<> test_weight = Tensor<>({out_channels, in_channels, weight_size, weight_size}, 0.0f);
    Tensor<> test_bias = Tensor<>({out_channels}, 0.0f);

    size_t val = 1;
    for (size_t i = 0; i < out_channels; i++)
    {
        for (size_t j = 0; j < in_channels; j++)
        {
            for (size_t k = 0; k < weight_size; k++)
            {
                for (size_t l = 0; l < weight_size; l++)
                {
                    test_weight[i, j, k, l] = val;
                    val++;
                }
            }
        }
    }

    val = 1;
    for (size_t i = 0; i < out_channels; i++)
    {
        test_bias[i] = val;
        val++;
    }

    Tensor<> test_weight_2 = Tensor<>({out_channels_2, out_channels, weight_size_2, weight_size_2}, 0.0f);
    Tensor<> test_bias_2 = Tensor<>({out_channels_2}, 0.0f);

    double val_2 = 0.1;
    for (size_t i = 0; i < out_channels_2; i++)
    {
        for (size_t j = 0; j < out_channels; j++)
        {
            for (size_t k = 0; k < weight_size_2; k++)
            {
                for (size_t l = 0; l < weight_size_2; l++)
                {
                    test_weight_2[i, j, k, l] = val_2;
                    val_2 += 0.1;
                }
            }
        }
    }

    val_2 = 0.1;
    for (size_t i = 0; i < out_channels_2; i++)
    {
        test_bias_2[i] = val_2;
        val_2 += 0.1;
    }

    Tensor<> test_input = Tensor<>({batch_size, in_channels, input_data_size, input_data_size}, 0.0f);

    val = 1;
    for (size_t i = 0; i < batch_size; i++)
    {
        for (size_t j = 0; j < in_channels; j++)
        {
            for (size_t k = 0; k < input_data_size; k++)
            {
                for (size_t l = 0; l < input_data_size; l++)
                {
                    test_input[i, j, k, l] = val;
                    val++;
                }
            }
        }
    }
    // cout << "Test weight: " << endl;
    // test_weight.print();
    // cout << endl;

    cout << "Test bias: " << endl;
    test_bias.print();
    cout << endl;

    Conv2d conv2d_1(in_channels, out_channels, weight_size, padding, stride, dilation, padding_mode, use_bias);

    cout << "Conv2d layer 1 initialized with in_channels = " << in_channels << " and out_channels = " << out_channels << endl;

    Conv2d conv2d_2(out_channels, out_channels_2, weight_size_2, padding_2, stride_2, dilation_2, padding_mode_2, use_bias_2);

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

    return 0;
}