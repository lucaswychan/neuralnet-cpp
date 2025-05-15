#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "tensor.hpp"
#include "math.h"

TEST_CASE("TensorTest - Constructor and Destructor")
{
    Tensor<> tensor;
    // No explicit assertions needed, just verify no crashes
}

TEST_CASE("TensorTest - Scaler Constructor")
{
    Tensor<> tensor(10.0f);
    CHECK(tensor.ndim() == 1);
    CHECK(tensor.size() == 1);
    CHECK(tensor.shapes()[0] == 1);
    CHECK(tensor[0] == 10);
}

TEST_CASE("TensorTest - 1D Tensor Constructor from initializer_list")
{
    Tensor<> tensor_1d = {1.0f, 2.0f, 3.0f, 4.0f};
    CHECK(tensor_1d.ndim() == 1);
    CHECK(tensor_1d.size() == 4);
    CHECK(tensor_1d.shapes()[0] == 4);
    CHECK(tensor_1d[0] == 1.0f);
    CHECK(tensor_1d[1] == 2.0f);
    CHECK(tensor_1d[2] == 3.0f);
    CHECK(tensor_1d[3] == 4.0f);

    Tensor<> tensor_1d_1val = {0};
    CHECK(tensor_1d_1val.ndim() == 1);
    CHECK(tensor_1d_1val.size() == 1);
    CHECK(tensor_1d_1val.shapes()[0] == 1);
    CHECK(tensor_1d_1val[0] == 0.0f);
}

TEST_CASE("TensorTest - 2D Tensor Constructor from initializer_list")
{
    Tensor<> tensor_2d = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    CHECK(tensor_2d.ndim() == 2);
    CHECK(tensor_2d.size() == 4);
    CHECK(tensor_2d.shapes()[0] == 2);
    CHECK(tensor_2d.shapes()[1] == 2);
    CHECK(tensor_2d[0, 0] == 1.0f);
    CHECK(tensor_2d[0, 1] == 2.0f);
    CHECK(tensor_2d[1, 0] == 3.0f);
    CHECK(tensor_2d[1, 1] == 4.0f);

    Tensor<> tensor_2d_1row = {{0.0f, 0.0f}};
    CHECK(tensor_2d_1row.ndim() == 2);
    CHECK(tensor_2d_1row.size() == 2);
    CHECK(tensor_2d_1row.shapes()[0] == 1);
    CHECK(tensor_2d_1row.shapes()[1] == 2);
    CHECK(tensor_2d_1row[0, 0] == 0.0f);
    CHECK(tensor_2d_1row[0, 1] == 0.0f);

    // Tensor<> tensor_2d_1col({{0.0f}, {0.0f}});
    // CHECK(tensor_2d_1col.shapes()[0] == 2);
    // CHECK(tensor_2d_1col.shapes()[1] == 1);
    // CHECK(tensor_2d_1col.ndim() == 2);
    // CHECK(tensor_2d_1col.size() == 2);
    // CHECK(tensor_2d_1col[0, 0] == 0.0f);
    // CHECK(tensor_2d_1col[1, 0] == 0.0f);
}

TEST_CASE("TensorTest - 3D Tensor Constructor from initializer_list")
{
    Tensor<> tensor = {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}};
    CHECK(tensor.ndim() == 3);
    CHECK(tensor.size() == 8);
    CHECK(tensor.shapes()[0] == 2);
    CHECK(tensor.shapes()[1] == 2);
    CHECK(tensor.shapes()[2] == 2);
    CHECK(tensor[0, 0, 0] == 1.0f);
    CHECK(tensor[0, 0, 1] == 2.0f);
    CHECK(tensor[0, 1, 0] == 3.0f);
    CHECK(tensor[0, 1, 1] == 4.0f);
    CHECK(tensor[1, 1, 1] == 8.0f);

    Tensor<> tensor2 = {{{0.0f, 0.0f}, {0.0f, 0.0f}}, {{0.0f, 0.0f}, {0.0f, 0.0f}}};
    CHECK(tensor2.ndim() == 3);
    CHECK(tensor2.size() == 8);
    CHECK(tensor2.shapes()[0] == 2);
    CHECK(tensor2.shapes()[1] == 2);
    CHECK(tensor2.shapes()[2] == 2);
    CHECK(tensor2[0, 0, 0] == 0.0f);
    CHECK(tensor2[0, 0, 1] == 0.0f);
    CHECK(tensor2[0, 1, 0] == 0.0f);
    CHECK(tensor2[0, 1, 1] == 0.0f);
    CHECK(tensor2[1, 1, 1] == 0.0f);
}

TEST_CASE("TensorTest - 1D Tensor Constructor from vector")
{
    vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor<> tensor1 = data;
    CHECK(tensor1.ndim() == 1);
    CHECK(tensor1.size() == 4);
    CHECK(tensor1.shapes()[0] == 4);
    CHECK(tensor1[0] == 1.0f);
    CHECK(tensor1[1] == 2.0f);
    CHECK(tensor1[2] == 3.0f);
    CHECK(tensor1[3] == 4.0f);

    vector<float> data2 = {0};
    Tensor<> tensor2 = data2;
    CHECK(tensor2.shapes()[0] == 1);
    CHECK(tensor2.ndim() == 1);
    CHECK(tensor2.size() == 1);
    CHECK(tensor2[0] == 0.0f);
}

TEST_CASE("TensorTest - 2D Tensor Constructor from vector")
{
    vector<vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Tensor<> tensor = data;
    CHECK(tensor.ndim() == 2);
    CHECK(tensor.size() == 4);
    CHECK(tensor.shapes()[0] == 2);
    CHECK(tensor.shapes()[1] == 2);
    CHECK(tensor[0, 0] == 1.0f);
    CHECK(tensor[0, 1] == 2.0f);
    CHECK(tensor[1, 0] == 3.0f);
    CHECK(tensor[1, 1] == 4.0f);

    vector<vector<float>> data2 = {{0.0f, 0.0f}};
    Tensor<> tensor2 = data2;
    CHECK(tensor2.ndim() == 2);
    CHECK(tensor2.size() == 2);
    CHECK(tensor2.shapes()[0] == 1);
    CHECK(tensor2.shapes()[1] == 2);
    CHECK(tensor2[0, 0] == 0.0f);
    CHECK(tensor2[0, 1] == 0.0f);
}

TEST_CASE("TensorTest - 3D Tensor Constructor from vector")
{
    vector<vector<vector<float>>> data = {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}};
    Tensor<> tensor = data;
    CHECK(tensor.ndim() == 3);
    CHECK(tensor.size() == 8);
    CHECK(tensor.shapes()[0] == 2);
    CHECK(tensor.shapes()[1] == 2);
    CHECK(tensor.shapes()[2] == 2);
    CHECK(tensor[0, 0, 0] == 1.0f);
    CHECK(tensor[0, 0, 1] == 2.0f);
    CHECK(tensor[0, 1, 0] == 3.0f);
    CHECK(tensor[0, 1, 1] == 4.0f);
    CHECK(tensor[1, 1, 1] == 8.0f);

    vector<vector<vector<float>>> data2 = {{{0.0f, 0.0f}, {0.0f, 0.0f}}, {{0.0f, 0.0f}, {0.0f, 0.0f}}};
    Tensor<> tensor2 = data2;
    CHECK(tensor2.ndim() == 3);
    CHECK(tensor2.size() == 8);
    CHECK(tensor2.shapes()[0] == 2);
    CHECK(tensor2.shapes()[1] == 2);
    CHECK(tensor2.shapes()[2] == 2);
    CHECK(tensor2[0, 0, 0] == 0.0f);
    CHECK(tensor2[0, 0, 1] == 0.0f);
    CHECK(tensor2[0, 1, 0] == 0.0f);
    CHECK(tensor2[0, 1, 1] == 0.0f);
    CHECK(tensor2[1, 1, 1] == 0.0f);
}

TEST_CASE("TensorTest - 4D Tensor Constructor from vector")
{
    vector<vector<vector<vector<float>>>> data = {{{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}}, {{{9.0f, 10.0f}, {11.0f, 12.0f}}, {{13.0f, 14.0f}, {15.0f, 16.0f}}}};
    Tensor<> tensor = data;
    CHECK(tensor.ndim() == 4);
    CHECK(tensor.size() == 16);
    CHECK(tensor.shapes()[0] == 2);
    CHECK(tensor.shapes()[1] == 2);
    CHECK(tensor.shapes()[2] == 2);
    CHECK(tensor.shapes()[2] == 2);
    CHECK(tensor[0, 0, 0, 0] == 1.0f);
    CHECK(tensor[0, 0, 0, 1] == 2.0f);
    CHECK(tensor[0, 0, 1, 0] == 3.0f);
    CHECK(tensor[0, 0, 1, 1] == 4.0f);
    CHECK(tensor[1, 1, 1, 1] == 16.0f);
}

TEST_CASE("TensorTest - Copy Constructor")
{
    // 1D tensor
    Tensor<> tensor1 = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor<> test_tensor = tensor1;
    CHECK(test_tensor.ndim() == 1);
    CHECK(test_tensor.size() == 4);
    CHECK(test_tensor.shapes()[0] == 4);
    CHECK(test_tensor[0] == 1.0f);
    CHECK(test_tensor[1] == 2.0f);
    CHECK(test_tensor[2] == 3.0f);
    CHECK(test_tensor[3] == 4.0f);

    // Scalar tensor
    Tensor<> tensor2 = {0.0f};
    test_tensor = tensor2;
    CHECK(test_tensor.ndim() == 1);
    CHECK(test_tensor.size() == 1);
    CHECK(test_tensor.shapes()[0] == 1);
    CHECK(test_tensor[0] == 0.0f);

    // 2D tensor
    Tensor<> tensor3 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    test_tensor = tensor3;
    CHECK(test_tensor.ndim() == 2);
    CHECK(test_tensor.size() == 4);
    CHECK(test_tensor.shapes()[0] == 2);
    CHECK(test_tensor.shapes()[1] == 2);
    CHECK(test_tensor[0, 0] == 1.0f);
    CHECK(test_tensor[0, 1] == 2.0f);
    CHECK(test_tensor[1, 0] == 3.0f);
    CHECK(test_tensor[1, 1] == 4.0f);

    // 3D tensor
    Tensor<> tensor4 = {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}};
    test_tensor = tensor4;
    CHECK(test_tensor.ndim() == 3);
    CHECK(test_tensor.size() == 8);
    CHECK(test_tensor.shapes()[0] == 2);
    CHECK(test_tensor.shapes()[1] == 2);
    CHECK(test_tensor.shapes()[2] == 2);
    CHECK(test_tensor[0, 0, 0] == 1.0f);
    CHECK(test_tensor[0, 0, 1] == 2.0f);
    CHECK(test_tensor[0, 1, 0] == 3.0f);
    CHECK(test_tensor[0, 1, 1] == 4.0f);
    CHECK(test_tensor[1, 1, 1] == 8.0f);
}

TEST_CASE("TensorTest - Move Constructor")
{
    // 1D tensor
    Tensor<> tensor1 = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor<> test_tensor = std::move(tensor1);
    CHECK(test_tensor.ndim() == 1);
    CHECK(test_tensor.size() == 4);
    CHECK(test_tensor.shapes()[0] == 4);
    CHECK(test_tensor[0] == 1.0f);
    CHECK(test_tensor[1] == 2.0f);
    CHECK(test_tensor[2] == 3.0f);
    CHECK(test_tensor[3] == 4.0f);

    // Scalar tensor
    Tensor<> tensor2 = {0.0f};
    test_tensor = std::move(tensor2);
    CHECK(test_tensor.ndim() == 1);
    CHECK(test_tensor.size() == 1);
    CHECK(test_tensor.shapes()[0] == 1);
    CHECK(test_tensor[0] == 0.0f);

    // 2D tensor
    Tensor<> tensor3 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    test_tensor = std::move(tensor3);
    CHECK(test_tensor.ndim() == 2);
    CHECK(test_tensor.size() == 4);
    CHECK(test_tensor.shapes()[0] == 2);
    CHECK(test_tensor.shapes()[1] == 2);
    CHECK(test_tensor[0, 0] == 1.0f);
    CHECK(test_tensor[0, 1] == 2.0f);
    CHECK(test_tensor[1, 0] == 3.0f);
    CHECK(test_tensor[1, 1] == 4.0f);

    // 3D tensor
    Tensor<> tensor4 = {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}};
    test_tensor = std::move(tensor4);
    CHECK(test_tensor.ndim() == 3);
    CHECK(test_tensor.size() == 8);
    CHECK(test_tensor.shapes()[0] == 2);
    CHECK(test_tensor.shapes()[1] == 2);
    CHECK(test_tensor.shapes()[2] == 2);
    CHECK(test_tensor[0, 0, 0] == 1.0f);
    CHECK(test_tensor[0, 0, 1] == 2.0f);
    CHECK(test_tensor[0, 1, 0] == 3.0f);
    CHECK(test_tensor[0, 1, 1] == 4.0f);
    CHECK(test_tensor[1, 1, 1] == 8.0f);
}

TEST_CASE("TensorTest - Certain Value Constructor")
{
    Tensor<> tensor_1d({1}, 0.0f);
    CHECK(tensor_1d.ndim() == 1);
    CHECK(tensor_1d.size() == 1);
    CHECK(tensor_1d.shapes()[0] == 1);
    CHECK(tensor_1d[0] == 0.0f);

    Tensor<> tensor_2d({2, 2}, 10.0f);
    CHECK(tensor_2d.ndim() == 2);
    CHECK(tensor_2d.size() == 4);
    CHECK(tensor_2d.shapes()[0] == 2);
    CHECK(tensor_2d.shapes()[1] == 2);
    CHECK(tensor_2d[0, 0] == 10.0f);
    CHECK(tensor_2d[0, 1] == 10.0f);
    CHECK(tensor_2d[1, 0] == 10.0f);
    CHECK(tensor_2d[1, 1] == 10.0f);

    Tensor<> tensor_3d({2, 2, 2}, 5.0f);
    CHECK(tensor_3d.ndim() == 3);
    CHECK(tensor_3d.size() == 8);
    CHECK(tensor_3d.shapes()[0] == 2);
    CHECK(tensor_3d.shapes()[1] == 2);
    CHECK(tensor_3d.shapes()[2] == 2);
    CHECK(tensor_3d[0, 0, 0] == 5.0f);
    CHECK(tensor_3d[0, 0, 1] == 5.0f);
    CHECK(tensor_3d[0, 1, 0] == 5.0f);
    CHECK(tensor_3d[0, 1, 1] == 5.0f);
    CHECK(tensor_3d[1, 1, 1] == 5.0f);
}

TEST_CASE("TensorTest - Indexing Operator")
{
    Tensor<> tensor = {1.0f, 2.0f, 3.0f, 4.0f};
    CHECK(tensor[0] == 1.0f);
    CHECK(tensor[1] == 2.0f);
    CHECK(tensor[2] == 3.0f);
    CHECK(tensor[3] == 4.0f);

    tensor = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    CHECK(tensor[0, 0] == 1.0f);
    CHECK(tensor[0, 1] == 2.0f);
    CHECK(tensor[1, 0] == 3.0f);
    CHECK(tensor[1, 1] == 4.0f);

    tensor = {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}};
    CHECK(tensor[0, 0, 0] == 1.0f);
    CHECK(tensor[0, 0, 1] == 2.0f);
    CHECK(tensor[0, 1, 0] == 3.0f);
    CHECK(tensor[0, 1, 1] == 4.0f);
    CHECK(tensor[1, 1, 1] == 8.0f);
}

TEST_CASE("TensorTest - Indexing Operator - Out of Bound")
{
    Tensor<> tensor = {1.0f, 2.0f, 3.0f, 4.0f};
    CHECK_THROWS(tensor[4]);

    tensor = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    CHECK_THROWS(tensor[2, 0]);
    CHECK_THROWS(tensor[0, 2]);

    tensor = {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}};
    CHECK_THROWS(tensor[2, 0, 0]);
    CHECK_THROWS(tensor[0, 2, 0]);
    CHECK_THROWS(tensor[0, 0, 2]);
}

TEST_CASE("TensorTest - Indexing Operator - Negative Indexing")
{
    Tensor<> tensor = {1.0f, 2.0f, 3.0f, 4.0f};
    CHECK(tensor[-1] == 4.0f);
    CHECK(tensor[-2] == 3.0f);
    CHECK(tensor[-3] == 2.0f);
    CHECK(tensor[-4] == 1.0f);

    tensor = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    CHECK(tensor[-1, 0] == 3.0f);
    CHECK(tensor[-1, -1] == 4.0f);
    CHECK(tensor[0, -1] == 2.0f);

    tensor = {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}};
    CHECK(tensor[-1, 0, 0] == 5.0f);
    CHECK(tensor[-1, -1, -1] == 8.0f);
    CHECK(tensor[0, -1, 0] == 3.0f);
}

TEST_CASE("TensorTest - Indexing Operator - Normal Slicing")
{
    Tensor<> tensor_1d = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor<> sliced_tensor_1d_1 = tensor_1d.index({":2"});
    CHECK(sliced_tensor_1d_1.ndim() == 1);
    CHECK(sliced_tensor_1d_1.size() == 2);
    CHECK(sliced_tensor_1d_1.shapes()[0] == 2);
    CHECK(sliced_tensor_1d_1[0] == 1.0f);
    CHECK(sliced_tensor_1d_1[1] == 2.0f);

    Tensor<> sliced_tensor_1d_2 = tensor_1d.index({"1:"});
    CHECK(sliced_tensor_1d_2.ndim() == 1);
    CHECK(sliced_tensor_1d_2.size() == 3);
    CHECK(sliced_tensor_1d_2.shapes()[0] == 3);
    CHECK(sliced_tensor_1d_2[0] == 2.0f);
    CHECK(sliced_tensor_1d_2[1] == 3.0f);
    CHECK(sliced_tensor_1d_2[2] == 4.0f);

    Tensor<> sliced_tensor_1d_3 = tensor_1d.index({":"});
    CHECK(sliced_tensor_1d_3.ndim() == 1);
    CHECK(sliced_tensor_1d_3.size() == 4);
    CHECK(sliced_tensor_1d_3 == tensor_1d);

    Tensor<> sliced_tensor_1d_4 = tensor_1d.index({"1:3"});
    CHECK(sliced_tensor_1d_4.ndim() == 1);
    CHECK(sliced_tensor_1d_4.size() == 2);
    CHECK(sliced_tensor_1d_4.shapes()[0] == 2);
    CHECK(sliced_tensor_1d_4[0] == 2.0f);
    CHECK(sliced_tensor_1d_4[1] == 3.0f);

    Tensor<> tensor_2d = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Tensor<> sliced_tensor_2d_1 = tensor_2d.index({":", ":2"});
    CHECK(sliced_tensor_2d_1.ndim() == 2);
    CHECK(sliced_tensor_2d_1.size() == 4);
    CHECK(sliced_tensor_2d_1.shapes()[0] == 2);
    CHECK(sliced_tensor_2d_1.shapes()[1] == 2);
    CHECK(sliced_tensor_2d_1[0, 0] == 1.0f);
    CHECK(sliced_tensor_2d_1[0, 1] == 2.0f);
    CHECK(sliced_tensor_2d_1[1, 0] == 3.0f);
    CHECK(sliced_tensor_2d_1[1, 1] == 4.0f);
}

TEST_CASE("TensorTest - Transpose")
{
    Tensor<> tensor_2d = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Tensor<> transposed_tensor_2d = tensor_2d.transpose();
    CHECK(transposed_tensor_2d.ndim() == 2);
    CHECK(transposed_tensor_2d.size() == 4);
    CHECK(transposed_tensor_2d.shapes()[0] == 2);
    CHECK(transposed_tensor_2d.shapes()[1] == 2);
    CHECK(transposed_tensor_2d[0, 0] == 1.0f);
    CHECK(transposed_tensor_2d[0, 1] == 3.0f);
    CHECK(transposed_tensor_2d[1, 0] == 2.0f);
    CHECK(transposed_tensor_2d[1, 1] == 4.0f);

    Tensor<> tensor_1d = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor<> transposed_tensor_1d = tensor_1d.transpose();
    CHECK(transposed_tensor_1d.ndim() == 2);
    CHECK(transposed_tensor_1d.size() == 4);
    CHECK(transposed_tensor_1d.shapes()[0] == 4);
    CHECK(transposed_tensor_1d.shapes()[1] == 1);
    CHECK(transposed_tensor_1d[0, 0] == 1.0f);
    CHECK(transposed_tensor_1d[1, 0] == 2.0f);
    CHECK(transposed_tensor_1d[2, 0] == 3.0f);
    CHECK(transposed_tensor_1d[-1, -1] == 4.0f);
}

TEST_CASE("TensorTest - flatten")
{
    Tensor<> tensor_2d = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Tensor<> flattened_tensor_2d = tensor_2d.flatten();
    CHECK(flattened_tensor_2d.ndim() == 1);
    CHECK(flattened_tensor_2d.size() == 4);
    CHECK(flattened_tensor_2d.shapes()[0] == 4);
    CHECK(flattened_tensor_2d[0] == 1.0f);
    CHECK(flattened_tensor_2d[1] == 2.0f);
    CHECK(flattened_tensor_2d[2] == 3.0f);
    CHECK(flattened_tensor_2d[3] == 4.0f);
}

TEST_CASE("TensorTest - reshape")
{
    Tensor<> tensor_2d = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}};
    Tensor<> reshaped_tensor_2d = tensor_2d.reshape({2, 3, 2});
    CHECK(reshaped_tensor_2d.ndim() == 3);
    CHECK(reshaped_tensor_2d.size() == 12);
    CHECK(reshaped_tensor_2d.shapes()[0] == 2);
    CHECK(reshaped_tensor_2d.shapes()[1] == 3);
    CHECK(reshaped_tensor_2d.shapes()[2] == 2);
    CHECK(reshaped_tensor_2d[0, 0, 0] == 1.0f);
    CHECK(reshaped_tensor_2d[0, 0, 1] == 2.0f);
    CHECK(reshaped_tensor_2d[0, 1, 0] == 3.0f);
    CHECK(reshaped_tensor_2d[0, 1, 1] == 4.0f);
    CHECK(reshaped_tensor_2d[0, 2, 0] == 5.0f);
    CHECK(reshaped_tensor_2d[0, 2, 1] == 6.0f);
    CHECK(reshaped_tensor_2d[1, 0, 0] == 7.0f);
    CHECK(reshaped_tensor_2d[-1, -1, -1] == 12.0f);

    Tensor<> tensor_1d = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor<> reshaped_tensor_1d = tensor_1d.reshape({2, 3});
    CHECK(reshaped_tensor_1d.ndim() == 2);
    CHECK(reshaped_tensor_1d.size() == 6);
    CHECK(reshaped_tensor_1d.shapes()[0] == 2);
    CHECK(reshaped_tensor_1d.shapes()[1] == 3);
    CHECK(reshaped_tensor_1d[0, 0] == 1.0f);
    CHECK(reshaped_tensor_1d[0, 1] == 2.0f);
    CHECK(reshaped_tensor_1d[0, 2] == 3.0f);
    CHECK(reshaped_tensor_1d[1, 0] == 4.0f);
    CHECK(reshaped_tensor_1d[1, 1] == 5.0f);
    CHECK(reshaped_tensor_1d[1, 2] == 6.0f);
}

TEST_CASE("TensorTest - abs")
{
    Tensor<> tensor_2d = {{-1.0f, -2.0f}, {3.0f, 4.0f}};
    Tensor<> abs_tensor_2d = tensor_2d.abs();
    CHECK(abs_tensor_2d.ndim() == 2);
    CHECK(abs_tensor_2d.size() == 4);
    CHECK(abs_tensor_2d.shapes()[0] == 2);
    CHECK(abs_tensor_2d.shapes()[1] == 2);
    CHECK(abs_tensor_2d[0, 0] == 1.0f);
    CHECK(abs_tensor_2d[0, 1] == 2.0f);
    CHECK(abs_tensor_2d[1, 0] == 3.0f);
    CHECK(abs_tensor_2d[1, 1] == 4.0f);
}

TEST_CASE("TensorTest - sum")
{
    Tensor<> tensor_1d = {1.0f, 2.0f, 3.0f, 4.0f};
    float sum_1d = tensor_1d.sum();
    CHECK(sum_1d == 10.0f);

    Tensor<> tensor_2d = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    float sum_2d = tensor_2d.sum();
    CHECK(sum_2d == 10.0f);

    Tensor<> tensor_3d = {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}};
    float sum_3d = tensor_3d.sum();
    CHECK(sum_3d == 36.0f);
}

TEST_CASE("TensorTest - filter")
{
    Tensor<> tensor_1d = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor<> filtered_tensor_1d = tensor_1d.filter([](float x)
                                                   { return x < 3.0f; });
    CHECK(filtered_tensor_1d.ndim() == 1);
    CHECK(filtered_tensor_1d.size() == 4);
    CHECK(filtered_tensor_1d.shapes()[0] == 4);
    CHECK(filtered_tensor_1d[0] == 1.0f);
    CHECK(filtered_tensor_1d[1] == 2.0f);
    CHECK(filtered_tensor_1d[2] == 0.0f);
    CHECK(filtered_tensor_1d[3] == 0.0f);

    Tensor<> tensor_2d = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Tensor<> filtered_tensor_2d = tensor_2d.filter([](float x)
                                                   { return x < 3.0f; });
    CHECK(filtered_tensor_2d.ndim() == 2);
    CHECK(filtered_tensor_2d.size() == 4);
    CHECK(filtered_tensor_2d.shapes()[0] == 2);
    CHECK(filtered_tensor_2d.shapes()[1] == 2);
    CHECK(filtered_tensor_2d[0, 0] == 1.0f);
    CHECK(filtered_tensor_2d[0, 1] == 2.0f);
    CHECK(filtered_tensor_2d[1, 0] == 0.0f);
    CHECK(filtered_tensor_2d[1, 1] == 0.0f);

    Tensor<> tensor_3d = {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}};
    Tensor<> filtered_tensor_3d = tensor_3d.filter([](float x)
                                                   { return x < 3.0f; });
    CHECK(filtered_tensor_3d.ndim() == 3);
    CHECK(filtered_tensor_3d.size() == 8);
    CHECK(filtered_tensor_3d.shapes()[0] == 2);
    CHECK(filtered_tensor_3d.shapes()[1] == 2);
    CHECK(filtered_tensor_3d.shapes()[2] == 2);
    CHECK(filtered_tensor_3d[0, 0, 0] == 1.0f);
    CHECK(filtered_tensor_3d[0, 0, 1] == 2.0f);
    CHECK(filtered_tensor_3d[0, 1, 0] == 0.0f);
    CHECK(filtered_tensor_3d[0, 1, 1] == 0.0f);
    CHECK(filtered_tensor_3d[1, 0, 0] == 0.0f);
    CHECK(filtered_tensor_3d[1, 0, 1] == 0.0f);
    CHECK(filtered_tensor_3d[1, 1, 0] == 0.0f);
    CHECK(filtered_tensor_3d[1, 1, 1] == 0.0f);
}

TEST_CASE("TensorTest - map")
{
    float eps = 1e-5f;

    Tensor<> tensor_1d = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor<> tensor_1d_exp = tensor_1d.map([](float x)
                                           { return exp(x); });
    CHECK(tensor_1d_exp.ndim() == 1);
    CHECK(tensor_1d_exp.size() == 4);
    CHECK(tensor_1d_exp.shapes()[0] == 4);
    CHECK(tensor_1d_exp[0] - exp(1.0f) < eps);
    CHECK(tensor_1d_exp[1] - exp(2.0f) < eps);
    CHECK(tensor_1d_exp[2] - exp(3.0f) < eps);
    CHECK(tensor_1d_exp[3] - exp(4.0f) < eps);

    Tensor<> tensor_2d = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Tensor<> tensor_2d_times_10 = tensor_2d.map([](float x)
                                                { return x * 10.0f; });
    CHECK(tensor_2d_times_10.ndim() == 2);
    CHECK(tensor_2d_times_10.size() == 4);
    CHECK(tensor_2d_times_10.shapes()[0] == 2);
    CHECK(tensor_2d_times_10.shapes()[1] == 2);
    CHECK(tensor_2d_times_10[0, 0] == 10.0f);
    CHECK(tensor_2d_times_10[0, 1] == 20.0f);
    CHECK(tensor_2d_times_10[1, 0] == 30.0f);
    CHECK(tensor_2d_times_10[1, 1] == 40.0f);

    Tensor<> tensor_3d = {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}};
    Tensor<> tensor_3d_log = tensor_3d.map([](float x)
                                           { return log(x); });
    CHECK(tensor_3d_log.ndim() == 3);
    CHECK(tensor_3d_log.size() == 8);
    CHECK(tensor_3d_log.shapes()[0] == 2);
    CHECK(tensor_3d_log.shapes()[1] == 2);
    CHECK(tensor_3d_log.shapes()[2] == 2);
    CHECK(tensor_3d_log[0, 0, 0] - log(1.0f) < eps);
    CHECK(tensor_3d_log[0, 0, 1] - log(2.0f) < eps);
    CHECK(tensor_3d_log[0, 1, 0] - log(3.0f) < eps);
    CHECK(tensor_3d_log[0, 1, 1] - log(4.0f) < eps);
    CHECK(tensor_3d_log[1, 0, 0] - log(5.0f) < eps);
    CHECK(tensor_3d_log[1, 0, 1] - log(6.0f) < eps);
    CHECK(tensor_3d_log[1, 1, 0] - log(7.0f) < eps);
    CHECK(tensor_3d_log[1, 1, 1] - log(8.0f) < eps);
}

TEST_CASE("TensorTest - equal")
{
    Tensor<> tensor_1d = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor<> another_tensor_1d = {1.0f, 2.0f, 5.0f, 4.0f};
    Tensor<int> equal_tensor_1d = tensor_1d.equal(another_tensor_1d);
    CHECK(equal_tensor_1d.ndim() == 1);
    CHECK(equal_tensor_1d.size() == 4);
    CHECK(equal_tensor_1d.shapes()[0] == 4);
    CHECK(equal_tensor_1d[0] == 1);
    CHECK(equal_tensor_1d[1] == 1);
    CHECK(equal_tensor_1d[2] == 0);
    CHECK(equal_tensor_1d[3] == 1);

    Tensor<> tensor_2d = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Tensor<> another_tensor_2d = {{9.0f, 2.0f}, {3.0f, 5.0f}};
    Tensor<int> equal_tensor_2d = tensor_2d.equal(another_tensor_2d);
    CHECK(equal_tensor_2d.ndim() == 2);
    CHECK(equal_tensor_2d.size() == 4);
    CHECK(equal_tensor_2d.shapes()[0] == 2);
    CHECK(equal_tensor_2d.shapes()[1] == 2);
    CHECK(equal_tensor_2d[0, 0] == 0);
    CHECK(equal_tensor_2d[0, 1] == 1);
    CHECK(equal_tensor_2d[1, 0] == 1);
    CHECK(equal_tensor_2d[1, 1] == 0);

    Tensor<> tensor_3d = {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}};
    Tensor<> another_tensor_3d = {{{9.0f, 2.0f}, {3.0f, 5.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}};
    Tensor<int> equal_tensor_3d = tensor_3d.equal(another_tensor_3d);
    CHECK(equal_tensor_3d.ndim() == 3);
    CHECK(equal_tensor_3d.size() == 8);
    CHECK(equal_tensor_3d.shapes()[0] == 2);
    CHECK(equal_tensor_3d.shapes()[1] == 2);
    CHECK(equal_tensor_3d.shapes()[2] == 2);
    CHECK(equal_tensor_3d[0, 0, 0] == 0);
    CHECK(equal_tensor_3d[0, 0, 1] == 1);
    CHECK(equal_tensor_3d[0, 1, 0] == 1);
    CHECK(equal_tensor_3d[0, 1, 1] == 0);
    CHECK(equal_tensor_3d[1, 0, 0] == 1);
    CHECK(equal_tensor_3d[1, 0, 1] == 1);
    CHECK(equal_tensor_3d[1, 1, 0] == 1);
    CHECK(equal_tensor_3d[1, 1, 1] == 1);
}

TEST_CASE("TensorTest - Matrix Multiplication")
{
    Tensor<> tensor_2d_1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Tensor<> transposed_tensor_2d_1 = tensor_2d_1.transpose();
    Tensor<> matrix_multiplication_2d_1 = tensor_2d_1.matmul(transposed_tensor_2d_1);

    CHECK(matrix_multiplication_2d_1.ndim() == 2);
    CHECK(matrix_multiplication_2d_1.size() == 4);
    CHECK(matrix_multiplication_2d_1.shapes()[0] == 2);
    CHECK(matrix_multiplication_2d_1.shapes()[1] == 2);
    CHECK(matrix_multiplication_2d_1[0, 0] == 5.0f);
    CHECK(matrix_multiplication_2d_1[0, 1] == 11.0f);
    CHECK(matrix_multiplication_2d_1[1, 0] == 11.0f);
    CHECK(matrix_multiplication_2d_1[1, 1] == 25.0f);
}