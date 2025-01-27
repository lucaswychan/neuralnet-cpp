#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "tensor.hpp"

TEST_CASE("TensorTest - Constructor and Destructor") {
    Tensor<> tensor;
    // No explicit assertions needed, just verify no crashes
}

TEST_CASE("TensorTest - Scaler Constructor") {
    Tensor<> tensor(10.0f);
    CHECK(tensor.ndim() == 1);
    CHECK(tensor.size() == 1);
    CHECK(tensor.shapes()[0] == 1);
    CHECK(tensor[0] == 10);
}

TEST_CASE("TensorTest - 1D Tensor Constructor from initializer_list") {
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

TEST_CASE("TensorTest - 2D Tensor Constructor from initializer_list") {
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

TEST_CASE("TensorTest - 3D Tensor Constructor from initializer_list") {
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

TEST_CASE("TensorTest - 1D Tensor Constructor from vector") {
    vector<double> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor<> tensor1 = data;
    CHECK(tensor1.ndim() == 1);
    CHECK(tensor1.size() == 4);
    CHECK(tensor1.shapes()[0] == 4);
    CHECK(tensor1[0] == 1.0f);
    CHECK(tensor1[1] == 2.0f);
    CHECK(tensor1[2] == 3.0f);
    CHECK(tensor1[3] == 4.0f);

    vector<double> data2 = {0};
    Tensor<> tensor2 = data2;
    CHECK(tensor2.shapes()[0] == 1);
    CHECK(tensor2.ndim() == 1);
    CHECK(tensor2.size() == 1);
    CHECK(tensor2[0] == 0.0f);
}

TEST_CASE("TensorTest - 2D Tensor Constructor from vector") {
    vector<vector<double>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Tensor<> tensor = data;
    CHECK(tensor.ndim() == 2);
    CHECK(tensor.size() == 4);
    CHECK(tensor.shapes()[0] == 2);
    CHECK(tensor.shapes()[1] == 2);
    CHECK(tensor[0, 0] == 1.0f);
    CHECK(tensor[0, 1] == 2.0f);
    CHECK(tensor[1, 0] == 3.0f);
    CHECK(tensor[1, 1] == 4.0f);
    
    vector<vector<double>> data2 = {{0.0f, 0.0f}};
    Tensor<> tensor2 = data2;
    CHECK(tensor2.ndim() == 2);
    CHECK(tensor2.size() == 2);
    CHECK(tensor2.shapes()[0] == 1);
    CHECK(tensor2.shapes()[1] == 2);
    CHECK(tensor2[0, 0] == 0.0f);
    CHECK(tensor2[0, 1] == 0.0f);
}

TEST_CASE("TensorTest - 3D Tensor Constructor from vector") {
    vector<vector<vector<double>>> data = {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}};
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

    vector<vector<vector<double>>> data2 = {{{0.0f, 0.0f}, {0.0f, 0.0f}}, {{0.0f, 0.0f}, {0.0f, 0.0f}}};
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

TEST_CASE("TensorTest - Copy Constructor") {
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

TEST_CASE("TensorTest - Move Constructor") {
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

TEST_CASE("TensorTest - Certain Value Constructor") {
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

TEST_CASE("TensorTest - Indexing Operator") {
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

TEST_CASE("TensorTest - Indexing Operator - Out of Bound") {
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

TEST_CASE("TensorTest - Indexing Operator - Negative Indexing") {
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

TEST_CASE("TensorTest - Indexing Operator - Normal Slicing") {
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

    Tensor<> sliced_tensor_1d_3 = tensor_1d.index({":2"});
    CHECK(sliced_tensor_1d_3.ndim() == 1);
    CHECK(sliced_tensor_1d_3.size() == 2);
    CHECK(sliced_tensor_1d_3.shapes()[0] == 2);
    CHECK(sliced_tensor_1d_3[0] == 1.0f);
    CHECK(sliced_tensor_1d_3[1] == 2.0f);

    Tensor<> sliced_tensor_1d_4 = tensor_1d.index({"1:3"});
    CHECK(sliced_tensor_1d_4.ndim() == 1);
    CHECK(sliced_tensor_1d_4.size() == 2);
    CHECK(sliced_tensor_1d_4.shapes()[0] == 2);
    CHECK(sliced_tensor_1d_4[0] == 2.0f);
    CHECK(sliced_tensor_1d_4[1] == 3.0f);

    // Tensor<> tensor_2d = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    // Tensor<> sliced_tensor_2d_1 = tensor_2d.index({":", ":2"});
    // CHECK(sliced_tensor_2d_1.ndim() == 2);
    // CHECK(sliced_tensor_2d_1.size() == 4);
    // CHECK(sliced_tensor_2d_1.shapes()[0] == 2);
    // CHECK(sliced_tensor_2d_1.shapes()[1] == 2);
    // CHECK(sliced_tensor_2d_1[0, 0] == 1.0f);
    // CHECK(sliced_tensor_2d_1[0, 1] == 2.0f);
    // CHECK(sliced_tensor_2d_1[1, 0] == 3.0f);
    // CHECK(sliced_tensor_2d_1[1, 1] == 4.0f);
}

