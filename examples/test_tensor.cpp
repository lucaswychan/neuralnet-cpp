#include "tensor.hpp"

int main() {
    Tensor<float> tensor_1d {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    Tensor<float> tensor_2d {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}};
    Tensor<float> tensor_3d {{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}}, {{10.0f, 11.0f, 12.0f}, {13.0f, 14.0f, 15.0f}, {16.0f, 17.0f, 18.0f}}};

    Tensor<> tensor_random({2, 4}, -2.0f);

    Tensor<float> tensor_negative {{-1.1f, -2.1f, -3.1f}, {-4.0f, -5.0f, -6.0f}, {-7.0f, -8.0f, -9.0f}};

    Tensor<float> copy_tensor_2d = tensor_2d;

    cout << "Tensor 1D: " << endl;
    tensor_1d.print();
    cout << endl;

    cout << "Tensor 2D: " << endl;
    tensor_2d.print();
    cout << endl;

    cout << "Tensor 3D: " << endl;
    tensor_3d.print();
    cout << endl;

    cout << "Tensor Random: " << endl;
    tensor_random.print();
    cout << endl;

    Tensor<float> transposed_2d = tensor_2d.transpose();

    cout << "Transposed Tensor 2D: " << endl;
    transposed_2d.print();
    cout << endl;

    Tensor<float> transposed_1d = tensor_1d.transpose();

    cout << "Transposed Tensor 1D: " << endl;
    transposed_1d.print();
    cout << endl;

    Tensor<float> mat_mul_tensor = tensor_2d.matmul(transposed_2d);

    cout << "Matrix Multiplication Result: " << endl;
    mat_mul_tensor.print();
    cout << endl;

    Tensor<float> mul_tensor_vector = tensor_2d * transposed_2d;

    cout << "Vector Multiplication Result: " << endl;
    mul_tensor_vector.print();
    cout << endl;

    tensor_2d[1, 2] = 100.0f;

    tensor_2d[0, 2] = 202.0f;

    tensor_2d[2, 2] = 200.0f;

    auto row1 = tensor_2d.row(1);

    row1[0] = 300.0f;
    row1[1] = 400.0f;
    row1[2] = 500.0f;
    cout << endl;

    cout << "Updated Tensor 2D: " << endl;
    tensor_2d.print();
    cout << endl;

    auto row_3d_1 = tensor_3d.col(0);

    row_3d_1[0] = 300.0f;
    row_3d_1[1] = 400.0f;
    row_3d_1[2] = 400.0f;
    cout << endl;


    cout << "Updated Tensor 3D: " << endl;
    tensor_3d.print();
    cout << endl;

    Tensor<float> tensor_positive = tensor_negative.abs();

    cout << "Positive Tensor: " << endl;
    tensor_positive.print();
    cout << endl;

    cout << "copy tensor 2d: " << endl;
    copy_tensor_2d.print();
    cout << endl;

    tensor_1d += tensor_1d;

    cout << "tensor_1d += tensor_1d: " << endl;
    tensor_1d.print();
    cout << endl;

    tensor_1d -= tensor_1d;

    cout << "tensor_1d -= tensor_1d: " << endl;
    tensor_1d.print();
    cout << endl;

    tensor_2d += tensor_2d;

    cout << "tensor_2d += tensor_2d: " << endl;
    tensor_2d.print();
    cout << endl;


    return 0;
}