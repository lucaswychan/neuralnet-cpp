#include <math.h>
#include "tensor.hpp"
#include <chrono>
using namespace std::chrono;
using namespace std;

int main()
{
    Tensor<> tensor_1d{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    cout << "finished creating tensor_1d" << endl;

    Tensor<> tensor_2d{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}};
    cout << "finished creating tensor_2d" << endl;

    Tensor<> tensor_3d{{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}}, {{10.0f, 11.0f, 12.0f}, {13.0f, 14.0f, 15.0f}, {16.0f, 17.0f, 18.0f}}};
    cout << "finished creating tensor_3d" << endl;

    Tensor<> tensor_random({2, 4}, -2.0f);
    cout << "finished creating tensor_random" << endl;

    Tensor<> tensor_negative{{-1.1f, -2.1f, -3.1f}, {-4.0f, -5.0f, -6.0f}, {-7.0f, -8.0f, -9.0f}};
    cout << "finished creating tensor_negative" << endl;

    Tensor<> copy_tensor_2d = tensor_2d;
    cout << "finished creating copy_tensor_2d" << endl;

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

    Tensor<> transposed_2d = tensor_2d.transpose();

    cout << "Transposed Tensor 2D: " << endl;
    transposed_2d.print();
    cout << endl;

    Tensor<> transposed_1d = tensor_1d.transpose();

    cout << "Transposed Tensor 1D: " << endl;
    transposed_1d.print();
    cout << endl;

    Tensor<> mat_mul_tensor = tensor_2d.matmul(transposed_2d);

    cout << "Matrix Multiplication Result: " << endl;
    mat_mul_tensor.print();
    cout << endl;

    Tensor<> mul_tensor_vector = tensor_2d * transposed_2d;

    cout << "Vector Multiplication Result: " << endl;
    mul_tensor_vector.print();
    cout << endl;

    Tensor<> tensor_positive = tensor_negative.abs();

    cout << "Positive Tensor: " << endl;
    tensor_positive.print();
    cout << endl;

    cout << "copy tensor 2d: " << endl;
    copy_tensor_2d.print();
    cout << endl;

    Tensor<> origional_tensor_2d = tensor_2d.clone();

    tensor_2d += tensor_2d;

    cout << "tensor_2d += tensor_2d: " << endl;
    tensor_2d.print();
    cout << endl;

    Tensor<> filtered_tensor = tensor_3d.filter([](float value)
                                                { return value <= 10.0f; });

    cout << "fitlered_values <= 10: " << endl;
    filtered_tensor.print();
    cout << endl;

    cout << "finding max value in tensor_2d: " << endl;

    Tensor<size_t> max_tensor_2d = tensor_2d.argmin();

    cout << "max_tensor_2d: " << endl;
    max_tensor_2d.print();
    cout << endl;

    Tensor<size_t> max_tensor_1d = tensor_1d.argmin();

    cout << "max_tensor_1d: " << endl;
    max_tensor_1d.print();
    cout << endl;

    Tensor<int> equal_tensor_2d = tensor_2d.equal(tensor_2d);

    cout << "equal_tensor_2d: " << endl;
    equal_tensor_2d.print();
    cout << endl;

    Tensor<int> equal_tensor_2d_original_transposed = origional_tensor_2d.equal(transposed_2d);

    cout << "equal_tensor_2d_original_transposed: " << endl;
    equal_tensor_2d_original_transposed.print();
    cout << endl;

    Tensor<> first_row_tensor_3d = tensor_3d.index({":", 0u, ":"});

    cout << "first_row_tensor_3d: " << endl;
    first_row_tensor_3d.print();
    cout << endl;

    float last_value = tensor_3d[-1, 0, 1];
    cout << "last_value: " << last_value << endl;

    Tensor<> A = {{1, 2, 3},
                  {4, 5, 6}}; // 2 x 3

    Tensor<> A_mapped = A.map([](float x)
                              { return exp(x); });

    A_mapped.print();

    Tensor<> new_tensor_3d = {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}};
    Tensor<> tensor_3d_log = new_tensor_3d.map([](float x)
                                               { return log(x); });

    cout << "tensor_3d_log: " << endl;
    tensor_3d_log.print();
    cout << endl;

    auto start = high_resolution_clock::now();

    // Tensor<> really_large_tensor = Tensor<>({100, 100, 100}, 1.0f);

    // for (size_t i = 0; i < 100; i++) {
    //     cout << "iteration: " << i << endl;
    //     // really_large_tensor = really_large_tensor.map([](float x) { return exp(x); });
    //     Tensor<> really_large_tensor_transpose = really_large_tensor.transpose();
    //     size_t num_elements = really_large_tensor.size();
    // }

    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);

    // cout << "Time required : " << duration.count() << endl;

    Tensor<> permuted_tensor_3d = tensor_3d.permute(1, 0, 2);

    cout << "permuted_tensor_3d: " << endl;
    permuted_tensor_3d.print();
    cout << endl;

    Tensor<> reshaped_permuted_tensor_3d = permuted_tensor_3d.reshape({3, 6});

    cout << "reshaped permuted_tensor_3d: " << endl;
    reshaped_permuted_tensor_3d.print();
    cout << endl;

    Tensor<> flattened_reshaped_permuted_tensor_3d = reshaped_permuted_tensor_3d.flatten();

    cout << "flattened_reshaped_permuted_tensor_3d: " << endl;
    flattened_reshaped_permuted_tensor_3d.print();
    cout << endl;


    return 0;
}