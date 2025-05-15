# Tensor Tutorial

`Tensor` provides a lot of useful methods such as `add`, `sub`, `mul`, `div`, `matmul`, `dtype`, etc. You can find the detailed documentation and implementation in [`include/core/tensor.hpp`](../include/core/tensor.hpp).

> [!NOTE]
> Currently, `Tensor` only supports up to 3-dimensional vectors. In the future, `Tensor` will support higher-dimensional vectors.

**Guide :**

-   [Create a tensor](#creteate-a-tensor)
-   [Access tensor metadata](#access-tensor-metadata)
-   [Index tensor](#index-tensor)
-   [Visualize tensor](#visualize-tensor)
-   [Perform arithmetic operations](#perform-arithmetic-operations)
-   [Reshape tensor](#reshape-tensor)
-   [Convert tensor data type](#convert-tensor-data-type)
-   [Filter the unwanted elements](#filter-the-unwanted-elements)
-   [Perform function mapping](#perform-function-mapping)
-   [Max, Min, Argmax, Argmin](#max-min-argmax-argmin)
-   [Flatten tensor](#flatten-tensor)

## Creteate a tensor

You can create your tensor from C++ array, or using `vector` in C++ STL. You can create a tensor with different variable type, even with your custom class.

```cpp
#include "tensor.hpp"

// default type is float
Tensor<> your_tensor = { { 1.2, 2.3, 3.4 }, { 4.5, 5.6, 6.7 } }; // shape: (2, 3)

// Or you can create a tensor with a specific type
Tensor<int> your_int_tensor = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } } // shape: (3, 3);

// Lots of operations are supported, including element-wise operations, matrix multiplication, etc.
Tensor<> transposed_tensor = your_tensor.transpose(); // shape: (3, 2)

// You can also create a tensor from a vector
vector<vector<float>> your_vec = { { 1.2, 2.3, 3.4 }, { 4.5, 5.6, 6.7 } };
Tensor<> your_tensor_from_vec = Tensor<>(your_vec);
```

## Access tensor metadata

Several function are provided to access the tensor's shape, dimensions, and size.

```cpp
Tensor<> tensor = { { 1.2, 2.3, 3.4 }, { 4.5, 5.6, 6.7 } };

// it stores the shapes of each dimension.
vector<size_t> shapes = tensor.shapes()
// {2, 3}

// it stores the number of dimensions of tensor.
size_t num_dimensions = tensor.ndim()
// 2

// it stores the total number of elements in tensor.
size_t size = tensor.size()
// 6
```

## Index tensor

Without calling any function or creating a vector, you can directly index your tensor with just `[]`.

```cpp
Tensor<int> A = { { 1, 2, 3 },
                  { 4, 5, 6 } }; // 2 x 3

int first_element = A[0, 0];
// 1

// Negative indexing is allowed as well
int last_element = A[-1, -1];
// 6

for (int i = 0, i < A.shapes()[0]; ++i) {
    for (int j = 0; j < A.shapes()[1]; ++j) {
        cout << A[i, j] << " ";
    }
}
cout << endl;
```

## Visualize tensor

Instead of using `cout` all the time, `print` is provided for convenience.

```cpp
Tensor<int> A = { { 1, 2, 3 },
                  { 4, 5, 6 } }; // 2 x 3

A.print()
/*
[[1, 2, 3],
[4, 5, 6]]
*/
```

## Perform arithmetic operations

You can also perform typical tensor's arithmetic operations including `add`, `sub`, `mul`, `div`, `matmul`, `transpose`, etc.

```cpp
Tensor<int> A = { { 1, 2, 3 },
                  { 4, 5, 6 } }; // 2 x 3

Tensor<int> B = { { 1, 2, 3 },
                  { 4, 5, 6 },
                  { 7, 8, 9 } };  // 3 x 3

Tensor<int> A_2 = A * A;
/*
{ { 1, 4, 9 },
  { 16, 25, 36 } }
*/

Tensor<int> B_plus_B = B + B;
/*
{ { 2, 4, 6 },
  { 8, 10, 12 },
  { 14, 16, 18 } }
*/

Tensor<int> B_x2 = B * 2
/*
{ { 2, 4, 6 },
  { 8, 10, 12 },
  { 14, 16, 18 } }
*/

Tensor<int> A_matmul_B = A.matmul(B);
/*
{ { 30, 36, 42 }
  { 66, 81, 96 } }
*/

Tensor<int> A_T = A.transpose();
/*
{ { 1, 4 },
  { 2, 5 },
  { 3, 6 } }
*/
```

## Reshape tensor

You can reshape your tensor. Note that your new shapes must have the same number of elements.

```cpp
Tensor<int> A = { { 1, 2, 3 },
                  { 4, 5, 6 } }; // 2 x 3

vector<size_t> new_shapes = {6};
A.reshape(new_shapes);  // or A.reshape({6})

size_t new_ndim = A.ndim()
// 1

vector<size_t> other_shapes = {3, 4};
A.reshape(other_shapes); // Error !!!!!
```

## Convert tensor data type

If you don't like the current tensor's data type, feel free to convert it to other data type using `dtype`. Since it is a template function, you should specify the desired type in the template argument instead of the funcion argument.

```cpp
Tensor<int> A = { { 1, 2, 3 },
                  { 4, 5, 6 } }; // 2 x 3

Tensor<float> A_float = A.dtype<float>();

Tensor<> A_float = A.dtype<float>(); // since the default type of tensor is float
```

## Filter the unwanted elements

Sometimes some of the elements in the tensor should be filtered (such as ReLU operation). Simply use `filter` to filter the unwanted elements. It takes a boolean function as argument to test each element of the tensor. It should return true if the element passes the test. All elements that fail the test are set to 0

```cpp
Tensor<int> A = { { -1, 2, 3 },
                  { 4, -5, 6 } }; // 2 x 3

Tensor<int> A_filtered = A.filter([](int x) { return x > 0; });
/*
{ { 0, 2, 3 },
  { 4, 0, 6 } }
*/
```

## Perform function mapping

Function mapping also can be applied to the tensor, simply by using `map`. It takes a function as argument to perform element-wise transformation to the tensor.

```cpp
#include <math.h>

Tensor<> A = { { 1, 2, 3 },
               { 4, 5, 6 } }; // 2 x 3

Tensor<> A_mapped = A.map([](float x) { return exp(x); });
/*
{ { 2.71828, 7.38906, 20.0855 },
  { 54.5982, 148.413, 403.429 } }
*/
```

## Max, Min, Argmax, Argmin

Row-wise max, min, argmax, argmin operations are also provided. Currently only support 1-D and 2-D tensor.

```cpp
Tensor<int> B = { { 10, 2, 3 },
                  { 4, 5, 60 },
                  { 7, 80, 90 } };  // 3 x 3

Tensor<int> B_max = B.max();
// { 10, 60, 90 }

Tensor<size_t> B_argmax = B.argmax();
// { 0, 2, 2 }

Tensor<int> B_max = B.min();
// { 2, 4, 7 }

Tensor<size_t> B_argmax = B.argmin();
// { 1, 0, 0 }

Tensor<int> tensor_1d = { 1, 2, 30, 4, 5 };

Tensor<int> tensor_1d_max = tensor_1d.max();
// { 30 }

Tensor<size_t> tensor_1d_argmax = tensor_1d.argmax();
// { 2 }

Tensor<int> tensor_1d_max = tensor_1d.min();
// { 1 }

Tensor<size_t> tensor_1d_argmax = tensor_1d.argmin();
// { 0 }
```

## Flatten tensor

You can flatten your tensor using `flatten` function. It flattens the dimensions of the tensor from start_dim to end_dim into a single dimension. Default of start_dim and end_dim is 0 and -1 respectively.

```cpp
Tensor<int> A = { { 1, 2, 3 },
                  { 4, 5, 6 } }; // 2 x 3

Tensor<int> A_flatten = A.flatten();
// [ 1, 2, 3, 4, 5, 6 ]

Tensor<> B_3d = { { { -1, -2, -3 },
                    {-4, -5, -6 } }, 
                  { { 1, 2, 3 },
                    { 4, 5, 6 } } }; // 2 x 2 x 3

Tensor<> B_flatten_12 = B_3d.flatten(0, 1) // flatten the first and second dimension
/* 
[
  [-1, -2, -3],
  [-4, -5, -6],
  [1, 2, 3],
  [4, 5, 6]
]
*/

Tensor<> B_flatten_23 = B_3d.flatten(1, 2) // flatten the second and the third (last) dimension
/*
[
  [-1, -2, -3, -4, -5, -6],
  [1, 2, 3, 4, 5, 6]
]
*/
```
