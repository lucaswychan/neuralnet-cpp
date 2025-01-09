# neuralnet-cpp

Neural Network in pure C++ without PyTorch and TensorFlow.

Currently supports:

-   [Linear layer](include/modules/layers/linear.hpp)
-   [ReLU activation](include/modules/activations/relu.hpp)
-   [Softmax activation](include/modules/activations/softmax.hpp)
-   [Mean Squared Error loss](include/modules/losses/mse.hpp)
-   [Cross Entropy loss](include/modules/losses/cross_entropy.hpp)

More to come.

**Current achievements: achieving 95% accuracy on MNIST.**

> It would be great if you could [star](https://github.com/lucaswychan/neuralnet-cpp) this project on GitHub. Discussion and suggestions are more welcome!

## Get Started

Make sure you have [CMake](https://cmake.org/) installed.

For Mac OS, run the following commands:

```bash
brew install cmake
```

For Linux, run the following commands:

```bash
sudo apt-get install cmake
```

Get the repository:

```bash
git clone https://github.com/lucaswychan/neuralnet-cpp.git
cd neuralnet-cpp
```

Build the project:

```bash
./build.sh
```

Run the example:

```bash
./main.sh
```

## Tensor from Scratch

I implemented a tensor from scratch as well and integrate it to my neural network implementation. The detailed implementation of `Tensor` can be found in [`include/core/tensor.hpp`](include/core/tensor.hpp).

`Tensor` provides a lot of useful methods such as `add`, `sub`, `mul`, `div`, `matmul`, `transpose`, etc. You can find the detailed documentation in [`include/core/tensor.hpp`](include/core/tensor.hpp).

Note that `Tensor` currently only supports up to 3-dimensional vectors.

### Example usage

```cpp
#include "tensor.hpp"

// default type is double
Tensor<> your_tensor = { { 1.2, 2.3, 3.4 }, { 4.5, 5.6, 6.7 } }; // shape: (2, 3)

// Or you can create a tensor with a specific type
Tensor<int> your_int_tensor = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } } // shape: (3, 3);

// Lots of operations are supported, including element-wise operations, matrix multiplication, etc.
Tensor<> transposed_tensor = your_tensor.transpose(); // shape: (3, 2)

// You can also create a tensor from a vector
vector<vector<double>> your_vec = { { 1.2, 2.3, 3.4 }, { 4.5, 5.6, 6.7 } };
Tensor<> your_tensor_from_vec = Tensor<>(your_vec);
```

## Module API

The module API is defined in [`include/core/module.hpp`](include/core/module.hpp).

To build your custom module, follow the instructions in `include/core/module.hpp`.

### Example usage

```cpp
class MyModule : public nn::Module {
    public:
        virtual Tensor<> forward(const Tensor<>& input) override {
            // Your code here
        }
        virtual Tensor<> backward(const Tensor<>& grad_output) override {
            // Your code here
        }
        virtual void update_params(const float lr) override {
            // Your code here
        }
};
```

## TODO

Please refer to the [TODO list](https://github.com/lucaswychan/neuralnet-cpp/blob/main/TODO.md).

## License
