# neuralnet-cpp

![C++](https://img.shields.io/badge/C%2B%2B-23-blue.svg)
![C++ Unit Tests](https://github.com/lucaswychan/neuralnet-cpp/actions/workflows/cpp_test.yaml/badge.svg)
[![GitHub license badge](https://img.shields.io/github/license/lucaswychan/neural-stock-prophet?color=blue)](https://opensource.org/licenses/MIT)

This is a PyTorch-like neural network framework in pure C++ from scratch, using only C++ STL.

Currently supports:

-   [Linear layer](include/modules/layers/linear.hpp)
-   [ReLU activation](include/modules/activations/relu.hpp)
-   [Softmax activation](include/modules/activations/softmax.hpp)
-   [Mean Squared Error loss](include/modules/losses/mse.hpp)
-   [Cross Entropy loss](include/modules/losses/cross_entropy.hpp)

More to come.

**Current achievements: achieving 95% accuracy on MNIST.**

> [!NOTE]
> To support and foster the growth of the project, you could ⭐ [star](https://github.com/lucaswychan/neuralnet-cpp) this project on GitHub. Discussion and suggestions are more welcome!

## Get Started

This project requires C++23, GCC >= 13.3, and CMake >= 3.20 to compile. Please make sure you have [GCC](https://gcc.gnu.org) and [CMake](https://cmake.org/) installed.

For **Mac OS**, run the following commands:

```bash
brew install cmake
brew install gcc
```

For **Linux**, run the following commands:

```bash
sudo apt update
sudo apt install cmake
sudo apt install build-essential
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

For more details about tensor, please refer to [tensor tutorial](docs/tensor.md).

## Module API

The module API is defined in [`include/core/module.hpp`](include/core/module.hpp).

To build your custom module, follow the instructions in `include/core/module.hpp`.

### Example usage

```cpp
#include <module.hpp>
using namespace nn;

// Your code here
class MyModule : public Module {
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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
