# neuralnet-cpp

Neural Network in C++ from scratch. Only pure C++ code. Currenltly supports:

-   Linear layer
-   Sigmoid activation
-   ReLU activation
-   Mean Squared Error loss
-   Cross Entropy loss

More to come.

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

## Usage

The module API is defined in `include/core/module.hpp`.

To build your custom module, follow the instructions in `include/core/module.hpp`.

```cpp
class MyModule : public nn::Module {
    public:
        virtual vector<vector<float>> forward(const vector<vector<float>>& input) override {
            // Your code here
        }
        virtual vector<vector<float>> backward(const vector<vector<float>>& grad_output) override {
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
