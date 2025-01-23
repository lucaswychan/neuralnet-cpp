#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "module.hpp"

namespace nn {

class MockModule : public Module {
public:
    // Since doctest doesn't have built-in mocking, we'll implement a simple mock
    Tensor<> expected_forward_output;
    Tensor<> expected_backward_output;
    bool update_params_called = false;
    
    Tensor<> forward(const Tensor<>& input) override {
        return expected_forward_output;
    }
    
    Tensor<> backward(const Tensor<>& grad_output) override {
        return expected_backward_output;
    }
    
    void update_params(float learning_rate) override {
        update_params_called = true;
    }
};

TEST_CASE("ModuleTest - Constructor and Destructor") {
    MockModule module;
    // No explicit assertions needed, just verify no crashes
}

TEST_CASE("ModuleTest - Forward Pass") {
    MockModule module;
    Tensor<> input = {1, 2, 3, 4};
    Tensor<> expected_output = {1, 2, 3, 4};
    
    module.expected_forward_output = expected_output;
    
    Tensor<> result = module.forward(input);
    CHECK(result == expected_output);
}

TEST_CASE("ModuleTest - Backward Pass") {
    MockModule module;
    Tensor<> grad_output = {1, 2, 3, 4};
    Tensor<> expected_grad_input = {1, 2, 3, 4};
    
    module.expected_backward_output = expected_grad_input;
    
    Tensor<> result = module.backward(grad_output);
    CHECK(result == expected_grad_input);
}

TEST_CASE("ModuleTest - Update Parameters") {
    MockModule module;
    
    module.update_params(0.001f);
    CHECK(module.update_params_called);
}

TEST_CASE("ModuleTest - Operator Overload") {
    MockModule module;
    Tensor<> input = {1, 2, 3, 4};
    Tensor<> expected_output = {1, 2, 3, 4};
    
    module.expected_forward_output = expected_output;
    
    Tensor<> result = module(input);
    CHECK(result == expected_output);
}

TEST_CASE("ModuleTest - Train Mode") {
    MockModule module;

    module.train(false);
    CHECK_FALSE(module.is_training());
    module.train();
    CHECK(module.is_training());
    module.eval();
    CHECK_FALSE(module.is_training());
}

} // namespace nn