#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "module.hpp"
using namespace testing;

namespace nn {

class MockModule : public Module {
public:
    MOCK_METHOD(Tensor<>, forward, (const Tensor<>&), (override));
    MOCK_METHOD(Tensor<>, backward, (const Tensor<>&), (override));
    MOCK_METHOD(void, update_params, (float), (override));
};

TEST(ModuleTest, ConstructorDestructor) {
    MockModule module;
    // No explicit assertions needed, just verify no crashes
}

TEST(ModuleTest, ForwardPass) {
    MockModule module;
    Tensor<> input = {1, 2, 3, 4};
    Tensor<> expected_output = {1, 2, 3, 4};

    EXPECT_CALL(module, forward(_))
        .WillOnce(Return(expected_output));

    Tensor<> result = module.forward(input);
    EXPECT_EQ(result, expected_output);
}

TEST(ModuleTest, BackwardPass) {
    MockModule module;
    Tensor<> grad_output = {1, 2, 3, 4};
    Tensor<> expected_grad_input = {1, 2, 3, 4};

    EXPECT_CALL(module, backward(_))
        .WillOnce(Return(expected_grad_input));

    Tensor<> result = module.backward(grad_output);
    EXPECT_EQ(result, expected_grad_input);
}

TEST(ModuleTest, UpdateParams) {
    MockModule module;

    EXPECT_CALL(module, update_params(_))
        .Times(1);

    module.update_params(0.001f);
}

TEST(ModuleTest, OperatorOverload) {
    MockModule module;
    Tensor<> input = {1, 2, 3, 4};;
    Tensor<> expected_output = {1, 2, 3, 4};

    EXPECT_CALL(module, forward(_))
        .WillOnce(Return(expected_output));

    Tensor<> result = module(input);
    EXPECT_EQ(result, expected_output);
}

TEST(ModuleTest, TrainMode) {
    MockModule module;

    module.train(false);
    EXPECT_FALSE(module.is_training());
    module.train();
    EXPECT_TRUE(module.is_training());
    module.eval();
    EXPECT_FALSE(module.is_training());
}

} // namespace nn