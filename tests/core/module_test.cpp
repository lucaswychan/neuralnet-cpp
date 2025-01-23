#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "module.hpp"
using namespace nn;

class MockModule : public Module {
public:
    MOCK_METHOD(Tensor<>, forward, (const Tensor<>&), (override));
    MOCK_METHOD(Tensor<>, backward, (const Tensor<>&), (override));
    MOCK_METHOD(void, update_params, (const float), ());
};

TEST(ModuleTest, InheritanceBehavior) {
    MockModule module;
    
    // Test training mode functionality
    module.train();
    EXPECT_TRUE(module.is_training());
    module.eval();
    EXPECT_FALSE(module.is_training());
}



