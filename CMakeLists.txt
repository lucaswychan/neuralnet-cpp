cmake_minimum_required(VERSION 3.14)

# Set the project name
project(neuralnet-cpp)

# Add option for building tests (OFF by default)
option(BUILD_TESTS "Build tests" OFF)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Only enable testing if BUILD_TESTS is ON
if(BUILD_TESTS)
    enable_testing()
endif()

# Add include directories
include_directories(
    include/
    include/core
    include/modules/layers
    include/modules/activations
    include/modules/losses
    include/utils
    include/datasets
    include/models
    include/metrics
)

# Add source files
set(SOURCE_FILES
    src/core/module.cpp
    src/modules/layers/linear.cpp
    src/modules/layers/dropout.cpp
    src/modules/losses/mse.cpp
    src/modules/activations/relu.cpp
    src/modules/activations/softmax.cpp
    src/modules/losses/cross_entropy.cpp
    src/utils/matrix_utils.cpp
    src/utils/tensor_utils.cpp
    src/datasets/mnist.cpp
    src/models/mlp.cpp
    src/metrics/accuracy.cpp
    src/utils/utils.cpp
)

# Create a library from your source files
add_library(neuralnet ${SOURCE_FILES})

# Add the executable for the main example
add_executable(main examples/main.cpp)
target_link_libraries(main neuralnet)

# Only build tests if BUILD_TESTS is ON
if(BUILD_TESTS)
    # Add Google Test
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
    )
    FetchContent_MakeAvailable(googletest)

    # Add test directory
    add_subdirectory(tests)
endif()

# If you have any libraries to link, you can add them here
# target_link_libraries(main <library_name>)