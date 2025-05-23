cmake_minimum_required(VERSION 3.20)

# Set the project name
project(neuralnet-cpp)

# Add option for building tests (OFF by default)
option(BUILD_TESTS "Build tests" OFF)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Only enable testing if BUILD_TESTS is ON
if(BUILD_TESTS)
    enable_testing()
endif()

# Add include directories
include_directories(
    include/
    include/core
    include/modules/containers
    include/modules/layers
    include/modules/activations
    include/modules/losses
    include/modules/optimizers
    include/utils
    include/datasets
    include/models
    include/metrics
)

# Add source files
set(SOURCE_FILES
    src/core/tensor.cpp
    src/utils/tensor_utils.cpp
    src/core/module.cpp
    src/core/optimizer.cpp
    src/modules/containers/sequential.cpp
    src/modules/layers/linear.cpp
    src/modules/layers/conv2d.cpp
    src/modules/layers/flatten.cpp
    src/utils/conv2d_utils.cpp
    src/modules/layers/dropout.cpp
    src/modules/optimizers/sgd.cpp
    src/modules/optimizers/adam.cpp
    src/modules/losses/mse.cpp
    src/modules/activations/relu.cpp
    src/modules/activations/softmax.cpp
    src/modules/losses/cross_entropy.cpp
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
    # Add Doctest
    # include(FetchContent)
    # FetchContent_Declare(
    #     doctest
    #     GIT_REPOSITORY https://github.com/doctest/doctest.git
    #     GIT_TAG        v2.4.11
    # )
    # FetchContent_MakeAvailable(doctest)

    # Add test directory
    add_subdirectory(tests)

endif()

# If you have any libraries to link, you can add them here
# target_link_libraries(main <library_name>)