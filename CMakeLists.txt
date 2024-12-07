cmake_minimum_required(VERSION 3.10)

# Set the project name
project(neuralnet-cpp)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add include directories
include_directories(
    include/
    include/tensors
    include/core
    include/modules/layers
    include/modules/activations
    include/modules/losses
    include/utils
)

# Add the executable for the main example
add_executable(main examples/main.cpp)

# Add source files
set(SOURCE_FILES
    src/tensors/one_dim_tensor.cpp
    src/core/module.cpp
    src/modules/layers/linear.cpp
    # src/modules/activations/relu.cpp
    # src/modules/activations/sigmoid.cpp
    # src/modules/activations/tanh.cpp
    # src/utils/matrix.cpp
    src/utils/matrix_utils.cpp
    src/modules/losses/mse.cpp
)

# Link the source files to the executable
target_sources(main PRIVATE ${SOURCE_FILES})

# If you have any libraries to link, you can add them here
# target_link_libraries(main <library_name>)