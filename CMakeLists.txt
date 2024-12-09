cmake_minimum_required(VERSION 3.10)

# Set the project name
project(neuralnet-cpp)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add include directories
include_directories(
    include/
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
    src/core/module.cpp
    src/modules/layers/linear.cpp
    src/modules/losses/mse.cpp
    src/modules/activations/relu.cpp
    src/modules/activations/softmax.cpp
    src/modules/losses/cross_entropy.cpp
    src/utils/matrix_utils.cpp
)

# Link the source files to the executable
target_sources(main PRIVATE ${SOURCE_FILES})

# If you have any libraries to link, you can add them here
# target_link_libraries(main <library_name>)