# Find all test files
file(GLOB TEST_FILES
    "core/*_test.cpp"
    "modules/layers/*_test.cpp"
    "modules/activations/*_test.cpp"
    "modules/losses/*_test.cpp"
    "utils/*_test.cpp"
    "datasets/*_test.cpp"
    "models/*_test.cpp"
    "metrics/*_test.cpp"
)

# Create test executable
add_executable(unit_tests ${TEST_FILES})

# Link with Google Test and your library
target_link_libraries(unit_tests
    neuralnet
    gtest
    gtest_main
    gmock
)

# Add tests to CTest
add_test(NAME unit_tests COMMAND unit_tests)