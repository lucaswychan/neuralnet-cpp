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

# Define the macro for Doctest
macro(add_doctest_test testname)
    add_executable(${testname} ${ARGN})
    # target_link_libraries(${testname} neuralnet doctest::doctest)
    target_link_libraries(${testname} neuralnet)
endmacro()

# Create test executables
foreach(file ${TEST_FILES})
    get_filename_component(testname ${file} NAME_WE)
    add_doctest_test(${testname} ${file})
endforeach()

# Add tests to CTest
foreach(file ${TEST_FILES})
    get_filename_component(testname ${file} NAME_WE)
    add_test(NAME ${testname} COMMAND ${testname})
endforeach()