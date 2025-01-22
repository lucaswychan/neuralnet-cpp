cd build-tests
cmake -DBUILD_TESTS=ON ..
make
./tests/unit_tests