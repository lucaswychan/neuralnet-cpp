cd build-tests
cmake -DBUILD_TESTS=ON -Wno-dev ..
make
./tests/unit_tests