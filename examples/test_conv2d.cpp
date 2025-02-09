#include "conv2d.hpp"
using namespace nn;

int main()
{
    Conv2d conv2d(1, 1, 3);
    cout << "Conv2d layer initialized with in_channels = 1 and out_channels = 1" << endl;
    return 0;
}