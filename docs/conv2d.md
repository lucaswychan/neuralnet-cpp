# Conv2D Forward and Backward Propagation

I implemented Conv2D forward and backward propagation from scratch. It takes me really a lot of time, and without [Rukayat Sadiq tutorial](https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf), it must take much more time. I am beyond grateful to this tutorial and I would like to give the credit here. Below I would explain my implementation details and logic for both propagations.

## Conv2D initialization

We first initialize our `Conv2D` module with the following input:

-   `in_channel` : the input data channel (e.g. 3 for RGB image / the output channel of the previous Conv2D layer)
-   `out_channel` : the output channel (can be set whatever you want)
-   `kernel_size` : the size of the convolving kernel, which is also the weight of this layer
-   `stride` : the step when convolving each time
-   `padding` : number of elements padded to each side of the input data
-   `dilation` : spacing between each kernel elements
-   `padding_mode` : the mode of padding. It controls the value to be padded to the input data
-   `bias` : to determine if using bias or not

For `kernel_size`, `stride`, `padding`, and `dilation`, their values can be an `int` or `tuple`, but the tuple size have to be length 2. The first value of the tuple controls the `height` (row) aspect, and the second value of the tuple controls the `width` (column) aspect.

After initialization, we will get the following weights:

-   `weight` : the kernel to convolve the input data and get the output. It is in the size of `(out_channel, in_channel, kernel_size[0], kernel_size[1])`
-   `bias` : The bias if `bias` is True. It is in the size of `(out_channel)`

## Forward Propagation
