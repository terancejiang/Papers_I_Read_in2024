# A convnet for 2020s

authors reexamine the design spaces and test the limits of what a pure ConvNet can achieve. Also gradually “modernize” a standard ResNet toward the design of a vision Transformer, and discover several key components that contribute to the performance difference along the way. 

The outcome of this exploration is a family of pure ConvNet models dubbed ConvNeXt.

## ConvNeXt

### 1. Training Techniques

Use a training recipe that is close to DeiT’s  and Swin Transformer’s

![Untitled](A%20convnet%20for%202020s%2002d065eb1519408b865702f0f7020655/Untitled.png)

![Untitled](A%20convnet%20for%202020s%2002d065eb1519408b865702f0f7020655/Untitled%201.png)

### 2. Macro Design

2.1 stage ratio

Changing stage compute ratio to (3, 3, 9, 3) / (2, 2, 6, 2)

2.2 Changing stem to “Patchify”

Replace the ResNet-style stem cell with a patchify layer implemented using a 4×4, stride 4 convolutional layer.

### 3. ResNeXt-ify

1. use Depthwise conv

### 4. Inverted Bottleneck and Large Kernel Sizes

![Untitled](A%20convnet%20for%202020s%2002d065eb1519408b865702f0f7020655/Untitled%202.png)

![Untitled](A%20convnet%20for%202020s%2002d065eb1519408b865702f0f7020655/Untitled%203.png)

### 5. Micro Design

5.1 Replacing ReLU with GELU

5.2 Fewer activation functions.

eliminate all GELU layers from the residual block except for one between two 1 × 1 layers

5.3 Fewer normalization layers. 

5.4 Substituting BN with LN

5.5 Separate downsampling layers. 

2×2 conv layers with stride 2 for spatial downsampling