# ParCNetV2: Oversized Kernel with Enhanced Attention

### source : [https://arxiv.org/abs/2211.07157](https://arxiv.org/abs/2211.07157)

### code:

[https://github.com/XuRuihan/ParCNetV2](https://github.com/XuRuihan/ParCNetV2)

To summarize, the main contributions of this paper are as follows:
• oversized convolutions for the effective modeling of long-range feature interactions in CNNs.
• propose two bifurcate gate units (spatial BGU and channel BGU), which are compact and powerful attention modules. 
•  bring oversized convolution to shallow layers of CNNs and unify the local-global convolution design across blocks.

## Method:

![Untitled](ParCNetV2%20Oversized%20Kernel%20with%20Enhanced%20Attention%20514dd782a72744c8a78990cf13ec3dfb/Untitled.png)

### 1. Oversized convolution

oversized depth-wise convolution with a kernel size approximately twice the input feature size (ParC-O-H and ParC-O-W),

Input feature map size of  X = C * H * W

The kernel weight for vertical and horizontal oversized convolution is
k_h = C×(2H−1)×1
k_w = C×1×(2W−1)

![Untitled](ParCNetV2%20Oversized%20Kernel%20with%20Enhanced%20Attention%20514dd782a72744c8a78990cf13ec3dfb/Untitled%201.png)

![Untitled](ParCNetV2%20Oversized%20Kernel%20with%20Enhanced%20Attention%20514dd782a72744c8a78990cf13ec3dfb/Untitled%202.png)

Pros: 

1. it encodes position information by embedding it into each location using zero-padding, eliminating the need for position embeddings.
2. it improves model capacity with limited computational complexity. For instance, the largest oversized kernel in ParCNetV2-Tiny is extended to 111 × 1 and 1 × 111 with input size 224 × 224.

Adaptability to multi-scale input: 

To deal with input images of different resolutions, each convolution kernel will be first zoomed with linear interpolation to C×(2H −1)×1 and C × 1 × (2W − 1).

### 2. Bifurcate Gate Unit

![Untitled](ParCNetV2%20Oversized%20Kernel%20with%20Enhanced%20Attention%20514dd782a72744c8a78990cf13ec3dfb/Untitled%203.png)

Spatial BGU:

![Untitled](ParCNetV2%20Oversized%20Kernel%20with%20Enhanced%20Attention%20514dd782a72744c8a78990cf13ec3dfb/Untitled%204.png)

Channel BGU:

![Untitled](ParCNetV2%20Oversized%20Kernel%20with%20Enhanced%20Attention%20514dd782a72744c8a78990cf13ec3dfb/Untitled%205.png)

### 3 Uniform local-global convolution

![Untitled](ParCNetV2%20Oversized%20Kernel%20with%20Enhanced%20Attention%20514dd782a72744c8a78990cf13ec3dfb/Untitled%206.png)

## Model Configuration

![Untitled](ParCNetV2%20Oversized%20Kernel%20with%20Enhanced%20Attention%20514dd782a72744c8a78990cf13ec3dfb/Untitled%207.png)

![Untitled](ParCNetV2%20Oversized%20Kernel%20with%20Enhanced%20Attention%20514dd782a72744c8a78990cf13ec3dfb/Untitled%208.png)