# Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks

Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks

Code:  [https://github](https://github/).com/JierunChen/FasterNet.

Paper: [https://arxiv.org/abs/2303.03667](https://arxiv.org/abs/2303.03667)

### Insights:

1. Memory access significantly impacts inference latency.
2. Fasternet is more effective in edge devices that are less sensitive to flash storage capacity than to processing power.
3. The ResNet50 feature map exhibits high redundancies across various channels, which might also be applicable to regular convolutional layers.
4. The calculation of memory access and floating-point operations per second (FLOPs)

**TL;DR:** The authors identify a common misconception in neural network optimization: reducing the number of FLOPs (Floating Point Operations) doesn't always translate to lower latency, primarily due to frequent memory access, as seen in operations like depth-wise convolution. To address this, they introduce a new method called Partial Convolution (PConv), designed to extract spatial features with reduced memory access.

Utilizing PConv and the standard Point-wise Convolution (PWConv), the researchers develop FasterNet. This network comprises four hierarchical stages, each initiated by either an embedding layer (a regular Convolution with a 4x4 filter and stride of 4) for spatial downsampling or a merging layer (a regular Convolution with a 2x2 filter and stride of 2) for expanding the number of channels. Each stage includes a series of FasterNet blocks.

## Method:

The paper presents a comparison between Depth-Wise Convolution (DWConv) and regular Convolution. In this context, 'k' represents the filter size, and 'c' denotes the number of channels. Typically, DWConv is followed by PWConv, where the number of channels is increased from 'c' to 'c**`' (with 'c`**' being greater than 'c').

|  | DWConv  | regular Conv |  |
| --- | --- | --- | --- |
| FLOPs  | h * w * k^2 * c | h * w * k^2 * c^2 |  |
| Memory Access | h * w* 2c + k^2 *c`  | h * w* 2c + k^2 *c^2 |  |

DWConv in priactice, the number of memory access is higher than regular Conv

### PConv:

The research uncovers a significant similarity in feature maps across various channels within neural networks. Addressing this, the authors introduce Partial Convolution (PConv), aiming to simultaneously decrease computational redundancy and memory access. PConv operates by applying a regular convolution to only a subset of the input channels (either the first or last Cp consecutive channels, with a default ratio of 0.25) for spatial feature extraction, while leaving the rest of the channels unchanged.

![Untitled](Run,%20Don%E2%80%99t%20Walk%20Chasing%20Higher%20FLOPS%20for%20Faster%20Ne%20d6e797d4cea94ea299bbdd3dfe52a7fb/Untitled.png)

|  | DWConv | regular Conv | PConv (ratio=0.25) |
| --- | --- | --- | --- |
| FLOPs | h * w * k^2 * c | h * w * k^2 * c^2 | h * w * k^2 * cp^2 (1/16 of regular Conv) |
| Memory Access | h * w* 2c + k^2 *c` | h * w* 2c + k^2 *c^2 | h * w* 2cp + k^2 *cp^2 (1/4 of regular Conv) |

If the (c - Cp) channels are removed, PConv effectively becomes a regular convolution with fewer channels. This innovative approach aims to optimize neural network performance by focusing on selective channel processing.

### PConv followed by PWConv

Similar with DWConv followed by PWConv, the goal of PWConv is forming a T-shaped Conv which foucuses more on the center position compared to a regular Conv uniformly processing a path, as shown blow. 

![Untitled](Run,%20Don%E2%80%99t%20Walk%20Chasing%20Higher%20FLOPS%20for%20Faster%20Ne%20d6e797d4cea94ea299bbdd3dfe52a7fb/Untitled%201.png)

|  | DWConv | regular Conv | PConv (ratio=0.25) | Tshape Conv |
| --- | --- | --- | --- | --- |
| FLOPs | h * w * k^2 * c | h * w * k^2 * c^2 | h * w * k^2 * cp^2 (1/16 of regular Conv) | h * w (k^2cp*c + c * ()c-cp)) |
| Memory Access | h * w* 2c + k^2 *c` | h * w* 2c + k^2 *c^2 | h * w* 2cp + k^2 *cp^2 (1/4 of regular Conv) | h*w*(k^2*cp^2+c^2) |

## FasterNet

![Untitled](Run,%20Don%E2%80%99t%20Walk%20Chasing%20Higher%20FLOPS%20for%20Faster%20Ne%20d6e797d4cea94ea299bbdd3dfe52a7fb/Untitled%202.png)

## Experiment results

FLOPS: floating-point operations per second

FLOPs: number of floating-point operations

![Untitled](Run,%20Don%E2%80%99t%20Walk%20Chasing%20Higher%20FLOPS%20for%20Faster%20Ne%20d6e797d4cea94ea299bbdd3dfe52a7fb/Untitled%203.png)

![Untitled](Run,%20Don%E2%80%99t%20Walk%20Chasing%20Higher%20FLOPS%20for%20Faster%20Ne%20d6e797d4cea94ea299bbdd3dfe52a7fb/Untitled%204.png)

![Untitled](Run,%20Don%E2%80%99t%20Walk%20Chasing%20Higher%20FLOPS%20for%20Faster%20Ne%20d6e797d4cea94ea299bbdd3dfe52a7fb/Untitled%205.png)