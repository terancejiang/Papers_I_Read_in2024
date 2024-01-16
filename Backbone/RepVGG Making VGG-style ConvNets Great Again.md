# RepVGG: Making VGG-style ConvNets Great Again

### **Problem Addressed**

The goal is to design a convolutional neural network architecture that is both simple and powerful, meeting the following conditions:

1. The inference model should be straightforward, consisting solely of 3x3 convolutions.
2. The training model should have a multi-branch topology.

TL;DR: The authors have developed a method to transform a multi-branch topology into a simple 3x3 convolution, a technique they refer to as structural re-parameterization.

A multi-branch contains 3x3conv-bn, 1x1conv-bn and a identity shortcut. Firstly, convert 3x3conv-bn to 3x3; 1x1conv-bn to 3x3 and identity to 3x3; then concatenate three 3x3 to one 3x3 conv. 

Using this technique, they upgraded the VGG network, naming it RepVGG. On ImageNet, RepVGG achieved over 80% top-1 accuracy, marking the first instance of such a performance by a plain model, to the best of the authors' knowledge.

## Insights:

1. Winograd Algorithm: This classic algorithm is used for accelerating 3x3 convolutions (applicable only when the stride is 1). It has received robust support and is enabled by default in libraries like cuDNN and MKL.
2. Conversion: The method involves converting a multi-branch block into a single 3x3 convolution based on the concept of The Linearity of Convolution(author explained more clearly in his Diverse Branch Block paper).
3. Model Flexibility: The training model can be complex, encompassing more parameters, but at the time of inference or deployment, it can be converted into a lightweight version, making it more suitable for edge devices.

## Method:

![Untitled](RepVGG%20Making%20VGG-style%20ConvNets%20Great%20Again%205c018e667bc442349cfdcf390050d2fc/Untitled.png)

### Convert a conv for conv-BN

![Untitled](RepVGG%20Making%20VGG-style%20ConvNets%20Great%20Again%205c018e667bc442349cfdcf390050d2fc/Untitled%201.png)

Let j be the channel index, µj and σj be the accumulated channel-wise mean and standard deviation, γj and βj be the learned scaling factor and bias term, respectively, the output channel j becomes

![Untitled](RepVGG%20Making%20VGG-style%20ConvNets%20Great%20Again%205c018e667bc442349cfdcf390050d2fc/Untitled%202.png)

In practice, we simply build a single conv with kernel F and bias b , assign the values transformed from the parameters of the original conv-BN sequence, then save the model for inference. For every output channel j 

![Untitled](RepVGG%20Making%20VGG-style%20ConvNets%20Great%20Again%205c018e667bc442349cfdcf390050d2fc/Untitled%203.png)

**More details:**

**M(1)**: input; **M(2)**: output; **W(1)**: weight of 1 x 1; **W(3)**: weight of 3 x 3 

**C1**: input channels; **C2**: out-put channels

**µ (3) ,σ (3) , γ (3), β (3)** :accumulated mean, standard deviation and learned scaling factor and bias of the BN layer following 3 × 3 conv

**µ (1) ,σ (1) , γ (1) , β (1)** :accumulated mean, standard deviation and learned scaling factor and bias of the BN layer following 1 × 1 conv

![Untitled](RepVGG%20Making%20VGG-style%20ConvNets%20Great%20Again%205c018e667bc442349cfdcf390050d2fc/Untitled%204.png)