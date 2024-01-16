# Diverse Branch Block: Building a Convolution as an Inception-like Unit

source:[https://arxiv.org/pdf/2103.13425.pdf](https://arxiv.org/pdf/2103.13425.pdf)

code: [https://github.com/](https://github.com/)DingXiaoH/DiverseBranchBlock.

Similar to the approach in another paper, "REPVGG," a complex inception block-like structure can be transformed into a regular convolutional layer for deployment. Thus, during the training phase, the model can capture more details without reducing the inference speed during deployment.

## Method:

![Untitled](Diverse%20Branch%20Block%20Building%20a%20Convolution%20as%20an%20%20d5c22935fcf24399bf3c56b5e0b8b646/Untitled.png)

### 1. The Linearity of Convolution

when two convolutions have the same configurations (e.g., number of channels, kernel size, stride, padding, etc.), so that they share the same sliding window correspondence X. Those two convolutions can be added together as follows. 

![Untitled](Diverse%20Branch%20Block%20Building%20a%20Convolution%20as%20an%20%20d5c22935fcf24399bf3c56b5e0b8b646/Untitled%201.png)

### 2. Transform I: a conv for conv-BN

Same with regvpp block, the output for conv-BN in the form of conv as follow; 

![Untitled](Diverse%20Branch%20Block%20Building%20a%20Convolution%20as%20an%20%20d5c22935fcf24399bf3c56b5e0b8b646/Untitled%202.png)

![Untitled](Diverse%20Branch%20Block%20Building%20a%20Convolution%20as%20an%20%20d5c22935fcf24399bf3c56b5e0b8b646/Untitled%203.png)

```python
# fuse conv-bn to conv
def transI_fusebn(kernel, bn):
		"""
		kernel = self.dbb_origin.conv.weight, 
		bn = self.dbb_origin.bn
		"""
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std
```

### 3. Transform II: a conv for branch addition

The additivity ensures that if the outputs of two or more conv layers with the same configurations are added up, we can merge them into a single conv.

![Untitled](Diverse%20Branch%20Block%20Building%20a%20Convolution%20as%20an%20%20d5c22935fcf24399bf3c56b5e0b8b646/Untitled%204.png)

![Untitled](Diverse%20Branch%20Block%20Building%20a%20Convolution%20as%20an%20%20d5c22935fcf24399bf3c56b5e0b8b646/Untitled%205.png)

```python
# fuse mutiple same configuration 3x3 conv to a single conv for branch addition. 
def transII_addbranch(kernels, biases):
		"""
		kernel = (k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged)
		bn = (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged)
		"""
    return sum(kernels), sum(biases)
```

### 4. Transform III: a conv for sequential convolutions

merge a sequence of 1 × 1 conv - BN - K × K conv - BN into one single K×K conv like :

![Untitled](Diverse%20Branch%20Block%20Building%20a%20Convolution%20as%20an%20%20d5c22935fcf24399bf3c56b5e0b8b646/Untitled%206.png)

the goal is convert following equation to above. 1 × 1 conv - BN - K × K conv - BN can be writen as 

![Untitled](Diverse%20Branch%20Block%20Building%20a%20Convolution%20as%20an%20%20d5c22935fcf24399bf3c56b5e0b8b646/Untitled%207.png)

![Untitled](Diverse%20Branch%20Block%20Building%20a%20Convolution%20as%20an%20%20d5c22935fcf24399bf3c56b5e0b8b646/Untitled%208.png)

![Untitled](Diverse%20Branch%20Block%20Building%20a%20Convolution%20as%20an%20%20d5c22935fcf24399bf3c56b5e0b8b646/Untitled%209.png)

As I*F(1) is 1×1 conv, which performs only channel wise linear combination but no spatial aggregation, we can merge it into the K × K conv by linearly recombining the parameters in K × K kernel. It is easy to verify that such a transformation can be accomplished by transpose conv. Thus the first part of the equation equal to:

![Untitled](Diverse%20Branch%20Block%20Building%20a%20Convolution%20as%20an%20%20d5c22935fcf24399bf3c56b5e0b8b646/Untitled%2010.png)

![Untitled](Diverse%20Branch%20Block%20Building%20a%20Convolution%20as%20an%20%20d5c22935fcf24399bf3c56b5e0b8b646/Untitled%2011.png)

```python
def transIII_1x1_kxk(k1, b1, k2, b2, groups):
		"""
		k1, b1=  1 × 1 conv - BN 
		k2, b2=  K × K conv - BN
		"""
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))      #
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))

    return k, b_hat + b2
```

### 5. Transform III: a conv for depth concatenation

equal to number of groups transform v concatenate together

![Untitled](Diverse%20Branch%20Block%20Building%20a%20Convolution%20as%20an%20%20d5c22935fcf24399bf3c56b5e0b8b646/Untitled%2012.png)

![Untitled](Diverse%20Branch%20Block%20Building%20a%20Convolution%20as%20an%20%20d5c22935fcf24399bf3c56b5e0b8b646/Untitled%2013.png)

```python
def transIII_1x1_kxk(k1, b1, k2, b2, groups):
		"""
		k1, b1=  1 × 1 conv - BN 
		k2, b2=  K × K conv - BN
		"""
    k_slices = []
    b_slices = []
    k1_T = k1.permute(1, 0, 2, 3)
    k1_group_width = k1.size(0) // groups
    k2_group_width = k2.size(0) // groups
    for g in range(groups):
        k1_T_slice = k1_T[:, g*k1_group_width:(g+1)*k1_group_width, :, :]
        k2_slice = k2[g*k2_group_width:(g+1)*k2_group_width, :, :, :]
        k_slices.append(F.conv2d(k2_slice, k1_T_slice))
        b_slices.append((k2_slice * b1[g*k1_group_width:(g+1)*k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
    k, b_hat = transIV_depthconcat(k_slices, b_slices)
    return k, b_hat + b2
def transIV_depthconcat(kernels, biases):
    return torch.cat(kernels, dim=0), torch.cat(biases)
```

### 6. Transform V: a conv for average pooling

An average pooling with kernel size K and stride s applied to C channels is equivalent to a conv with the same K and s.

In average pooling, each element in the pooling window contributes equally to the output. To mimic this in a convolutional layer, you should set all the weights in the *K* × *K* filter to 1/*K^2* This way, the filter sums up all the inputs in its receptive field and then divides that sum by the total number of elements, effectively computing their average.

```python
def transV_avg(channels, kernel_size, groups):
    input_dim = channels // groups
    k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
    k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
    return k
```

### 7. Transform VI: a conv for multi-scale convolutions

![Untitled](Diverse%20Branch%20Block%20Building%20a%20Convolution%20as%20an%20%20d5c22935fcf24399bf3c56b5e0b8b646/Untitled%2014.png)

```python
def transVI_multiscale(kernel, target_kernel_size):
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    return F.pad(kernel, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])
```

**Results:**

![Untitled](Diverse%20Branch%20Block%20Building%20a%20Convolution%20as%20an%20%20d5c22935fcf24399bf3c56b5e0b8b646/Untitled%2015.png)