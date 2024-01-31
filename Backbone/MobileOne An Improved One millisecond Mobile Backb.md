# MobileOne: An Improved One millisecond Mobile Backbone

code: [https://github.com/apple/ml-mobileone?tab=readme-ov-file](https://github.com/apple/ml-mobileone?tab=readme-ov-file)

paper: [https://arxiv.org/abs/2206.04040](https://arxiv.org/abs/2206.04040)

Summary:

The goal of this paper is increase the model inference speed on mobile device. To improve the latency cost of architecture while improving the accuracy, authors analysis the key architectural and optimization bottlenecks that affect the latency. 

Based on the analysis, authors propose a new backbone namely MobileOne that is 38x faster than mobileformer with similar performance to it.

This paper is close to what i did in the past. In this paper, authors altered mobilenet v1 with reparameterized block (based on repvgg) along with a more recent model scaling technique and anneal weight decay. What i did in my past work is alter mobilenet with dbbnet block which is very hard to train and also not friendly to quantization. I modified the MobileOne a bit so that it can fit into my work flow. I will test it compare to the dbb variant see which method is better. 

Method:

Correlation analysis between FLOPs and parameter counts wrt. latency:

1. Latency:
    1. latency is moderately correlated with FLOPs. Means on mobile devices, a more complex model (higher FLOPs) usually consist with more inference time, but this relation is not consistent.
    2. Latency is weakly correlated with parameter counts. A large model may consist with lower latency. 
    3. The correlation is much weaker on a desktop GPU. GPU is too powerful so that model complicity has weaker effect on latency(no matter FLOPs or parameter counts)
2.  Key bottlenecks analysis:
    1. Activation functions
    
    ![Untitled](MobileOne%20An%20Improved%20One%20millisecond%20Mobile%20Backb%20a03cb31a19ea475dbcad6cd7310ee614/Untitled.png)
    
    1. Architectural blocks
        
        Two of the key factors that affect latency are memory access cost and degree of parallelism. 
        
        Memory access cost such as multi branch architecture where activations from each branch have to stored to compute the next tensor in the graph.
        
        architecture blocks that forces synchronization like global pooling operations used in Squeeze-Exciteblock also affect overall run-time due to synchronization costs.
        
         
        
        ![Untitled](MobileOne%20An%20Improved%20One%20millisecond%20Mobile%20Backb%20a03cb31a19ea475dbcad6cd7310ee614/Untitled%201.png)
        
3. MobileOne Architecture
    1. MobileOne Block
        1. based on the Mobilenet v1 block design 
            1. 3 x 3 dw conv  + 1 x 1 pw conv
            2. reparameterizble skip connection

![Untitled](MobileOne%20An%20Improved%20One%20millisecond%20Mobile%20Backb%20a03cb31a19ea475dbcad6cd7310ee614/Untitled%202.png)

My adjustment to MobileOne backbone so that it can be used on 112*112 input:

1. The first stage does now downsample the feature map, so that the output size is equivalent to the output of stem block.
2. Replace the global average pooling layer with 1x1 conv for more stability. And add 1d bn after the final layer.