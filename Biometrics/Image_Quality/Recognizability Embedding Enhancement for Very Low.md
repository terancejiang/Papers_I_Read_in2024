# Recognizability Embedding Enhancement for Very Low Resolution Face Recognition and Quality Estimation

source: [https://arxiv.org/pdf/2304.10066.pdf](https://arxiv.org/pdf/2304.10066.pdf)

## Problem addressed:

There are three kind of face images in the world, 1. easy to recognize; 2 hard to recognize and 3 unrecognizable. The goal of the paper is improve the recognizability of hard-to-recognize instances by pushing them away from the unrecognizable image center.

**tl;dr:** 

There are no official implementations.

To conquer this issue, author made following changes to the main loss function

![Untitled](Recognizability%20Embedding%20Enhancement%20for%20Very%20Low%20f23012c2cbea43909d18bd70e44b7f2f/Untitled.png)

where L_cls is Arcface loss

L_L1 

![Untitled](Recognizability%20Embedding%20Enhancement%20for%20Very%20Low%20f23012c2cbea43909d18bd70e44b7f2f/Untitled%201.png)

L_ID:

![Untitled](Recognizability%20Embedding%20Enhancement%20for%20Very%20Low%20f23012c2cbea43909d18bd70e44b7f2f/Untitled%202.png)

L_MSE:

![Untitled](Recognizability%20Embedding%20Enhancement%20for%20Very%20Low%20f23012c2cbea43909d18bd70e44b7f2f/Untitled%203.png)

α, β, and γ are the weighting factors for each loss term; more details in blow. 

![Untitled](Recognizability%20Embedding%20Enhancement%20for%20Very%20Low%20f23012c2cbea43909d18bd70e44b7f2f/Untitled%204.png)

## Method:

![Untitled](Recognizability%20Embedding%20Enhancement%20for%20Very%20Low%20f23012c2cbea43909d18bd70e44b7f2f/Untitled%205.png)

### 1. Recognizability Index (RI) $\xi{i}$  Formulation

where the $d_i^P$ is intra-class proximity and $d_i^N$ is inter-class proximity

where $θ_{yi}$ is the positive angle between $\hat{v_{bi}}$(embedding) and $\hat{w_{yi}}$(positive prototype)

where $θ_{j}$ is the negative anglebetween $\hat{v_{bi}}$(embedding) and $\hat{w_{j}}$(nearest negative prototype)

![Untitled](Recognizability%20Embedding%20Enhancement%20for%20Very%20Low%20f23012c2cbea43909d18bd70e44b7f2f/Untitled%206.png)

- **Positive Prototype** is usually calculated as the mean or centroid of the feature vectors (embeddings) of all face images in a particular class (e.g., images of the same individual). This prototype represents the 'average' or most common features of faces in that class.
- **Negative Prototype** could be derived in several ways, depending on the specific approach of the paper. It might be the mean of the feature vectors of face images from different classes, representing common features across various faces that are not useful for distinguishing any particular individual. Alternatively, it could be a representation of features that are commonly associated with unrecognizable or low-quality images.

Given the UIs (unrecognizable images) cluster center, (i.e., the average across all normalized UIs embeddings) ($\bar{v_{UI}}$) , $d_i^{UI}$ is the proximity between $\hat{v_{bi}}$(embedding) and $\bar{v_{UI}}$ is defined as:

![Untitled](Recognizability%20Embedding%20Enhancement%20for%20Very%20Low%20f23012c2cbea43909d18bd70e44b7f2f/Untitled%207.png)

where $θ_{UI_i}$ is the angle between  $\hat{v_{bi}}$(embedding)  and $\bar{v_{UI}}$

|  | distance $θ_{UI_i}$ | $cos(θ_{UI_i})$ | $d_i^{UI}$ |
| --- | --- | --- | --- |
| Easy | large | small | large |
| Hard | small | large | small |

Recognizability Index (RI), ξi： 

![Untitled](Recognizability%20Embedding%20Enhancement%20for%20Very%20Low%20f23012c2cbea43909d18bd70e44b7f2f/Untitled%208.png)

|  | $d_i^{UI}$ | $d_i^N$ | $d_i^P$  |  ξi |
| --- | --- | --- | --- | --- |
| Easy | large | large | small | large |
| Hard | small | small | large | small |

### Perceptibility Regression Module

RI is learned by Backbone → dropout → FC regression → $L_{L1} smooth L1$ 

where $\beta = 0.75$

![Untitled](Recognizability%20Embedding%20Enhancement%20for%20Very%20Low%20f23012c2cbea43909d18bd70e44b7f2f/Untitled%209.png)

### Index Diversion Loss

Goal: enhance the hard-torecognize instances’ recognizability.

![Untitled](Recognizability%20Embedding%20Enhancement%20for%20Very%20Low%20f23012c2cbea43909d18bd70e44b7f2f/Untitled%2010.png)

where $τ$ is the confidence interval hyperparameter.

|  | ξi | $L_{ID}$​ |  |  |
| --- | --- | --- | --- | --- |
| Easy | large | small |  |  |
| Hard | small | large |  |  |

The ID loss enforces a deviation of at least τ between $\hatξ_i$ and $µ_{UI}$ .

push the hard-to-recognize instances outside the designated τ by minimizing the ID loss

### Perceptibility Attention Module

**goal**: to enhance the recognizability of very low-resolution (VLR) face images by focusing on the most salient and recognizable features within these images.

![Untitled](Recognizability%20Embedding%20Enhancement%20for%20Very%20Low%20f23012c2cbea43909d18bd70e44b7f2f/Untitled%2011.png)

![Untitled](Recognizability%20Embedding%20Enhancement%20for%20Very%20Low%20f23012c2cbea43909d18bd70e44b7f2f/Untitled%2012.png)

where the goal of $v_i^·$ projecting the face embedding away from the center of the UIs cluster. This means adjusting the embedding of a face so that it is less similar to the embeddings of unrecognizable faces. Enhance the recognizability of very low-resolution face images by ensuring that their embeddings in the model are distinct from those of unrecognizable instances, thus improving the model's ability to accurately identify faces.

![Untitled](Recognizability%20Embedding%20Enhancement%20for%20Very%20Low%20f23012c2cbea43909d18bd70e44b7f2f/Untitled%2013.png)

$v_i^{attn}$ is output of the PAM network

Author did not given more details on how $v_i^·$ is mathematically calculated. I assume its vector subtraction or transformation