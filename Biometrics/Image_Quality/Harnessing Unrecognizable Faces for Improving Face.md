# Harnessing Unrecognizable Faces for Improving Face Recognition

**insights**:

Simple yet effective method can be used to clean training dataset. 

**tl;dr**: based on the hypothesis and empirical observation that unrecognizable images embedding are more likely to join a common cluster such as: 

![Untitled](Harnessing%20Unrecognizable%20Faces%20for%20Improving%20Face%201110092d1d9e440f80e2eed34cd444d9/Untitled.png)

Author based on this assumption proposed:

1. measure of recognizability that leverages the existence of a single UI cluster in the learned embedding. The “embedding recognizability score” (ERS) is simply the Euclidean (cordal) distance of the embedding of an image from the center of the UI cluster in the hypersphere. (Distance between unrecognizable identity (UI) cluster vs given image)
2. use the ERS to mitigate the detrimental effect of using different datasets for FD and FR. This results in 58% error reduction
3. propose an aggregation method for set-based face recognition, by simply calculating the weighted average relative to the ERS, and report 24% error reduction

## Method:

### Embedding Recognizability Score (ERS)

The normalized average embedding of UI images $f_{UI}$ , which we call the UI centroid (UIC), is then used to represent the UI. Given an embedding vector  $f_i$ , its ERS if image i $e_i$ is defined as

![Untitled](Harnessing%20Unrecognizable%20Faces%20for%20Improving%20Face%201110092d1d9e440f80e2eed34cd444d9/Untitled%201.png)

During face identification, image can be filter using $e_i >=γ(0.60)$

### Implementation details

We use the deep learning framework MXNet [2] in our model training and evaluation.
We train a face embedding model using CosFace [34] loss, ResNet-101 (R101) [9] backbone and DeepGlintFace dataset (including MS1M-DeepGlint and AsianDeepGlint) [5]. HAC algorithm [4] is used to cluster extracted embeddings and generate UI clusters. We select threshold γ = 0.60 for ERS via cross-validation on the TinyFace [37] benchmark.