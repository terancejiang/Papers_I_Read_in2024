# SDD-FIQA: Unsupervised Face Image Quality Assessment with Similarity Distribution Distance

### Insight:

1. Wasserstein Distance can be used to describe the distance between two distributions.
2. From my implementation, if an ID with few low quality training images, the generated pseudo-labels can be very dirty, that low quality image gets a very high score. 
    
    Therefore, I firstly removed class that have less than 10 images from my training set, then I remove top 5% highest score images and re-normalized the rest, achieved a better results.
    

**TL;DR**:The authors propose a novel method for generating pseudo-labels to assess the quality of face images, which are then used to train a regression neural network. 

This method is based on the principle that a high-quality face image should be similar to its intra-class samples and dissimilar to its inter-class samples. Therefore, a robust face recognition model is required to extract features from each image. These features are then used to generate pseudo-labels for face image quality assessment

## Method:

**Step 1:** Utilize a robust face feature neural network to extract features from all images in the given training dataset.

**Step 2:** Pos-sim Distribution: For each face image within the same class (ID), calculate the cosine similarity with other images in the same class. This results in a cosine similarity distribution for each image.

**Step 3:** Neg-sim Distribution: For each image, calculate the cosine similarity with a randomly selected set of **n** images from different classes (IDs). This operation is repeated 12 times (an empirical choice made by the authors), forming a cosine similarity distribution for Neg-sim.

**Step 4: T**he distance between the Pos-sim and Neg-sim distributions is calculated using the Wasserstein Distance. Based on the assumption that a high-quality face image should be similar to its intra-class samples and dissimilar to its inter-class samples, If these two distributions are far apart, the quality of the face image is considered high, and vice versa.

**Step 5:** Normalize the Wasserstein Distance to a range of 0 to 100. The resulting value is the quality score for that image.

**Step 6:** Use the generated scores to train a regression neural network. The authors chose Huber loss as their loss function.

![Untitled](SDD-FIQA%20Unsupervised%20Face%20Image%20Quality%20Assessmen%20c202dd29fa864e3d80b02d421a874c98/Untitled.png)