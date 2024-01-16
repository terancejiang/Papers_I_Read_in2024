# Masked Autoencoders As Spatiotemporal Learners

### Insights:

1. the difference for masked autoencoding between vision and language:
    1. The difference in the use of positional embeddings or mask tokens versus regular grids. In vision, autoencoders often utilize regular grids, while in language processing, positional embeddings or mask tokens are more common.
    2. The information density in language is typically high in semantics and information, whereas images are characterized by significant spatial redundancy.
    3. The nature of the autoencoder’s decoder in reconstructing text versus images. In language, the decoder focuses on text reconstruction, while in vision, it's geared towards image reconstruction.
2. Simple self-supervised method like MAE provides scalable benefits like in NLP.
3. MAE reconstruct pixels which are not semantic entities but experiments shows it has learned numerous visual concepts(semantics) 
    
    *Scalable benefits in NLP refer to the advantages gained when NLP systems and algorithms are designed to efficiently handle increasing amounts of data and computational tasks. 
    

### **Problem Addressed**

The paper focuses on the application of masked autoencoding, a technique pivotal in NLP self-supervised pretraining through models like BERT. Masked autoencoding operates on a straightforward principle: it omits a portion of the data and learns to predict this removed content. The authors aim to extend this concept, primarily used in natural language processing, to the realm of vision tasks. The paper's title suggests their objective to establish Masked Autoencoders as scalable learners in visual data processing.

## Approach

![Untitled](Masked%20Autoencoders%20As%20Spatiotemporal%20Learners%20f37dfd7b11a44478926929757714e0a8/Untitled.png)

Encoder: maps the observed signal to a latent representation; use an asymmetric design ensure encoder to operate only on the partial observed signal. 

Light weight decoder: reconstructs the original signal from the latent representation.

### Masking:

Sample (i.e., remove) random patches without replacement, following a uniform distribution

### MAE encoder:

Vit that only on **unmasked** patches (linear projection with added **positional embeddings**)

### MAE decoder:

input: 1. encoded visible patches; 2. mask tokens which is a shared learned vector. Positional embeddings for 1 and 2.

decoder only used during pre-training to perform the image reconstruction task.

only the encoder is used to produce image representations for recognition.

decoder design is independent and can be small, that significantly reduce pretraining time.

Decoder design experiments: 

### Reconstruction target:

MAE reconstructs the input by predicting the **pixel values** for each masked patch

Loss = MAE(reconstructed; original image), image normalised using mean and std of single patch. 

### Implementation:

step 1: input Image(B,C,H,W) to patch(B,N,PxPxC) where N = H*W/P^2 P is patch size.  → token(positional embedding + linear projection) (B, N, D)

step 2: shuffle list of tokens, remove last n portion (75%) of tokens.

step 3: tokens → MAE encoding → encoded patches

step 4: append mask tokens to encoded patches, and unshuffle to align with original image.

step 5: unshuffled tokens (with positional embedding) → MAE decoder → decoded patches

step 6: MSE(decoded patches, original image)

![Untitled](Masked%20Autoencoders%20As%20Spatiotemporal%20Learners%20f37dfd7b11a44478926929757714e0a8/Untitled%201.png)