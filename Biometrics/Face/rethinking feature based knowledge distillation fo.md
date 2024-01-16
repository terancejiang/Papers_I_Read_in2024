# rethinking feature based knowledge distillation for face recognition

Source: [https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Rethinking_Feature-Based_Knowledge_Distillation_for_Face_Recognition_CVPR_2023_paper.pdf](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Rethinking_Feature-Based_Knowledge_Distillation_for_Face_Recognition_CVPR_2023_paper.pdf)

### Insights:

1. **Intrinsic dimension** in the context of neural networks refers to the minimum number of parameters needed to accurately represent the underlying structure of the data.
2. **Intrinsic gap** is intrinsic dimension difference between teacher and student
3. teacher student capacity gap partially influenced by the intrinsic gap
4. Align teacher and student intrinsic dimension can improved kd outcomes. 

**TL;DR:** The teacher model becomes more effective due to its lower intrinsic dimension, typically linked with enhanced generalization capability and performance. This research proposes that the capacity gap in Knowledge Distillation (KD) is partly due to the difference in intrinsic dimensions between teacher and student models, referred to as the 'intrinsic gap'. The main objective shifts to minimizing this intrinsic gap.

To achieve this, the strategy involves constraining the teacher model’s training within the student's feature space, ensuring a tailored approach. Subsequently, the student model undergoes training through feature-only distillation. This method aims to yield improved KD outcomes by reducing the intrinsic gap between the teacher and student models.

### **Method:**

1. **Stage 1:** Initially, the student model (S`) is trained on dataset D1 using only classification loss.
2. **Stage 2:** The student model (S) infers dataset D1 to obtain feature embeddings (SD1).
3. **Stage 3:** The teacher model is trained with dataset D1 using classification loss and the embeddings S`-D1 (distillation loss). This step narrows the teacher’s exploration to higher intrinsic dimensions, aligning more closely with the student's inherent characteristics.
4. **Stage 4:** The teacher model infers dataset D2, generating feature embeddings (T`-D2). Since class information isn't needed in stage 5, D2 can be a large, unlabeled dataset.
5. **Stage 5:** The student model (S) is trained on the dataset D2 embeddings (T`-D2) using only distillation loss.

![Untitled](rethinking%20feature%20based%20knowledge%20distillation%20fo%20caf82eef9b1d4a1a8cc05258658c0e53/Untitled.png)

### Results:

![Untitled](rethinking%20feature%20based%20knowledge%20distillation%20fo%20caf82eef9b1d4a1a8cc05258658c0e53/Untitled%201.png)

## intrinsic dimension

Intrinsic dimension in the context of neural networks refers to the minimum number of parameters needed to accurately represent the underlying structure of the data. This concept is essential in understanding and optimizing neural networks.

🔍 **Intrinsic Dimension**: At its core, intrinsic dimension is about how complex or simple the structure of your data is. For instance, even if you have a dataset with hundreds of features (like pixels in an image), the actual variation in the data might be captured by a far smaller number of underlying factors or dimensions. This "true" number of dimensions needed to represent the data without significant loss of information is what we call the intrinsic dimension.

💻 **Neural Networks and Intrinsic Dimension**: When designing neural networks, understanding the intrinsic dimension of your data can be crucial. If your network has too few parameters (underfitting), it might not capture the full complexity of the data. Conversely, too many parameters (overfitting) can lead to a model that is unnecessarily complex, which can increase training time, require more data, and decrease the model's ability to generalize to new data.

📉 **Dimensionality Reduction Techniques**: Techniques like Principal Component Analysis (PCA) are often used to estimate the intrinsic dimension of a dataset. PCA reduces the dimensionality of the data by finding the most significant dimensions, which can give an insight into its intrinsic dimension.

🌀 **Manifold Hypothesis**: This concept is closely related to the manifold hypothesis in machine learning, which suggests that real-world high-dimensional data (like images, sound recordings, etc.) actually lie on a low-dimensional manifold. This implies that the intrinsic dimension of such data is lower than its superficial dimensionality.

📚 **Implications in Model Design**: Understanding the intrinsic dimension helps in designing more efficient neural networks. It guides the selection of an appropriate architecture, the number of layers, and the number of neurons in each layer, balancing complexity and performance.