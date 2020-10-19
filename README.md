# Autoencoder_mnist

The repository contains an autoencoder model implementation in *Keras*, which is trained on *MNIST* dataset of handwritten digits.

Animations are created to demonstrate the interpolation property of the latent space, i.e. linear interpolation in latent space results in smooth changes in image space.
Some of the outstanding examples are:

![alt text](https://github.com/AniKar/Autoencoder_mnist/blob/master/output/animation_0_9.gif)
![alt text](https://github.com/AniKar/Autoencoder_mnist/blob/master/output/animation_8_3.gif)
![alt text](https://github.com/AniKar/Autoencoder_mnist/blob/master/output/animation_1_7.gif)
![alt text](https://github.com/AniKar/Autoencoder_mnist/blob/master/output/animation_2_5.gif)
![alt text](https://github.com/AniKar/Autoencoder_mnist/blob/master/output/animation_4_6.gif)
![alt text](https://github.com/AniKar/Autoencoder_mnist/blob/master/output/animation_0_6.gif)


Also, Principal component analysis (PCA) and Linear Discriminant Analysis (LDA) dimensionaluty reduction techniques are used to transform the latent representation (32D) of images to 2D embedding.

PCA results:

![alt text](https://github.com/AniKar/Autoencoder_mnist/blob/master/output/pca_encoding.png)

LDA results:

![alt text](https://github.com/AniKar/Autoencoder_mnist/blob/master/output/lda_encoding.png)

It's clear that *none* of the used methods achieves impressive separation between classes (for 2D).
