# Autoencoder_mnist

The repository contains an autoencoder model implementation in *Keras*, which is trained on *MNIST* dataset of handwritten digits.

Animations are created to demonstrate the interpolation property of the latent space, i.e. linear interpolation in latent space results in smooth changes in image space.
Some of the outstanding examples are:

<p float="left">
  <img src="https://github.com/AniKar/Autoencoder_mnist/blob/master/output/animation_0_9.gif" width="300" />
  <img src="https://github.com/AniKar/Autoencoder_mnist/blob/master/output/animation_8_3.gif" width="300" /> 
  <img src="https://github.com/AniKar/Autoencoder_mnist/blob/master/output/animation_1_7.gif" width="300" />
  <img src="https://github.com/AniKar/Autoencoder_mnist/blob/master/output/animation_2_5.gif" width="300" />
  <img src="https://github.com/AniKar/Autoencoder_mnist/blob/master/output/animation_4_6.gif" width="300" /> 
  <img src="https://github.com/AniKar/Autoencoder_mnist/blob/master/output/animation_0_6.gif" width="300" />
</p>

Also, several dimensionaluty reduction techniques are used to transform the latent representation (32D) of images into 2D embedding, namaely PCA, LDA, Isomap and tSNE methods.

![alt text](https://github.com/AniKar/Autoencoder_mnist/blob/master/output/pca_embedding.png)

![alt text](https://github.com/AniKar/Autoencoder_mnist/blob/master/output/lda_embedding.png)

![alt text](https://github.com/AniKar/Autoencoder_mnist/blob/master/output/isomap_embedding.png)

![alt text](https://github.com/AniKar/Autoencoder_mnist/blob/master/output/tsne_embedding.png)

It can be observed that out of the above methods tSNE achieves relatively better results in finding 2D embedding that exhibits quite good separation property between the 10 classes.
