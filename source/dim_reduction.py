import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import random

(_, _), (x, y) = mnist.load_data()
x = x.astype('float32') / 255.
n_samples = x.shape[0]
flat_dim = x.shape[1] * x.shape[2]
x = x.reshape(n_samples, flat_dim)

mdir = '../models/'
encoder = keras.models.load_model(mdir+'encoder_model', compile=False)
decoder = keras.models.load_model(mdir+'decoder_model', compile=False)
encoding_lenght = 32
enc = encoder.predict(x)

# PCA
pca = PCA(n_components=2)
transformed_enc_pca = pca.fit_transform(enc)

fig = plt.figure(figsize=(8,8))
plt.scatter(transformed_enc_pca[:,0], transformed_enc_pca[:,1], c=y)
plt.show()

# LDA
lda = LinearDiscriminantAnalysis(n_components=2)
transformed_enc_lda = lda.fit_transform(enc, y)

fig = plt.figure(figsize=(8,8))
plt.scatter(transformed_enc_lda[:,0], transformed_enc_lda[:,1], c=y)
plt.show()

# show the test images on the corresponding 2D encoding locations 
def create_2D_encoding_plot(enc, ax_lim, fname, n_samples=300):
    fig, ax = plt.subplots()
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    
    rind = random.sample(range(transformed_enc_lda.shape[0]), n_samples)    
    for i in rind:
        img = x[i].reshape(28, 28)
        imagebox = OffsetImage(img, zoom=0.4)
        ab = AnnotationBbox(imagebox, (enc[i,0], enc[i,1]), pad=0.3)
        ax.add_artist(ab)
    plt.savefig(fname)
    plt.show()
    
odir = '../output/'
create_2D_encoding_plot(transformed_enc_pca, ax_lim=2, fname=odir+'pca_encoding.png')
create_2D_encoding_plot(transformed_enc_lda, ax_lim=8, fname=odir+'lda_encoding.png')