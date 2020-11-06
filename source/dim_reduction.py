import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE, Isomap
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

# LDA
lda = LDA(n_components=2)
transformed_enc_lda = lda.fit_transform(enc, y)

# ISOMAP
isomap = Isomap(n_components=2, n_neighbors=10)
transformed_enc_isomap = isomap.fit_transform(enc, y)

# tSNE
tsne = TSNE(n_components=2, perplexity=10)
transformed_enc_tsne = tsne.fit_transform(enc, y)

# show the test images on the corresponding 2D encoding locations 
def create_2D_embedding_plot(enc, ax_lim, file_name, plot_name, n_samples=100):
    _, ax = plt.subplots()
    plt.scatter(enc[:,0], enc[:,1], c=y)
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    
    rind = random.sample(range(transformed_enc_lda.shape[0]), n_samples)    
    for i in rind:
        img = x[i].reshape(28, 28)
        imagebox = OffsetImage(img, zoom=0.4)
        ab = AnnotationBbox(imagebox, (enc[i,0], enc[i,1]), pad=0.3)
        ax.add_artist(ab)
    plt.title(plot_name)
    plt.savefig(file_name)
    plt.show()
    
odir = '../output/'
create_2D_embedding_plot(transformed_enc_pca, ax_lim=2, file_name=odir+'pca_embedding.png', plot_name='PCA')
create_2D_embedding_plot(transformed_enc_lda, ax_lim=8, file_name=odir+'lda_embedding.png', plot_name='LDA')
create_2D_embedding_plot(transformed_enc_isomap, ax_lim=3, file_name=odir+'isomap_embedding.png', plot_name='ISOMAP')
create_2D_embedding_plot(transformed_enc_tsne, ax_lim=100, file_name=odir+'tsne_embedding.png', plot_name='tSNE')