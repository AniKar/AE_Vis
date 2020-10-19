from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import initializers
import matplotlib.pyplot as plt
import os

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

n_train_samples = x_train.shape[0]
n_test_samples = x_test.shape[0]
flat_dim = x_train.shape[1] * x_train.shape[2]

x_train = x_train.reshape(n_train_samples, flat_dim)
x_test = x_test.reshape(n_test_samples, flat_dim)

input_pholder = Input(shape=(flat_dim,))
encoding_length = 32

# build the autoencoder model
e1 = Dense(256,
           activation='sigmoid', 
           kernel_initializer=initializers.RandomNormal(stddev=0.01),
           bias_initializer=initializers.RandomNormal(stddev=0.01)
           )(input_pholder)

e2 = Dense(128,
           activation='sigmoid',
           kernel_initializer=initializers.RandomNormal(stddev=0.01),
           bias_initializer=initializers.RandomNormal(stddev=0.01)
           )(e1)

encoding = Dense(encoding_length,
                 activation='sigmoid',
                 kernel_initializer=initializers.RandomNormal(stddev=0.01),
                 bias_initializer=initializers.RandomNormal(stddev=0.01)
                 )(e2)

d1 = Dense(128,
           activation='sigmoid',
           kernel_initializer=initializers.RandomNormal(stddev=0.01),
           bias_initializer=initializers.RandomNormal(stddev=0.01)
           )(encoding)


d2 = Dense(256,
           activation='sigmoid',
           kernel_initializer=initializers.RandomNormal(stddev=0.01),
           bias_initializer=initializers.RandomNormal(stddev=0.01)
           )(d1)

decoding = Dense(flat_dim,
                 activation='sigmoid',
                 kernel_initializer=initializers.RandomNormal(stddev=0.01),
                 bias_initializer=initializers.RandomNormal(stddev=0.01)
                 )(d2)

autoencoder = Model(input_pholder, decoding)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


# extract encoder and decoder models
encoder = Model(input_pholder, encoding)
decoder_input = Input(shape=(encoding_length,))
deco = autoencoder.layers[-3](decoder_input)
deco = autoencoder.layers[-2](deco)
deco = autoencoder.layers[-1](deco)
decoder = Model(decoder_input, deco)


def plot_images(images, grid_size=10):
    assert len(images) >= grid_size**2
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28, 28), cmap=plt.cm.binary)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show(block=False)

# plot some of the original and reconstructed test images
plot_images(x_test)
decoded_imgs = autoencoder.predict(x_test)
plot_images(decoded_imgs)

# save the models
mdir = '../models/'
if not os.path.exists(mdir+'autoencoder_model'):
    autoencoder.save(mdir+'autoencoder_model')
    
if not os.path.exists(mdir+'encoder_model'):
    encoder.save(mdir+'encoder_model')

if not os.path.exists(mdir+'decoder_model'):
    encoder.save(mdir+'decoder_model')