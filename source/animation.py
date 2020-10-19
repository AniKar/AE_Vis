import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow import keras

mdir = '../models/'
encoder = keras.models.load_model(mdir+'encoder_model', compile=False)
decoder = keras.models.load_model(mdir+'decoder_model', compile=False)
encoding_lenght = 32

img_arr = []
for i in range(10):
    img_arr.append(plt.imread('../images/{}.png'.format(i))[:,:,0])
    
enc_arr = []
for i in range(len(img_arr)):
    enc_arr.append(encoder.predict(img_arr[i].reshape(1, 784)))
    
# create animation by linearly interpolating between latent codes of the images.
def create_interpolation_animation(enc1, enc2, fname):
    fig = plt.figure()
    ims = []
    N = 20
    for t in range(N+1):
      t = t/N
      mid = (1-t) * enc1 + t * enc2
      dec = decoder.predict(mid.reshape(-1, encoding_lenght))
      im = plt.imshow(dec.reshape(28, 28))
      ims.append([im])  
    ims = ims + ims[::-1]
    anim = animation.ArtistAnimation(fig, ims, interval=300, repeat_delay=1000)
    anim.save(fname, writer=animation.PillowWriter(fps=60))
    
odir = '../output/'
create_interpolation_animation(enc_arr[0], enc_arr[9], odir+'animation_0_9.gif')
create_interpolation_animation(enc_arr[8], enc_arr[3], odir+'animation_8_3.gif')
create_interpolation_animation(enc_arr[1], enc_arr[7], odir+'animation_1_7.gif')
create_interpolation_animation(enc_arr[2], enc_arr[5], odir+'animation_2_5.gif')
create_interpolation_animation(enc_arr[4], enc_arr[6], odir+'animation_4_6.gif')
create_interpolation_animation(enc_arr[0], enc_arr[6], odir+'animation_0_6.gif')
