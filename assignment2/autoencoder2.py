import keras

from keras.layers import Input, Dense
from keras.models import Model

import numpy as np
import imageio

import sklearn.model_selection

import matplotlib.pyplot as plt

import glob

#Trying to autoencode fruits

batch_size = 128
num_classes = 10
epochs = 20

x_train = []
x_test = []

for i in glob.glob('/home/s1546449/data/fruits-360/Training/*'):
    # i =  fruit directory
    print (i)
    for j in glob.glob(i+'/*'):
        im = imageio.imread(j) #100x100x3 pixels
        im = im.reshape(30000)
        x_train.append(im)
        # j = images




x_train = np.asarray(x_train,dtype='float')
# x_train is then shape (num_examples,10000)

x_train, x_test = sklearn.model_selection.train_test_split(x_train)

x_train /= 255.
x_test /= 255.

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# this is the size of our encoded representations
encoding_dim = 100  # 100 floats -> compression of factor 300, assuming the input is 30000 floats

# this is our input placeholder
input_img = Input(shape=(30000,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(30000, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# Create seperate encoder and decoder models
encoder = Model(input_img, encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))


autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

hist = autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# use Matplotlib 

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(100, 100, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(100, 100, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

display = False

if display:
    # display the encoded representation
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2,n,i+1)
        plt.imshow(encoded_imgs[i].reshape(4,8))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display reconstruction
        x = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)    
    plt.show()