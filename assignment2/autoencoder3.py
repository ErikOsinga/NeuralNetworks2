import keras

from keras.layers import Input, Dense
from keras.models import Model

from keras.datasets import mnist
import numpy as np

#Trying to autoencode arabic characters

batch_size = 128
num_classes = 29
epochs = 20

# 13440 images of 32x32 pixels, but flattened
x_train = np.loadtxt('/home/s1546449/NeuralNetworks/assignment2/data/arabic/csvTrainImages13440x1024.csv',delimiter=',')
x_test = np.loadtxt('/home/s1546449/NeuralNetworks/assignment2/data/arabic/csvTestImages3360x1024.csv',delimiter=',')
y_train = np.loadtxt('/home/s1546449/NeuralNetworks/assignment2/data/arabic/csvTrainLabel13440x1.csv',delimiter=',')
y_test = np.loadtxt('/home/s1546449/NeuralNetworks/assignment2/data/arabic/csvTestLabel3360x1.csv',delimiter=',')

x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 32, assuming the input is 1024 floats

# this is our input placeholder
input_img = Input(shape=(1024,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(1024, activation='sigmoid')(encoded)

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
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(32, 32))
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