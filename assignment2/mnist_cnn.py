'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# To retrieve the precision, recall, f-score and support for each class.
from sklearn.metrics import classification_report

import numpy as np

batch_size = 128
num_classes = 10
epochs = 12#12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

Permute = True

if Permute: 
  # Make a random permutation for each example in the train/test set
  p = np.arange(28*28)
  np.random.shuffle(p)

  for i in range(x_train.shape[0]):
    xtrain_i = x_train[i].flatten()
    xtrain_i = xtrain_i[p] # permute it and put it back
    x_train[i] = xtrain_i.reshape(28,28,1)

  for i in range(x_test.shape[0]):
    xtest_i = x_test[i].flatten()
    xtest_i = xtest_i[p] # permute it and put it back
    x_test[i] = xtest_i.reshape(28,28,1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



test_loss = []
test_accuracy = []
for i in range(0,50):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,#keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    test_loss.append(score[0])
    test_accuracy.append(score[1])
    
np.save('./test_loss_CNN_permute',test_loss)
np.save('./test_accuracy_CNN_permute',test_accuracy)

"""
Much much slower.

Test loss: 0.04145523567434866
Test accuracy: 0.9863

We define most misclassified as the 3 digits that have the least precision.
Precision: Out of all digits that were classified as 'C' how many were actually correct?

For now these are digit 7, 0 and 4 in order of worst to best.

"""

predictions = model.predict(x_test)
predictions = predictions.argmax(axis=1)
y_test2 = y_test.argmax(axis=1)

print (classification_report(y_test2,predictions,digits=5))


"""
For using mean squared error we get:

Test loss: 0.006500150103870692
Test accuracy: 0.9596

So it's worse than categorical cross-entropy cost.

Now the worst digits are 8, 3 and 0 in order of worst to best

Running it again gives:
Test loss: 0.007368972765450598
Test accuracy: 0.9501

But we should do this a lot of times and plot the distribution.
"""

"""
Using a random permutation p and running it again with MSE we get:

Test loss: 0.012154618894844316
Test accuracy: 0.9173


"""
