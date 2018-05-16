'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''
from __future__ import print_function

import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

Permute = True

if Permute: 
	# Make a random permutation for each example in the train/test set
	p = np.arange(784)
	np.random.shuffle(p)

	for i in range(x_train.shape[0]):
	  xtrain_i = x_train[i]
	  x_train[i] = xtrain_i[p] # permute it and put it back

	for i in range(x_test.shape[0]):
	  xtest_i = x_test[i]
	  x_test[i] = xtest_i[p] # permute it and put it back

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



# Make 50 models and save their accuracy and loss on the test set
test_loss = []
test_accuracy = []
for i in range(0,50):
	model = Sequential()
	model.add(Dense(512, activation='relu', input_shape=(784,)))
	model.add(Dropout(0.2))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))

	model.summary()

	model.compile(loss='categorical_crossentropy',
	              optimizer=RMSprop(),
	              metrics=['accuracy'])
	history = model.fit(x_train, y_train,
	                    batch_size=batch_size,
	                    epochs=epochs,
	                    verbose=0,
	                    validation_data=(x_test, y_test),shuffle=False) #Shuffle=False for reproducibility
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	test_loss.append(score[0])
	test_accuracy.append(score[1])

np.save('./test_loss_MLP_default_permute',test_loss)
np.save('./test_accuracy_MLP_default_permute',test_accuracy)

"""
Test loss: 0.11605441395899883
Test accuracy: 0.9828
"""

"""
After randomly permuting the input we get
Test loss: 0.10164622664727367
Test accuracy: 0.9844

"""


"""
When fixing the random seed to be 1337 we get
- No permutation:
	run 0
	Test loss: 0.1175703917319342
	Test accuracy: 0.9822
	run 1
	Test loss: 0.11221849588753162
	Test accuracy: 0.983

It seems to change, but according to some guy on the internet we have to close python
and then run it again to get consistent results, also set shuffle=False
- Try again, no permutation: 
	Test loss: 0.11383037965887834
	Test accuracy: 0.9827

	Test loss: 0.1262502011892741
	Test accuracy: 0.9824

So still doesnt work by setting the random seed.
So we'll settle on this and run it with permutation:
- Random Permutation:
	Test loss: 0.10750451239173577
	Test accuracy: 0.9846

	Test loss: 0.11830423643202967
	Test accuracy: 0.9819








"""

