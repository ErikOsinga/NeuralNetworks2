from __future__ import print_function

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, adam

import glob
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from sklearn import model_selection

def load_data(amount_of_examples,data_dir):
	'''
	Loads the satellite and roadmap images, and splits into train/test set

	amount_of_examples -- number of examples to load in
	'''

	counter = 0
	x_train = []
	y_train = []
	for train_ex in sorted(glob.glob(data_dir+'*_satellite.png')):
		train_input = mpimg.imread(train_ex)
		x_train.append(train_input)

		train_label = mpimg.imread(train_ex.replace('satellite','roadmap'))
		y_train.append(train_label)

		counter += 1
		if counter == amount_of_examples:
			break

	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train)


	print ('Splitting training and test set..')
	x_train, x_test, y_train, y_test = model_selection.train_test_split(x_train, y_train)

	print ('Shape of train/test sets:')
	print ('Shape of x-train', x_train.shape)
	print ('Shape of y-train', y_train.shape)
	print ('Shape of x-test', x_test.shape)
	print ('Shape of y-test', y_test.shape)

	return x_train, y_train, x_test, y_test

data_dir = '/data/s1546449/maps_data/'

x_train, y_train, x_test, y_test = load_data(100,data_dir)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])

y_train = y_train.reshape(y_train.shape[0], y_train.shape[1]*y_train.shape[2]*y_train.shape[3])
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1]*y_test.shape[2]*y_test.shape[3])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print('Final shapes after flattening:')
print (x_train.shape)
print (y_train.shape)
print (x_test.shape)
print (y_test.shape)


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


batch_size = None
epochs = 100

model = Sequential()
# divide number of inputs by 1024
model.add(Dense(786432//(2048*8), activation='relu', input_shape=(786432,)))
# model.add(Dropout(0.2))
# and then multiplying again
model.add(Dense(786432, activation='sigmoid'))
# model.add(Dropout(0.2))

model.summary()

print ('Required GB of memory:',get_model_memory_usage(1,model))

# initial try
# model.compile(loss='mean_squared_error',
#               optimizer=RMSprop(),
#               metrics=['accuracy'])

# stolen from keras MNIST autoencoder blog
# model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy')

# model.compile(optimizer=RMSprop(), loss='squared_hinge')

model.compile(optimizer=adam(lr=0.001), loss='mean_squared_error')

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

def show_image(image):
	plt.imshow(image.reshape(512,512,3))
	plt.show()

def show_input_and_prediction(image_number,predictions):
	x_image = x_train[image_number]
	y_image = y_train[image_number]
	pred_image = predictions[image_number]

	fig = plt.figure()

	ax =  fig.add_subplot(221)
	ax.imshow(x_image.reshape(512,512,3))
	ax.set_title('Input image, %i'%image_number)

	ax = fig.add_subplot(222)
	ax.set_title('True image, %i'%image_number)
	ax.imshow(y_image.reshape(512,512,3))
	
	ax = fig.add_subplot(223)
	ax.imshow(pred_image.reshape(512,512,3))
	ax.set_title('Predicted image, %i'%image_number)
	ax.imshow(pred_image.reshape(512,512,3))
	
	plt.show()


predictions = model.predict(x_train)
predictions_test = model.predict(x_test)
# show_image(predictions[0])
show_input_and_prediction(0,predictions)
show_input_and_prediction(50,predictions)
