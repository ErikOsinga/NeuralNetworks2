from __future__ import print_function

import sys
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, adam
from keras.callbacks import Callback

import glob
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from sklearn import model_selection

'''
Second try of building a simple MLP. This time we will try to fit 28x28x3 images
with a zoom level of 12
or try the splitted images which are 28x28 pix with a zoom level of 15
'''

def load_big_data(amount_of_examples,data_dir):
	"""
	Function that does not use glob to load the images, but uses the consistency in their
	filenames, because globbing millions of images will take too long
	"""
	print ('Loading maximally %i images..'%amount_of_examples)
	counter = 0
	x_train = []
	y_train = []

	for i in range(8900):
		sys.stdout.write("\r%i"%counter)
		sys.stdout.flush()
		
		imagenames = [data_dir+'{}_{}_{}_satellite.png'.format(i,k,l) for k in range(0,17) for l in range(0,18)]
		for train_ex in imagenames:
			try:
				train_input = mpimg.imread(train_ex)
				x_train.append(train_input)

				train_label = mpimg.imread(train_ex.replace('satellite','roadmap'))
				y_train.append(train_label)
				
				counter += 1
		
			except FileNotFoundError:
				print ('File not found: ',train_ex)

			if counter == amount_of_examples:
				break
		if counter == amount_of_examples:
			break

	print ('\n%i images were loaded'%counter)
	
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

def load_data(amount_of_examples,data_dir):
	'''
	Loads the satellite and roadmap images, and splits into train/test set

	amount_of_examples -- number of examples to load in
	!!! Don't use this function in a directory with millions of images.
		Computer does not like that
	'''

	print ('Loading maximally %i images..'%amount_of_examples)
	counter = 0
	x_train = []
	y_train = []
	for train_ex in sorted(glob.glob(data_dir+'*_satellite.png')):
		sys.stdout.write("\r%i"%counter)
		sys.stdout.flush()
		
		try:
			train_input = mpimg.imread(train_ex)
			x_train.append(train_input)

			train_label = mpimg.imread(train_ex.replace('satellite','roadmap'))
			y_train.append(train_label)
		
		except FileNotFoundError:
			print ('File not found: ',train_ex)


		counter += 1
		if counter == amount_of_examples:
			break
	print ('\n%i images were loaded'%counter)

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

def show_image(image):
	plt.imshow(image.reshape(int((dimensionality/3)**0.5),int((dimensionality/3)**0.5),3))
	plt.show()

def show_input_and_prediction(image_number,predictions,x_train,y_train,epoch):
	x_image = x_train[image_number]
	y_image = y_train[image_number]
	pred_image = predictions[image_number]

	fig = plt.figure()

	ax =  fig.add_subplot(221)
	ax.set_title('Input image, %i'%image_number)
	ax.imshow(x_image.reshape(int((dimensionality/3)**0.5),int((dimensionality/3)**0.5),3))

	ax = fig.add_subplot(222)
	ax.set_title('True image, %i'%image_number)
	ax.imshow(y_image.reshape(int((dimensionality/3)**0.5),int((dimensionality/3)**0.5),3))
	
	ax = fig.add_subplot(223)
	ax.set_title('Predicted image, %i'%image_number)
	ax.imshow(pred_image.reshape(int((dimensionality/3)**0.5),int((dimensionality/3)**0.5),3))
	
	# plt.show()
	output_dir = './predictions_network6/'
	print ('Saving output prediction to %s'%output_dir)
	plt.savefig(output_dir+'prediction_%i.png'%epoch)
	plt.close()

def get_model_memory_usage(batch_size, model):
	if batch_size == None:
		batch_size = 32

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

class show_inputprediction(Callback):
    """"
    callback to observe the output of the network
    shows one input image, one true and one predicted every n epochs
    """

    def __init__(self, n, nn, x_train, y_train):
        self.n = n # number of epochs before output
        self.model = nn # the neural network model
        self.x_train = x_train
        self.y_train = y_train


    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.n == 0:
            predictions = self.model.predict(self.x_train)
            image_number = np.random.randint(0,x_train.shape[0])
            show_input_and_prediction(image_number,predictions,self.x_train,self.y_train,epoch)

data_dir = '/data/s1546449/maps_data_28_zoom12_2/'

# Load the data
amount_of_examples = int(12000) 
x_train, y_train, x_test, y_test = load_data(amount_of_examples,data_dir)

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

batch_size = None
epochs = 500

dimensionality = x_train.shape[1]

model = Sequential()
# divide number of inputs by 8
model.add(Dense(dimensionality//(8), activation='relu', input_shape=(dimensionality,)))
# model.add(Dropout(0.2))
# and then multiplying again
model.add(Dense(dimensionality, activation='sigmoid'))
# model.add(Dropout(0.2))

model.summary()

print ('Required GB of memory:',get_model_memory_usage(batch_size,model))

# initial try
# model.compile(loss='mean_squared_error',
#			   optimizer=RMSprop(),
#			   metrics=['accuracy'])

# stolen from keras MNIST autoencoder blog
model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy')

# model.compile(optimizer=RMSprop(), loss='squared_hinge')

# model.compile(optimizer=adam(lr=0.001), loss='mean_squared_error')

# save output image every output_epochs epochs
output_epochs = 50
show_callback = show_inputprediction(output_epochs,model,x_train,y_train)
history = model.fit(x_train, y_train,
					batch_size=batch_size,
					epochs=epochs,
					verbose=1,
					validation_data=(x_test, y_test),
					callbacks=[show_callback])

score = model.evaluate(x_test, y_test, verbose=0)

plt.title('Loss and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(history.history['loss'],label='train_loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()
plt.savefig('./predictions_network6/loss.png')

print ('Saving Model...')
model.save('./archive/network6.h5')

# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


# predictions = model.predict(x_train)
# predictions_test = model.predict(x_test)
# show_image(predictions[0])
# show_input_and_prediction(0,predictions)
# show_input_and_prediction(50,predictions)
