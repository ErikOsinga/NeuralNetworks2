import numpy as np
from matplotlib import pyplot as plt

train_data = np.loadtxt('data/train_in.csv',delimiter=',') # Shape is (1707,256)
train_labels = np.loadtxt('data/train_out.csv') # Shape is (1707,)
test_data = np.loadtxt('data/test_in.csv',delimiter=',')# Shape is (1000,256)
test_labels = np.loadtxt('data/test_out.csv') # Shape is (1000,)
digits = range(0,10)

# make sure X is in shape (256, number of examples)
train_data = train_data.T 
test_data = test_data.T

# Add the bias node
train_data = np.append(train_data,[np.ones(train_data.shape[1])],axis=0) 
test_data = np.append(test_data, [np.ones(test_data.shape[1])],axis=0)

def train_single_layer_perceptron(learning_rate = 0.05):
	"""
	Train on the training set untill 100 % accurate. Return the weights 
	"""

	w = np.random.rand(257,10)  # 256 + 1 for the bias, 10 output nodes.

	bad_class = 1e6 # initialization
	full_iterations = 0
	while bad_class > 0:
		bad_class = 0
		full_iterations += 1 

		for i in range(train_data.shape[1]):
			X = train_data[:,i]
			X = X.reshape(257,1)
			a = np.dot(w.T,X) 
			yhat = np.argmax(a)
			
			ylabel = int(train_labels[i]) # the digit that it should be

			if yhat == ylabel:
				pass
			else:
				# Increase the weights were too low
				w[:,ylabel] += train_data[:,i] * learning_rate
				# Decrease the weights that were too high.
				w[:,yhat] -= train_data[:,i] * learning_rate
				bad_class += 1

		# print ('Number of wrong classifications: ', bad_class)
	# print ('Required number of full iterations over the training set: %i' %full_iterations)

	return w, full_iterations

def test_single_layer_perceptron(w):
	"""
	Test given weights w on the test set. Print the accuracy.
	"""

	bad_class = 0
	for i in range(test_data.shape[1]):
		X = test_data[:,i]
		X = X.reshape(257,1)
		a = np.dot(w.T,X) 
		yhat = np.argmax(a)
		
		ylabel = int(test_labels[i]) # the digit that it should be

		if yhat != ylabel:
			bad_class += 1

	# print ('Number of wrong classifications on the test set: ', bad_class)
	accuracy = 1 - bad_class/test_data.shape[1]

	return accuracy

# Quick bool to decide if we want 1 or 1000 iterations.
one_iteration = True

if one_iteration: # Train the single layer perceptron once until 100\% accuracy
	w, full_iterations = train_single_layer_perceptron()
	accuracy = test_single_layer_perceptron(w)
	print ('Accuracy on the test set: ', accuracy)

else: # Train the single layer perceptron 1000 times until 100\% accuracy
	all_iterations = []
	all_accuracy = []
	for initial in range(1000):
		w, full_iterations = train_single_layer_perceptron()
		accuracy = test_single_layer_perceptron(w)

		all_iterations.append(full_iterations)
		all_accuracy.append(accuracy)

	plt.hist(all_iterations)
	plt.title('Number of iterations needed for %i random initializations'  % (initial+1))
	plt.xlabel('Number of iterations')
	plt.ylabel('Count')
	plt.show()

	plt.hist(all_accuracy)
	plt.title('Accuracy for %i random initializations' % (initial+1))
	plt.xlabel('Accuracy')
	plt.ylabel('Count')
	plt.show()
