import numpy as np
from matplotlib import pyplot as plt

def sigmoid(X):
	"""
	Calculate the sigmoid function for vector X
 	"""

	A = 1/(1+np.exp(-X))

	return A


def xor_net(x1,x2,weights):
	"""
	The forward pass of the network for the XOR function.
	x1, x2 are the inputs (0 or 1) and weights is the 
	vector [w1, ... , w9] containing the weights. 
	"""

	# reshape weights for matrix multiplications
	w1 = weights[:6].reshape(3,2)
	w2 = weights[6:].reshape(3,1)

	X1 = np.array([+1,x1,x2]).reshape(3,1)
	# forward pass to hidden layer 1
	A1 = sigmoid( np.dot(w1.T,X1) ) # shape (2,1)
	assert A1.shape == (2,1)
	A1 = np.append(A1,[[+1]],axis=0) # add the bias node
	# forward pass and activation of output layer
	A2 = np.squeeze(sigmoid( np.dot(w2.T,A1) )) # shape (1,1)

	if A2 > 0.5:
		yhat = 1
	else:
		yhat = 0

	return A2, yhat

def mse(weights):
	"""
	Calculate the MSE for the 4 possible examples of the XOR problem.
	Returns the MSE and the amount of wrong predictions by xor_net
	"""
	input_vectors = [ [0,0], [0,1], [1,0], [1,1] ]
	targets = [0, 1, 1, 0]

	predictions = [] # numeric
	yhats = [] # 0 or 1
	for x1,x2 in input_vectors:
		A2, yhat = xor_net(x1,x2,weights)
		predictions.append(A2)
		yhats.append(yhat)

	wrong_predictions = np.sum(np.abs( np.asarray(targets) - np.asarray(yhats)))

	MSE = 1./4 * np.sum( (np.asarray(targets) - np.asarray(predictions) )**2 )

	return MSE, wrong_predictions

def grdmse(weights):
	"""
	Calculate the gradient of the MSE with respect to every weight. 
	"""
	eta = 1e-3 # small step value for numeric derivative calculation

	dw = []
	MSE, wrong_predictions = mse(weights)
	# print ('Mean squared error: ', MSE)
	# print ('Wrong predictions: ', wrong_predictions)

	for i in range(len(weights)):
		new_weights = np.copy(weights)
		new_weights[i] = new_weights[i]+eta
		
		dw_i = (mse(new_weights)[0] - MSE) / eta
		dw.append(dw_i)

	return np.asarray(dw), MSE, wrong_predictions


def gradient_descent(learning_rate=1.0):
	"""
	Implement the gradient descent algorithm using the previous functions.
	"""
	weights = np.random.randn(9)

	all_MSE = []
	all_wrong_predictions = []
	for i in range(3000):
		dw, MSE, wrong_predictions = grdmse(weights)
		all_MSE.append(MSE)
		all_wrong_predictions.append(wrong_predictions)
		weights = weights - learning_rate * dw

	return all_MSE, all_wrong_predictions

def plot_all(all_MSE,all_wrong_predictions,iteration):
	f, axarr = plt.subplots(2, sharex=True)
	axarr[0].set_title('3000 iterations of gradient descent')
	axarr[0].plot(all_MSE)
	axarr[1].plot(all_wrong_predictions)
	axarr[1].set_xlabel('Iterations')
	axarr[0].set_ylabel('Mean squared error')
	axarr[1].set_ylabel('Wrong predictions')
	axarr[1].set_ylim(0,4)
	axarr[0].set_ylim(0,0.4)

	f.tight_layout()
	# plt.show()
	plt.savefig('./figures/%i.png'%iteration)
	plt.close()
	

"""Run the gradient descent algorithm for 20 different random seeds"""
all_i_till_converge = []
n = 20 # Number of different seeds used for the initialization of the weights
for iteration in range(0,n):
	np.random.seed(iteration)
	all_MSE, all_wrong_predictions = gradient_descent(learning_rate=0.5)
	try:
		i_till_converge = np.where(np.asarray(all_wrong_predictions) == 0)[0][0]
	except IndexError:
		i_till_converge = 5000

	all_i_till_converge.append(i_till_converge)
	# plot_all(all_MSE,all_wrong_predictions,iteration)

plt.plot(all_i_till_converge)
plt.xlabel('Random seed')
plt.xticks(range(0,n))
plt.ylabel('Amount of iterations until 0 wrong predictions')
plt.ylim(0,3000)
plt.show()
