import numpy as np 
from matplotlib import pyplot as plt




X = np.array([ [0,0,1,1],
				 [0,1,0,1]])


AND = np.array([0,0,0,1])
XOR = np.array([0,1,1,0])

def question1():
	plt.title('Feature space for AND')
	plt.scatter(X[0],X[1],c=AND)
	plt.show()

	plt.title('Feature space for XOR')
	plt.scatter(X[0],X[1],c=XOR)
	plt.show()

w = np.array([ [1.31] , [0.844]])
# b = np.array([[-1.63], [-1.38]])
b = -1.63


def question2():
	yhat = np.dot(w.T,X) + b
	H = np.heaviside(yhat,0)
	print (H),'\n\n'
	# yhat = w[0] * X[0] + w[1] * X[1] + b
	# H = np.heaviside(yhat,0)
	# print (H),'\n\n'

def question3():
	x1, x2 =  np.arange(0,1,0.01), np.arange(0,1,0.01)
	x1, x2 = np.meshgrid(x1,x2)
	# x1, x2 = np.meshgrid(X[0],X[1]) # doesnt work because not enough values?

	yhat = w[0] * x1 + w[1] * x2 + b
	H = np.heaviside(yhat,0)
	# print (H)

	plt.contourf(x1,x2,H)
	plt.colorbar()
	plt.xlim(-0.2,1.1)
	plt.ylim(-0.2,1.1)
	plt.show()

# question1()
# question2()
question3()
