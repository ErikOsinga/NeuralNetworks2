
import numpy as np 
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

def plot_test_acc(testacc,Title):
	"""
	Plots the test accuracy (or loss) for the default settings of the MNIST_mlp 
	as is on the KERAS git.
	"""

	print ('Mean accuracy %4f, standard deviation: %4f' %(np.mean(testacc),np.std(testacc))) 

	plt.title('Test accuracy for 50 initializations of the ' + Title)
	plt.hist(testacc)
	plt.xlabel('Accuracy')
	plt.ylabel('Count')
	# plt.show()
	plt.savefig('./figures/'+Title+'accuracy.png')
	plt.close()


# testacc = np.load('./results/test_accuracy_MLP_default.npy')
# plot_test_acc(testacc,'MLP')

testacc = np.load('./results/test_accuracy_CNN_default.npy')
plot_test_acc(testacc,'CNN')

testacc = np.load('./results/test_accuracy_CNN_permute.npy')
plot_test_acc(testacc,'CNN_permuted')

testacc = np.load('./results/test_accuracy_MLP_default_permute.npy')
plot_test_acc(testacc,'MLP_permuted')

