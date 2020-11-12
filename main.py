import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles

plt.figure()


# plt.subplot()



#generate data
plt.title("Two informative features, one cluster per class", fontsize='small')
X, Y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, s=25, edgecolor='k')
plt.show()

#train_model


# generate adv examples

#adversarial activity detection

#adversarial activity type detection

#defense strategy

#risk configuration
#calculate the probability of risk associated wih each class
#choose the class with minimal risk