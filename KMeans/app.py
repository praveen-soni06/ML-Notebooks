# from kmeans import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

centroid = [(-5,-5), (5,5)]
cluster_std = [1,1]

X,y = make_blobs(n_samples=100, cluster_std=cluster_std, centers=centroid, n_features=2, random_state=2)

plt.scatter(X[:,0], X[:,1])
plt.show()