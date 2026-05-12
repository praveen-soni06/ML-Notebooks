import numpy as np
import random


class KMeans:

    def __init__(self, clusters=2, max_iter=100):

        self.clusters = clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self,X):

        random_index = random.sample(range(0,X.shape[0]), self.clusters)
        self.centroids = X[random_index]

        for i in range(self.max_iter):

            # assign clusters
            cluster_group = self.assign_cluster(X)
            old_centroids = self.centroids

            # move centroids
            self.centroids = self.move_cemntroids(X, cluster_group)

            # check finish
            if (old_centroids == self.centroids).all():
                break

        return cluster_group


    def assign_cluster(self,X):

        cluster_group = []
        distances = []

        for row in X:
            for centroid in self.centroids:
                distances.append(np.sqrt(np.dot(row-centroid, row-centroid)))