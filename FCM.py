import numpy as np

class FuzzyCMeans:
    def __init__ (self, n_clusters=2, m=2, max_iter=100, error=1e-5, random_state=None):
        self.n_clusters = n_clusters
        self.m = m  
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state

    def _euclidean(self, a, b):
        return np.linalg.norm(a - b)

    def _initialize_centroids(self, X):
        np.random.seed(self.random_state)
        centroids = []

        
        first_idx = np.random.choice(len(X))
        centroids.append(X[first_idx])

        
        for _ in range(1, self.n_clusters):
            distances = np.array([
                min([self._euclidean(x, c) for c in centroids]) for x in X
            ])
            next_idx = np.argmax(distances)
            centroids.append(X[next_idx])

        return np.array(centroids)

    def _update_membership(self, X, centroids):
        n_samples = X.shape[0]
        U = np.zeros((n_samples, self.n_clusters))

        for i in range(n_samples):
            for j in range(self.n_clusters):
                denom = 0
                for k in range(self.n_clusters):

                    num = self._euclidean(X[i], centroids[j]) + 1e-10
                    den = self._euclidean(X[i], centroids[k]) + 1e-10

                    denom += (num / den) ** (2 / (self.m - 1))
                U[i][j] = 1 / denom
        return U

    def _update_centroids(self, X, U):
        um = U ** self.m
        centroids = (np.dot(um.T , X)) / np.sum(um.T, axis=1, keepdims=True)
        return centroids

    def fit(self, X):
        X = np.array(X)
        self.centroids_ = self._initialize_centroids(X)
        for i in range(self.max_iter):
            U = self._update_membership(X, self.centroids_)
            new_centroids = self._update_centroids(X, U)

            if np.linalg.norm(new_centroids - self.centroids_) < self.error:
                break
            self.centroids_ = new_centroids
        self.U_ = U

    def predict(self, X):
        X = np.array(X)
        U = self._update_membership(X, self.centroids_)
        return np.argmax(U, axis=1)

    def get_membership_matrix(self):
        return self.U_

    def get_centroids(self):
        return self.centroids_