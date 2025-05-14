import numpy as np

def euclidean_distance(X, C):
    return np.linalg.norm(X[:, np.newaxis] - C, axis=2) ** 2


def initialize_centroids(X, k):
    indices = np.random.choice(len(X), size=k, replace=False)
    return X[indices]


def update_membership(X, C, gamma, m):
    dist = euclidean_distance(X, C)
    U = 1 / (1 + (dist / gamma[np.newaxis, :]) ** (1 / (m - 1)))
    return U


def update_centroids(X, U, m):
    um = U ** m
    C = (um.T @ X) / np.sum(um.T, axis=1)[:, np.newaxis]
    return C


def update_gamma(U, dist, m, eta):
    um = U ** m
    numerator = np.sum(um * dist, axis=0)
    denominator = np.sum(um, axis=0)
    gamma = eta * (numerator / denominator)
    return gamma


def pcm(X, k, m=2.0, eta=0.5, max_iter=100, tol=1e-4):
    C = initialize_centroids(X, k)
    gamma = np.ones(k)
    prev_C = C.copy()

    for _ in range(max_iter):
        dist = euclidean_distance(X, C)
        U = update_membership(X, C, gamma, m)
        C = update_centroids(X, U, m)
        gamma = update_gamma(U, dist, m, eta)

        # Check for convergence
        if np.linalg.norm(C - prev_C) < tol:
            break
        prev_C = C.copy()

    return C, U