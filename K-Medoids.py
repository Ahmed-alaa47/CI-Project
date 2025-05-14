import numpy as np
import pandas as pd
from tqdm import tqdm

def manhattan_distance(x1, x2):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    return np.sum(np.abs(x1 - x2))

class KMedoids:
    def __init__(self, k, max_iterations=100, distance_func=manhattan_distance, random_state=None):
        self.k = k
        self.max_iterations = max_iterations
        self.distance_func = distance_func
        self.random_state = random_state
        self.medoid_indices = None
        self.labels_ = None

    def _initialize_medoids(self, n_samples):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        return np.random.choice(n_samples, self.k, replace=False)

    def _assign_labels(self, data, medoids):
        labels = np.zeros(data.shape[0], dtype=int)
        for i in range(data.shape[0]):
            distances = [self.distance_func(data[i], medoid) for medoid in medoids]
            labels[i] = np.argmin(distances)
        return labels

    def _compute_cost(self, data, medoids, labels):
        return sum(self.distance_func(data[i], medoids[labels[i]]) for i in range(data.shape[0]))

    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.apply(lambda x: x.astype(int) if x.dtype == bool else x)
        
        n_samples = data.shape[0]
        self.medoid_indices = self._initialize_medoids(n_samples)
        medoids = data.iloc[self.medoid_indices].values

        for iteration in tqdm(range(self.max_iterations), desc="K-Medoids Iterations"):
            old_labels = self.labels_.copy() if self.labels_ is not None else None
            self.labels_ = self._assign_labels(data.values, medoids)
            current_cost = self._compute_cost(data.values, medoids, self.labels_)

            best_cost = float('inf')
            best_medoids = medoids.copy()
            best_labels = self.labels_.copy()
            best_medoid_indices = self.medoid_indices.copy()

            for k_idx in tqdm(range(self.k), desc="Updating medoids", leave=False):
                for i in range(n_samples):
                    if i not in self.medoid_indices:
                        temp_medoids = medoids.copy()
                        temp_medoids[k_idx] = data.iloc[i].values
                        temp_labels = self._assign_labels(data.values, temp_medoids)
                        temp_cost = self._compute_cost(data.values, temp_medoids, temp_labels)

                        if temp_cost < best_cost:
                            best_cost = temp_cost
                            best_medoids = temp_medoids.copy()
                            best_labels = temp_labels.copy()
                            best_medoid_indices = self.medoid_indices.copy()
                            best_medoid_indices[k_idx] = i

            if best_cost < current_cost:
                medoids = best_medoids
                self.labels_ = best_labels
                self.medoid_indices = best_medoid_indices

            if old_labels is not None and np.array_equal(self.labels_, old_labels):
                break

        return self

    def predict(self, data):
        if self.medoid_indices is None:
            raise ValueError("Model not trained. Call `fit` first.")
        medoids = data.iloc[self.medoid_indices].values
        return self._assign_labels(data.values, medoids)