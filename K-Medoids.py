class PAM:
    def __init__(self, n_clusters=4, max_iter=500, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.medoids_indices = None
        self.labels = None

    def _calculate_distance_matrix(self, X):
        n_samples = X.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = np.linalg.norm(X[i] - X[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        return distance_matrix

    def _initialize_medoids(self, n_samples):
        np.random.seed(self.random_state)
        return np.random.choice(n_samples, self.n_clusters, replace=False)

    def _assign_points_to_medoids(self, distance_matrix):
        labels = np.argmin(distance_matrix[:, self.medoids_indices], axis=1)
        cost = np.sum(np.min(distance_matrix[:, self.medoids_indices], axis=1))
        return labels, cost

    def _update_medoids(self, distance_matrix, labels):
        changed = False
        for i in range(self.n_clusters):
            cluster_points = np.where(labels == i)[0]
            if len(cluster_points) == 0:
                continue
            costs = np.sum(distance_matrix[np.ix_(cluster_points, cluster_points)], axis=1)
            new_medoid = cluster_points[np.argmin(costs)]
            if new_medoid != self.medoids_indices[i]:
                self.medoids_indices[i] = new_medoid
                changed = True
        return changed

    def fit(self, X):
        n_samples = X.shape[0]
        distance_matrix = self._calculate_distance_matrix(X)
        self.medoids_indices = self._initialize_medoids(n_samples)

        for _ in tqdm(range(self.max_iter), desc="PAM iterations"):
            self.labels, cost = self._assign_points_to_medoids(distance_matrix)
            if not self._update_medoids(distance_matrix, self.labels):
                print('break')
                break
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels

    def get_medoids(self, X):
        return X[self.medoids_indices]
