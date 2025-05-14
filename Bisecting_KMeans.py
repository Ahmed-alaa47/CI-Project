from sklearn.cluster import KMeans
import numpy as np

def bisecting_kmeans(X, final_k=3, max_iter=100, num_trials=5, random_state=42):
  clusters = {0: X}
  cluster_assignments = {0: np.arange(len(X))} # Track indices of points in each cluster
  cluster_labels = np.zeros(len(X), dtype=int) # Final cluster labels
  
  current_k = 1
  
  while current_k < final_k:
    # Choose the cluster with highest SSE
    sse_per_cluster = {}
    for i in clusters:
      model = KMeans(n_clusters=1, random_state=random_state)
      model.fit(clusters[i])
      sse_per_cluster[i] = model.inertia_
    
    to_split = max(sse_per_cluster, key=sse_per_cluster.get)
    
    # Try multiple 2-means splits and choose the best one (lowest total SSE)
    best_sse = np.inf
    best_labels = None
    for _ in range(num_trials):
      model = KMeans(n_clusters=2, max_iter=max_iter, random_state=random_state)
      labels = model.fit_predict(clusters[to_split])
      sse = model.inertia_

      if sse < best_sse:
          best_sse = sse
          best_labels = labels

    # Split the cluster into two
    data_to_split = clusters[to_split]
    indices_to_split = cluster_assignments[to_split]

    new_cluster_id_1 = max(clusters) + 1
    new_cluster_id_2 = new_cluster_id_1 + 1

    mask1 = best_labels == 0
    mask2 = best_labels == 1

    clusters[new_cluster_id_1] = data_to_split[mask1]
    clusters[new_cluster_id_2] = data_to_split[mask2]

    cluster_assignments[new_cluster_id_1] = indices_to_split[mask1]
    cluster_assignments[new_cluster_id_2] = indices_to_split[mask2]

    # Remove the original cluster
    del clusters[to_split]
    del cluster_assignments[to_split]

    current_k += 1

  # Assign final labels
  for new_label, cluster_id in enumerate(cluster_assignments):
      cluster_labels[cluster_assignments[cluster_id]] = new_label

  return cluster_labels