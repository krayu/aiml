import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Simple dataset
np.random.seed(42)
X = np.vstack([
    np.random.randn(50, 2) + np.array([2, 2]),
    np.random.randn(50, 2) + np.array([6, 6]),
    np.random.randn(50, 2) + np.array([10, 2])
])

#K-Means clustering (3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

#Visualization
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", edgecolors="k", alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=200, label="Cluster Centers")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering")
plt.legend()
plt.show()
