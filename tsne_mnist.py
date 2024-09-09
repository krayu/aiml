import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load MNIST dataset
from sklearn.datasets import fetch_openml

# Fetch MNIST dataset (it comes with labels)
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

# Preprocess the data: standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dimensionality using PCA (initial reduction to 50 dimensions)
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

# Apply t-SNE to further reduce the dimensionality to 2D
tsne = TSNE(n_components=2, random_state=42, n_jobs=1)
X_tsne = tsne.fit_transform(X_pca)

# Visualize the 2D t-SNE output
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y.astype(int), cmap='tab10', s=5)
plt.colorbar()
plt.title("t-SNE visualization of MNIST data")
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.show()