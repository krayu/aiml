import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#Simple 2D dataset
np.random.seed(42)
X = np.random.randn(100, 2) * [2, 1] + [5, 3]

#PCA
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)

#Reconstruct back to 2D for visualization
X_reconstructed = pca.inverse_transform(X_pca)

#Visualization
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label="Original Data", color="blue")
plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], alpha=0.8, label="Projected Data", color="red")
plt.plot(X_reconstructed[:, 0], X_reconstructed[:, 1], color="gray", linestyle="--", alpha=0.5)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("PCA Dimensionality Reduction")
plt.legend()
plt.show()
