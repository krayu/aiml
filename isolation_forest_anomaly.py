import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

# Generate sample data (2D points)
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.60, random_state=0)
# Introduce some outliers (anomalies)
X_with_anomalies = np.vstack([X, np.random.uniform(low=-6, high=6, size=(30, 2))])

# Create Isolation Forest model
model = IsolationForest(contamination=0.1, random_state=42)  # contamination=0.1 means 10% anomalies

# Fit the model to the data with anomalies
model.fit(X_with_anomalies)

# Predict anomalies (1 for normal, -1 for anomaly)
predictions = model.predict(X_with_anomalies)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.scatter(X_with_anomalies[:, 0], X_with_anomalies[:, 1], c=predictions, cmap='coolwarm', edgecolors='k', s=50)
plt.title("Anomaly Detection with Isolation Forest")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()