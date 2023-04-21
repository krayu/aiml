import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Simple dataset
X = np.array([[1, 2], [2, 3], [3, 1], [6, 7], [7, 8], [8, 6], [2, 1], [3, 2], [4, 3], [7, 6], [8, 7], [9, 8]])
y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])

#Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train SVM classifier with RBF kernel
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train, y_train)

#Predictions
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

#Visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="coolwarm", label="Data points")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Support Vector Machine")
plt.legend()
plt.show()
