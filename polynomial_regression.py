import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#Simple dataset
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1) 
y = np.array([2, 5, 10, 17, 26]) 

#Transform X to include polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

#Train Polynomial Regression model
model = LinearRegression()
model.fit(X_poly, y)

#Make predictions
y_pred = model.predict(X_poly)

# 📌 5️⃣ Visualization
plt.scatter(X, y, label="Actual data", color="blue")
plt.plot(X, y_pred, label="Polynomial Regression", color="red")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.title("Polynomial Regression")
plt.show()