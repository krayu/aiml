import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Simple dataset
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])  
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]) 

#Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

#Print accuracy
print(f"Model Accuracy: {accuracy * 100:.2f}%")

#Visualization
X_range = np.linspace(0, 11, 100).reshape(-1, 1)
y_prob = model.predict_proba(X_range)[:, 1] 

plt.scatter(X, y, label="Actual Data", color="blue")
plt.plot(X_range, y_prob, label="Logistic Regression Curve", color="red")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.title("Logistic Regression Example")
plt.show()