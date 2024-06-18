import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.metrics import accuracy_score

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the SVM model
svm = SVC()

# Define the hyperparameter space
param_space = {
    'C': Real(1e-6, 1e+6, prior='log-uniform'),  # Regularization parameter
    'gamma': Real(1e-6, 1e+1, prior='log-uniform'),  # Kernel coefficient
    'kernel': ['linear', 'rbf'],  # Types of kernel functions
}

# Perform Bayesian optimization using BayesSearchCV
opt = BayesSearchCV(svm, param_space, n_iter=50, cv=3, random_state=42, n_jobs=-1, verbose=1)

# Fit the model
opt.fit(X_train, y_train)

# Print the best parameters found by BayesSearchCV
print("Best hyperparameters found: ", opt.best_params_)

# Evaluate the model on the test set
y_pred = opt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.4f}".format(accuracy))