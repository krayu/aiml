from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TPOT with desired generations and population size
tpot = TPOTClassifier( generations=5, population_size=20, random_state=42, verbosity=2)

# Fit TPOT to the training data
tpot.fit(X_train, y_train)

# Evaluate the model on test data
y_pred = tpot.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))

# Export the pipeline found by TPOT
tpot.export('best_model_pipeline.py')