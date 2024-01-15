import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000):
        self.weights = np.zeros(input_size + 1)  # Initialize weights and bias (extra weight for bias)
        self.learning_rate = learning_rate
        self.epochs = epochs

    # Step function that outputs 0 or 1 based on the input
    def step_function(self, x):
        return 1 if x >= 0 else 0

    # Train the perceptron
    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                output = self.predict(xi)
                # Update weights and bias using the perceptron rule
                self.weights[1:] += self.learning_rate * (target - output) * xi
                self.weights[0] += self.learning_rate * (target - output)  # Bias update

    # Predict the output for a given input
    def predict(self, X):
        linear_output = np.dot(X, self.weights[1:]) + self.weights[0]  # Weighted sum + bias
        return self.step_function(linear_output)

# Example usage with a simple AND gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input
y = np.array([0, 0, 0, 1])  # Output for AND gate

# Initialize and train the perceptron
perceptron = Perceptron(input_size=2)
perceptron.fit(X, y)

# Test the perceptron
for x in X:
    print(f"Input: {x}, Prediction: {perceptron.predict(x)}")