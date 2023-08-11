import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset: 
texts = [
    "Win a free lottery now", 
    "Congratulations, you won a prize!", 
    "This is an urgent message, act now!", 
    "Hello, how are you today?", 
    "Meeting scheduled for tomorrow at 3 PM", 
    "Lunch with Sarah on Friday?", 
    "Reminder: Your bank statement is available",
    "Click here to claim your reward!",
    "Let's catch up soon", 
    "Update your payment information immediately"
]

# Corresponding labels: 1 = spam, 0 = non-spam
labels = [1, 1, 1, 0, 0, 0, 0, 1, 0, 1]

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Convert text data into numerical feature vectors using Bag-of-Words (CountVectorizer)
vectorizer = CountVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)  # Learn vocabulary and transform training data
X_test_transformed = vectorizer.transform(X_test)  # Transform test data using the same vocabulary

# Initialize and train the Naïve Bayes classifier (MultinomialNB for text classification)
model = MultinomialNB()
model.fit(X_train_transformed, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_transformed)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}\n")

# Display a detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Test with a new example
new_texts = ["You have won a cash prize!", "Let's meet for coffee"]
new_texts_transformed = vectorizer.transform(new_texts)
predictions = model.predict(new_texts_transformed)

# Print predictions for new examples
for text, pred in zip(new_texts, predictions):
    label = "Spam" if pred == 1 else "Not Spam"
    print(f"Message: '{text}' → Predicted: {label}")