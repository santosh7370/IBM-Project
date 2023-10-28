import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Example dataset (you would typically use a larger, real-world dataset)
data = {
    "message": [
        "Buy cheap watches!!!",
        "Congratulations, you've won a prize!",
        "Hello, can we schedule a meeting?",
        "Important business proposal",
        "Special offer for you"
    ],
    "label": [1, 1, 0, 0, 1]  # 1 for spam, 0 for not spam
}

df = pd.DataFrame(data)

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the spam classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predict on test data
y_pred = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

# Example input and output
input_message = "You've won a free iPhone!"
predicted_label = classifier.predict(vectorizer.transform([input_message]))[0]

print(f"Input Message: {input_message}")
print(f"Predicted Label: {'Spam' if predicted_label == 1 else 'Not Spam'}")
print(f"Model Accuracy: {accuracy}")
print("Confusion Matrix:")
print(confusion)
