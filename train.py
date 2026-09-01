import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# Load training data
data = pd.read_csv("data/spendsmart_expense_dataset.csv")

X = data["title"]
y = data["category"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Create ML pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        MultinomialNB()
    )
])

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report")
print("---------------------")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print("----------------")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "models/category_classifier.pkl")

print("Model trained and saved successfully.")

test_expenses = [
    "I had pizza for lunch",
    "Uber to college",
    "Netflix monthly payment",
    "Netflix subscription",
    "Netflix premium subscription",
    "Paid my Netflix bill",
    "Bought a new shirt",
    "Paid electricity bill",
    "Monthly electricity payment",
    "Internet monthly payment",
    "Paid my rent",
    "Spotify monthly subscription",
    "Movie ticket with friends",
    "Bought medicine from pharmacy",
    "Paid college tuition",
    "Bought textbooks",
    "Petrol for my bike",
]

predictions = model.predict(test_expenses)
probabilities = model.predict_proba(test_expenses)

print("\nCustom Predictions")
print("------------------")

for expense, prediction, probability in zip(
    test_expenses,
    predictions,
    probabilities
):
    confidence = probability.max()

    print(
        f"{expense} -> "
        f"{prediction} "
        f"(confidence: {confidence:.2%})"
    )
for expense, prediction in zip(test_expenses, predictions):
    print(f"{expense} -> {prediction}")