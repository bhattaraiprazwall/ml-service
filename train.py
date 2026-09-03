import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------

# Load cleaned main dataset
main_data = pd.read_csv("data/expenses_clean.csv")

# Load additional manually curated examples
additional_data = pd.read_csv("data/additional_training.csv")

# Combine datasets
data = pd.concat(
    [main_data, additional_data],
    ignore_index=True
)

print("Total training dataset:", len(data))
X = data["title"]
y = data["category"]


# --------------------------------------------------
# 2. Train / Test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Dataset")
print("-------")
print(f"Total samples : {len(data)}")
print(f"Training      : {len(X_train)}")
print(f"Testing       : {len(X_test)}")


# --------------------------------------------------
# 3. Create ML pipeline
# --------------------------------------------------

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


# --------------------------------------------------
# 4. Train model
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# 5. Evaluate model
# --------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Evaluation")
print("----------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        y_test,
        predictions
    )
)


print("\nConfusion Matrix")
print("----------------")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# --------------------------------------------------
# 6. Save trained model
# --------------------------------------------------

joblib.dump(
    model,
    "models/category_classifier.pkl"
)

print("\nModel trained and saved successfully.")