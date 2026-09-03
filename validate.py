import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Load trained model
# --------------------------------------------------

MODEL_PATH = "models/category_classifier.pkl"
VALIDATION_PATH = "data/validation.csv"

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# 2. Load unseen validation data
# --------------------------------------------------

data = pd.read_csv(VALIDATION_PATH)

X = data["title"]
y = data["category"]

# --------------------------------------------------
# 3. Make predictions
# --------------------------------------------------

predictions = model.predict(X)

# --------------------------------------------------
# 4. Evaluate
# --------------------------------------------------

accuracy = accuracy_score(y, predictions)

print("\nValidation Evaluation")
print("---------------------")
print(f"Validation samples: {len(data)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        y,
        predictions,
        zero_division=0
    )
)

print("\nConfusion Matrix")
print("----------------")

print(
    confusion_matrix(
        y,
        predictions
    )
)

# --------------------------------------------------
# 5. Detailed predictions
# --------------------------------------------------

print("\nDetailed Predictions")
print("--------------------")

for title, actual, predicted in zip(
    X,
    y,
    predictions
):
    status = "✓" if actual == predicted else "✗"

    print(
        f"{status} "
        f"{title} | "
        f"Actual: {actual} | "
        f"Predicted: {predicted}"
    )