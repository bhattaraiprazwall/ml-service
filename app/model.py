import joblib

MODEL_PATH = "models/category_classifier.pkl"

model = joblib.load(MODEL_PATH)


def predict_category(title: str):
    prediction = model.predict([title])[0]

    probabilities = model.predict_proba([title])[0]

    classes = model.classes_

    probability_map = {
        category: float(probability)
        for category, probability in zip(classes, probabilities)
    }

    confidence = max(probability_map.values())

    return {
        "category": prediction,
        "confidence": confidence,
        "probabilities": probability_map,
    }