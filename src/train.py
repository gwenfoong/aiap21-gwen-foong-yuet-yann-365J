from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from . import config
from .preprocess import prepare_features
from .models import get_models


def evaluate_model(name: str, y_true, y_pred) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
    print("Classification report:")
    print(classification_report(y_true, y_pred))
    return {"model": name, "accuracy": acc, "f1_macro": f1_macro}


def main():
    print("Loading and preprocessing data...")
    X, y = prepare_features(include_metaloxide3=config.INCLUDE_METALOXIDE3)
    print(f"Dataset after preprocessing: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    models = get_models(random_state=config.RANDOM_STATE)

    results = []
    for name, model in models.items():
        print(f"\nTraining model: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(name, y_test, y_pred)
        results.append(metrics)

    # Optionally store results in a CSV
    results_df = pd.DataFrame(results)
    print("\nSummary of model performance:")
    print(results_df)


if __name__ == "__main__":
    main()