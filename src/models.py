from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_models(random_state: int = 42) -> Dict[str, object]:
    """
    Returns a dictionary of candidate models.
    """
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            multi_class="auto",
            n_jobs=None,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            class_weight="balanced",
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=random_state
        ),
    }
    return models