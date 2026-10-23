"""Моделі для порівняння (варіант 2): Logistic Regression vs Random Forest Classifier."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_loader import build_preprocessor


def build_logistic_regression(seed: int = 42) -> Pipeline:
    # class_weight="balanced": атаки становлять меншість, а пропуск атаки (FN) дорожчий за хибну тривогу
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def build_random_forest(seed: int = 42) -> Pipeline:
    clf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=2,
                                 class_weight="balanced", n_jobs=1, random_state=seed)
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def get_models(seed: int = 42) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": build_logistic_regression(seed),
        "Random Forest Classifier": build_random_forest(seed),
    }
