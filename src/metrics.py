"""Метрики якості класифікації, час навчання, затримка інференсу та пам'ять."""
from __future__ import annotations

import time
import tracemalloc

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate


def measure_training(model, X_train, y_train):
    """Повертає (навчена модель, час навчання в мс, піковий обсяг пам'яті в КБ).

    Час (медіана з 3 запусків) і пам'ять вимірюються окремими запусками: tracemalloc суттєво уповільнює код,
    тому змішувати ці виміри в одному прогоні не можна.
    """
    times = []
    for _ in range(3):  # медіана з 3 запусків: перший запуск «холодний» і завжди повільніший
        t0 = time.perf_counter()
        fitted = clone(model).fit(X_train, y_train)
        times.append((time.perf_counter() - t0) * 1000.0)
    train_ms = float(np.median(times))

    tracemalloc.start()
    clone(model).fit(X_train, y_train)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return fitted, train_ms, peak / 1024.0


def measure_inference_latency(model, X, repeats: int = 20) -> float:
    """Середня затримка передбачення, мкс на один запис (мінімум із кількох повторів)."""
    model.predict(X)  # прогрів
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(X)
        best = min(best, time.perf_counter() - t0)
    return best / len(X) * 1e6


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "FPR": fp / (fp + tn),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def cross_validation_summary(model, X, y, folds: int = 5, seed: int = 42) -> dict:
    """k-fold крос-валідація (стратифікована): середнє та σ для F1 і ROC-AUC."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    res = cross_validate(clone(model), X, y, cv=cv, scoring=["f1", "roc_auc"])
    return {
        "CV F1 (mean)": float(np.mean(res["test_f1"])), "CV F1 (std)": float(np.std(res["test_f1"])),
        "CV ROC-AUC (mean)": float(np.mean(res["test_roc_auc"])), "CV ROC-AUC (std)": float(np.std(res["test_roc_auc"])),
    }


def error_profile(X_test, y_test, y_pred):
    """Медіанні характеристики записів у кожній клітинці матриці помилок (для аналізу помилок)."""
    import pandas as pd

    y = np.asarray(y_test); p = np.asarray(y_pred)
    groups = {"TN": (p == 0) & (y == 0), "FP": (p == 1) & (y == 0), "FN": (p == 0) & (y == 1), "TP": (p == 1) & (y == 1)}
    rows = []
    for name, mask in groups.items():
        d = X_test[mask]
        rows.append({"Група": name, "N": int(mask.sum()), "медіана IAT (мс)": d["IAT"].median(),
                     "медіана ActiveSessions": d["ActiveSessions"].median(), "медіана PacketSize (Б)": d["PacketSize"].median(),
                     "частка SYN, %": (d["TCP_Flags"] == "SYN").mean() * 100 if len(d) else float("nan")})
    return pd.DataFrame(rows)
