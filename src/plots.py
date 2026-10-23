"""Графіки для звіту (зберігаються у reports/figures)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без GUI: працює і в терміналі, і на сервері
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

from src.data_loader import TARGET

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
COLORS = {"Logistic Regression": "#1f77b4", "Random Forest Classifier": "#d62728"}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_eda(df, path: Path) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    normal, attack = df[df[TARGET] == 0], df[df[TARGET] == 1]

    bins = np.logspace(np.log10(0.01), np.log10(2000), 40)
    ax[0, 0].hist(normal["IAT"], bins=bins, alpha=.6, label="Normal", color="#2ca02c")
    ax[0, 0].hist(attack["IAT"], bins=bins, alpha=.6, label="Attack", color="#d62728")
    ax[0, 0].set_xscale("log"); ax[0, 0].set_xlabel("IAT, мс (лог. шкала)"); ax[0, 0].set_ylabel("Кількість"); ax[0, 0].legend()

    bins = np.logspace(0, np.log10(10000), 40)
    ax[0, 1].hist(normal["ActiveSessions"], bins=bins, alpha=.6, label="Normal", color="#2ca02c")
    ax[0, 1].hist(attack["ActiveSessions"], bins=bins, alpha=.6, label="Attack", color="#d62728")
    ax[0, 1].set_xscale("log"); ax[0, 1].set_xlabel("ActiveSessions (лог. шкала)"); ax[0, 1].legend()

    bins = np.linspace(40, 1500, 40)
    ax[1, 0].hist(normal["PacketSize"], bins=bins, alpha=.6, label="Normal", color="#2ca02c")
    ax[1, 0].hist(attack["PacketSize"], bins=bins, alpha=.6, label="Attack", color="#d62728")
    ax[1, 0].set_xlabel("PacketSize, Б"); ax[1, 0].set_ylabel("Кількість"); ax[1, 0].legend()

    order = ["SYN", "SYN-ACK", "ACK", "PSH-ACK", "FIN-ACK", "RST"]
    x = np.arange(len(order)); w = 0.38
    pn = normal["TCP_Flags"].value_counts(normalize=True).reindex(order).fillna(0) * 100
    pa = attack["TCP_Flags"].value_counts(normalize=True).reindex(order).fillna(0) * 100
    ax[1, 1].bar(x - w / 2, pn, w, label="Normal", color="#2ca02c")
    ax[1, 1].bar(x + w / 2, pa, w, label="Attack", color="#d62728")
    ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels(order, fontsize=8); ax[1, 1].set_ylabel("Частка в класі, %")
    ax[1, 1].set_xlabel("TCP_Flags"); ax[1, 1].legend()
    fig.suptitle("Розвідувальний аналіз: Normal vs Attack", fontsize=12)
    _save(fig, path)


def plot_roc(y_test, outputs: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5))
    for name, out in outputs.items():
        fpr, tpr, _ = roc_curve(y_test, out["proba"])
        ax.plot(fpr, tpr, lw=2, color=COLORS[name], label=f"{name} (AUC = {out['metrics']['ROC-AUC']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Випадкове вгадування")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC-криві")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def plot_confusion(y_test, outputs: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(outputs), figsize=(4.4 * len(outputs), 4.2))
    for ax, (name, out) in zip(np.atleast_1d(axes), outputs.items()):
        cm = confusion_matrix(y_test, out["pred"])
        ax.imshow(cm, cmap="Blues"); ax.grid(False)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=14, color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Attack"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Attack"])
        ax.set_xlabel("Передбачений клас"); ax.set_ylabel("Справжній клас"); ax.set_title(name, fontsize=10)
    fig.suptitle("Матриці помилок (тестова вибірка)", fontsize=12)
    _save(fig, path)


def plot_feature_importance(models: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    # Logistic Regression: |коефіцієнт| при стандартизованих ознаках
    lr = models["Logistic Regression"]
    names = lr.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    coef = lr.named_steps["clf"].coef_[0]
    idx = np.argsort(np.abs(coef))
    axes[0].barh(np.array(names)[idx], coef[idx], color=["#d62728" if c > 0 else "#1f77b4" for c in coef[idx]])
    axes[0].set_title("Logistic Regression: коефіцієнти\n(червоні — підвищують ризик атаки)", fontsize=10)
    axes[0].set_xlabel("Коефіцієнт")
    rf = models["Random Forest Classifier"]
    imp = rf.named_steps["clf"].feature_importances_
    idx = np.argsort(imp)
    axes[1].barh(np.array(names)[idx], imp[idx], color="#d62728")
    axes[1].set_title("Random Forest: важливість ознак\n(зменшення нечистоти Gini)", fontsize=10)
    axes[1].set_xlabel("Важливість")
    _save(fig, path)


def plot_probability_hist(y_test, outputs: dict, path: Path) -> None:
    """Графік помилок: розподіл ймовірності «Attack» окремо для справжніх Normal та Attack."""
    fig, axes = plt.subplots(1, len(outputs), figsize=(5.4 * len(outputs), 4))
    y = np.asarray(y_test)
    for ax, (name, out) in zip(np.atleast_1d(axes), outputs.items()):
        p = out["proba"]
        bins = np.linspace(0, 1, 26)
        ax.hist(p[y == 0], bins=bins, alpha=.65, color="#2ca02c", label="Справжній Normal")
        ax.hist(p[y == 1], bins=bins, alpha=.65, color="#d62728", label="Справжній Attack")
        ax.axvline(0.5, color="k", ls="--", lw=1)
        ax.set_yscale("log"); ax.set_xlabel("Передбачена P(Attack)"); ax.set_ylabel("Кількість (лог. шкала)")
        ax.set_title(name, fontsize=10); ax.legend(fontsize=8)
    fig.suptitle("Розподіл передбачених ймовірностей; помилки — зелене праворуч / червоне ліворуч від порога 0,5", fontsize=10)
    _save(fig, path)
