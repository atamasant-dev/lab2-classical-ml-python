"""Лабораторна робота №2. Варіант 2 (CS): детекція аномалій та атак у мережевому трафіку (NIDS).

Порівняння: Logistic Regression vs Random Forest Classifier.
Запуск:  python main.py            (використає data/telemetry_dataset.csv; якщо файлу немає — згенерує)
         python main.py --regenerate   (примусово перегенерувати датасет)
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import pandas as pd

from src.data_loader import (DEFAULT_CSV, FEATURES, TARGET, generate_synthetic_nids, load_dataset,
                             save_dataset, split_data)
from src.metrics import (classification_metrics, cross_validation_summary, error_profile,
                         measure_inference_latency, measure_training)
from src.models import get_models
from src.plots import (plot_confusion, plot_eda, plot_feature_importance, plot_probability_hist, plot_roc)

SEED = 42
ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "reports" / "figures"
RES_DIR = ROOT / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(description="Лаб. №2, варіант 2: NIDS (Logistic Regression vs Random Forest)")
    parser.add_argument("--regenerate", action="store_true", help="перегенерувати data/telemetry_dataset.csv")
    parser.add_argument("--samples", type=int, default=6000, help="кількість записів при генерації")
    args = parser.parse_args()

    print("=" * 90)
    print("  ЛАБОРАТОРНА РОБОТА №2: КЛАСИЧНЕ МАШИННЕ НАВЧАННЯ (PYTHON 3.10+)")
    print("  ВАРІАНТ 2: Детекція аномалій та атак у мережевому трафіку (NIDS)")
    print("=" * 90 + "\n")

    # 1. Дані
    if args.regenerate or not DEFAULT_CSV.exists():
        df = generate_synthetic_nids(n_samples=args.samples, seed=SEED)
        save_dataset(df)
        print(f"[ДАНІ] Згенеровано та збережено {len(df)} записів -> {DEFAULT_CSV.relative_to(ROOT)}")
    df = load_dataset()
    share = df[TARGET].mean() * 100
    print(f"[ДАНІ] Записів: {len(df)}, ознак: {len(FEATURES)}, частка атак: {share:.1f}%\n")

    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, seed=SEED)
    print(f"[ПОДІЛ] train = {len(X_train)}, test = {len(X_test)} (стратифікований, random_state={SEED})\n")

    # 2. Навчання та оцінка
    models = get_models(SEED)
    fitted, outputs, rows = {}, {}, []
    for name, model in models.items():
        print(f"[НАВЧАННЯ] {name} ...")
        m, train_ms, peak_kb = measure_training(model, X_train, y_train)
        latency_us = measure_inference_latency(m, X_test)
        proba = m.predict_proba(X_test)[:, 1]
        pred = m.predict(X_test)
        met = classification_metrics(y_test, pred, proba)
        cv = cross_validation_summary(model, df[FEATURES], df[TARGET], folds=5, seed=SEED)
        fitted[name] = m
        outputs[name] = {"proba": proba, "pred": pred, "metrics": met}
        rows.append({"Алгоритм": name, **met, "Час навчання (мс)": train_ms, "Затримка (мкс/запис)": latency_us,
                     "Пам'ять (КБ)": peak_kb, **cv})

    res = pd.DataFrame(rows)
    show = pd.DataFrame({
        "Алгоритм": res["Алгоритм"],
        "Accuracy": res["Accuracy"].map("{:.4f}".format),
        "Precision": res["Precision"].map("{:.4f}".format),
        "Recall": res["Recall"].map("{:.4f}".format),
        "F1-Score": res["F1-Score"].map("{:.4f}".format),
        "ROC-AUC": res["ROC-AUC"].map("{:.4f}".format),
        "Час навчання (мс)": res["Час навчання (мс)"].map("{:.1f}".format),
        "Затримка (мкс/запис)": res["Затримка (мкс/запис)"].map("{:.2f}".format),
        "Пам'ять (КБ)": res["Пам'ять (КБ)"].map("{:.1f}".format),
    })
    print("\n" + show.to_string(index=False))
    print("\nМатриці помилок (TN, FP, FN, TP) та FPR:")
    print(res[["Алгоритм", "TN", "FP", "FN", "TP"]].assign(FPR=res["FPR"].map("{:.4f}".format)).to_string(index=False))
    print("\n5-fold крос-валідація (середнє ± σ):")
    for _, r in res.iterrows():
        print(f"  {r['Алгоритм']:<26} F1 = {r['CV F1 (mean)']:.4f} ± {r['CV F1 (std)']:.4f}   "
              f"ROC-AUC = {r['CV ROC-AUC (mean)']:.4f} ± {r['CV ROC-AUC (std)']:.4f}")

    print("\nАналіз помилок (медіани характеристик у клітинках матриці помилок):")
    profiles = []
    for name, out in outputs.items():
        prof = error_profile(X_test, y_test, out["pred"])
        prof.insert(0, "Алгоритм", name)
        profiles.append(prof)
        print(f"\n  {name}")
        print(prof.drop(columns="Алгоритм").round(2).to_string(index=False))

    # 3. Збереження результатів та графіків
    RES_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RES_DIR / "results.csv", index=False)
    pd.concat(profiles).to_csv(RES_DIR / "error_profile.csv", index=False)
    meta = {"python": sys.version.split()[0], "platform": platform.platform(), "seed": SEED,
            "n_samples": len(df), "attack_share_pct": round(share, 2), "n_train": len(X_train), "n_test": len(X_test)}
    (RES_DIR / "run_info.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    names = [n.split("__", 1)[1] for n in fitted["Logistic Regression"].named_steps["prep"].get_feature_names_out()]
    pd.DataFrame({"Ознака": names,
                  "LR коефіцієнт": fitted["Logistic Regression"].named_steps["clf"].coef_[0],
                  "RF важливість": fitted["Random Forest Classifier"].named_steps["clf"].feature_importances_}
                 ).to_csv(RES_DIR / "feature_importance.csv", index=False)

    plot_eda(df, FIG_DIR / "01_eda.png")
    plot_roc(y_test, outputs, FIG_DIR / "02_roc.png")
    plot_confusion(y_test, outputs, FIG_DIR / "03_confusion.png")
    plot_feature_importance(fitted, FIG_DIR / "04_feature_importance.png")
    plot_probability_hist(y_test, outputs, FIG_DIR / "05_probabilities.png")
    print(f"\n[ЗБЕРЕЖЕНО] reports/results.csv, error_profile.csv, feature_importance.csv, run_info.json, 5 графіків у reports/figures/")
    print("\n[УСПІШНО] Моделювання та оцінка метрик завершені.")


if __name__ == "__main__":
    main()
