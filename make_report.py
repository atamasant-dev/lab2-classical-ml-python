"""Формує README.md (звіт) із файлів reports/*.csv та run_info.json, створених main.py.

Порядок:   python main.py   ->   python make_report.py
Так числа в звіті (особливо час навчання, затримка та пам'ять) завжди відповідають
саме тій машині, на якій ви запускали main.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REP = ROOT / "reports"

res = pd.read_csv(REP / "results.csv").set_index("Алгоритм")
prof = pd.read_csv(REP / "error_profile.csv")
imp = pd.read_csv(REP / "feature_importance.csv")
info = json.loads((REP / "run_info.json").read_text(encoding="utf-8"))

LR, RF = "Logistic Regression", "Random Forest Classifier"
lr, rf = res.loc[LR], res.loc[RF]


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# --- таблиці ---------------------------------------------------------------
main_tbl = pd.DataFrame({
    "Алгоритм": res.index,
    "Accuracy": res["Accuracy"].map("{:.4f}".format).values,
    "Precision": res["Precision"].map("{:.4f}".format).values,
    "Recall": res["Recall"].map("{:.4f}".format).values,
    "F1-Score": res["F1-Score"].map("{:.4f}".format).values,
    "ROC-AUC": res["ROC-AUC"].map("{:.4f}".format).values,
})
cost_tbl = pd.DataFrame({
    "Алгоритм": res.index,
    "Час навчання (мс)": res["Час навчання (мс)"].map("{:.1f}".format).values,
    "Затримка (мкс/запис)": res["Затримка (мкс/запис)"].map("{:.2f}".format).values,
    "Пропускна здатність (записів/с)": (1e6 / res["Затримка (мкс/запис)"]).map("{:,.0f}".format).str.replace(",", " ").values,
    "Пам'ять (КБ)": res["Пам'ять (КБ)"].map("{:.1f}".format).values,
})
cm_tbl = pd.DataFrame({
    "Алгоритм": res.index,
    "TN": res["TN"].values, "FP (хибні тривоги)": res["FP"].values,
    "FN (пропущені атаки)": res["FN"].values, "TP": res["TP"].values,
    "FPR": res["FPR"].map("{:.4f}".format).values,
})
cv_tbl = pd.DataFrame({
    "Алгоритм": res.index,
    "F1 (5-fold, середнє ± σ)": [f"{r['CV F1 (mean)']:.4f} ± {r['CV F1 (std)']:.4f}" for _, r in res.iterrows()],
    "ROC-AUC (5-fold, середнє ± σ)": [f"{r['CV ROC-AUC (mean)']:.4f} ± {r['CV ROC-AUC (std)']:.4f}" for _, r in res.iterrows()],
})
prof_tbls = {}
for name in (LR, RF):
    p = prof[prof["Алгоритм"] == name].drop(columns="Алгоритм").copy()
    for c in p.columns[2:]:
        p[c] = p[c].map("{:.1f}".format)
    prof_tbls[name] = md_table(p)

top = imp.sort_values("RF важливість", ascending=False).head(4)
top_txt = ", ".join(f"**{r['Ознака']}** ({r['RF важливість']:.2f})" for _, r in top.iterrows())
fp_cut = (1 - rf["FP"] / lr["FP"]) * 100
fn_cut = (1 - rf["FN"] / lr["FN"]) * 100
train_x = rf["Час навчання (мс)"] / lr["Час навчання (мс)"]
lat_x = rf["Затримка (мкс/запис)"] / lr["Затримка (мкс/запис)"]
mem_x = rf["Пам'ять (КБ)"] / lr["Пам'ять (КБ)"]
tp_rf = 1e6 / rf["Затримка (мкс/запис)"]
fp_lr = prof[(prof["Алгоритм"] == LR) & (prof["Група"] == "FP")].iloc[0]
fn_lr = prof[(prof["Алгоритм"] == LR) & (prof["Група"] == "FN")].iloc[0]
fn_rf = prof[(prof["Алгоритм"] == RF) & (prof["Група"] == "FN")].iloc[0]
tn_lr = prof[(prof["Алгоритм"] == LR) & (prof["Група"] == "TN")].iloc[0]

readme = f"""# Лабораторна робота №2
**Тема:** «Класичні алгоритми машинного навчання та регресійний/класифікаційний аналіз інженерних даних на мові Python»
**Варіант 2 (CS):** детекція аномалій та атак у мережевому трафіку (NIDS)
**Виконав:** Атамась Антон Валерійович, група: ______

---

## 1. Мета та постановка задачі

Формалізувати інженерну задачу виявлення шкідливого мережевого трафіку як задачу бінарної класифікації (Supervised Learning), реалізувати конвеєр предобробки, навчити та порівняти два алгоритми.

| Параметр | Значення |
| --- | --- |
| Вхідні ознаки X | `PacketSize`, `IAT`, `TCP_Flags`, `WindowSize`, `ActiveSessions` |
| Цільова змінна Y | 0 — Normal, 1 — Attack (PortScan / DDoS) |
| Алгоритми | Logistic Regression vs Random Forest Classifier |
| Тип задачі | Бінарна класифікація |
| Інженерна мета | Виявляти шкідливий мережевий трафік у реальному часі |

## 2. Дані

Реальний розмічений трафік недоступний, тому використано **синтетичний набір**, згенерований функцією `generate_synthetic_nids` (`src/data_loader.py`, `seed = {info['seed']}`) і збережений у `data/telemetry_dataset.csv`: **{info['n_samples']} записів, частка атак {info['attack_share_pct']}%** (приблизно порівну PortScan і DDoS).

| Ознака | Тип | Зміст | Нормальний трафік | PortScan | DDoS |
| --- | --- | --- | --- | --- | --- |
| `PacketSize` | числова, Б | розмір пакета | двомодальний: малі ACK (~90) та великі дані (~1100) | малі SYN-проби (~54) | малі SYN-flood (~60) та великі HTTP-flood |
| `IAT` | числова, мс | інтервал між пакетами | медіана ~25 | ~3 | ~0,15 |
| `TCP_Flags` | категоріальна | прапорці TCP | здебільшого ACK, PSH-ACK | SYN (70%), RST, FIN | SYN (60%), ACK, PSH-ACK |
| `WindowSize` | числова, Б | вікно TCP | типові 8–65 КБ | малі (1024–5840) | підроблені / випадкові |
| `ActiveSessions` | числова | активні сесії джерела | ~18 (інколи до ~120) | ~90 | ~1500 |

Щоб задача не була тривіальною, у генератор свідомо закладено «складні» випадки:
* **5% легітимних сплесків** (CDN, резервне копіювання): короткий IAT, багато сесій, багато SYN — схожі на атаку;
* **10% «стелс-атак»**, які імітують нормальну швидкість та кількість сесій;
* **0,5% хибних міток** (шум розмітки).

![Розвідувальний аналіз](reports/figures/01_eda.png)
*Рис. 1 — Розподіл ознак для Normal та Attack. Атаки відрізняються за IAT і ActiveSessions, але розподіли частково перекриваються.*

## 3. Предобробка (Scaling, Encoding)

Усі кроки зібрано в `ColumnTransformer` усередині `Pipeline`, тому масштабування «вчиться» лише на навчальній вибірці (без витоку даних із тесту):

| Ознаки | Перетворення | Чому |
| --- | --- | --- |
| `IAT`, `ActiveSessions` | `log(1 + x)` → `StandardScaler` | розподіли мають важкі хвости (від 0,01 до 2000 мс; від 1 до 10 000 сесій) |
| `PacketSize`, `WindowSize` | `StandardScaler` | вирівнювання масштабів для логістичної регресії |
| `TCP_Flags` | `OneHotEncoder` | прапорці — номінальна ознака без порядку |

Поділ на train/test — **80 % / 20 %** ({info['n_train']} / {info['n_test']} записів), **стратифікований** (однакова частка атак), `random_state = {info['seed']}`.

## 4. Моделі

| Модель | Гіперпараметри |
| --- | --- |
| Logistic Regression | `max_iter=1000`, `class_weight="balanced"` |
| Random Forest Classifier | `n_estimators=200`, `max_depth=12`, `min_samples_leaf=2`, `class_weight="balanced"` |

`class_weight="balanced"` використано, бо атак менше, ніж нормального трафіку, а **пропущена атака (FN) для NIDS дорожча за хибну тривогу (FP)**.

## 5. Методика оцінки

* **Якість:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, а також FPR (частка хибних тривог серед нормального трафіку) на тестовій вибірці. Для NIDS ключові Recall (скільки атак виявлено) і Precision/FPR (скільки хибних тривог отримає адміністратор).
* **Стійкість:** стратифікована 5-fold крос-валідація (F1, ROC-AUC) на всьому наборі.
* **Ресурси:** час навчання (медіана з 3 запусків), затримка передбачення (мкс на запис, мінімум із 20 повторів), піковий обсяг пам'яті під час навчання (`tracemalloc`). Час і пам'ять вимірюються окремими запусками, бо `tracemalloc` уповільнює код. `tracemalloc` бачить лише виділення пам'яті на рівні Python/NumPy, тому значення — оцінка, а не точний обсяг процесу.

## 6. Результати

Середовище виконання: Python {info['python']}, `{info['platform']}`.

**Таблиця 1 — Якість класифікації (тестова вибірка, {info['n_test']} записів)**

{md_table(main_tbl)}

**Таблиця 2 — Ресурси**

{md_table(cost_tbl)}

**Таблиця 3 — Матриці помилок (тестова вибірка)**

{md_table(cm_tbl)}

**Таблиця 4 — Крос-валідація (весь набір, 5 фолдів)**

{md_table(cv_tbl)}

![ROC-криві](reports/figures/02_roc.png)
*Рис. 2 — ROC-криві: Random Forest (AUC = {rf['ROC-AUC']:.4f}) стабільно вища за Logistic Regression (AUC = {lr['ROC-AUC']:.4f}).*

![Матриці помилок](reports/figures/03_confusion.png)
*Рис. 3 — Матриці помилок на тестовій вибірці.*

## 7. Аналіз помилок

![Розподіл ймовірностей](reports/figures/05_probabilities.png)
*Рис. 4 — Розподіл передбаченої ймовірності атаки. Зелені стовпці правіше порога 0,5 — хибні тривоги, червоні лівіше — пропущені атаки.*

Медіанні характеристики записів у кожній клітинці матриці помилок (`reports/error_profile.csv`):

*Logistic Regression*

{prof_tbls[LR]}

*Random Forest Classifier*

{prof_tbls[RF]}

* **Хибні тривоги логістичної регресії ({int(fp_lr['N'])} шт.)** — це легітимний трафік, що «поводиться як атака»: медіана ActiveSessions {fp_lr['медіана ActiveSessions']:.0f} проти {tn_lr['медіана ActiveSessions']:.0f} у правильно розпізнаного нормального трафіку, IAT {fp_lr['медіана IAT (мс)']:.1f} мс проти {tn_lr['медіана IAT (мс)']:.1f} мс, більше половини записів із прапорцем SYN. Логістична регресія проводить одну лінійну межу (в логарифмічному просторі), тому не може відокремити легітимні сплески від справжніх атак із поєднанням «багато сесій + малі пакети + SYN».
* **Random Forest** будує нелінійну межу з комбінацій порогів і лишає лише {int(res.loc[RF, 'FP'])} хибних тривог — це легітимні сплески, найбільш схожі на атаку.
* **Пропущені атаки (FN)** у обох моделей схожі: медіана IAT {min(fn_lr['медіана IAT (мс)'], fn_rf['медіана IAT (мс)']):.0f}–{max(fn_lr['медіана IAT (мс)'], fn_rf['медіана IAT (мс)']):.0f} мс і ActiveSessions {min(fn_lr['медіана ActiveSessions'], fn_rf['медіана ActiveSessions']):.0f}–{max(fn_lr['медіана ActiveSessions'], fn_rf['медіана ActiveSessions']):.0f} (проти IAT ≈ {tn_lr['медіана IAT (мс)']:.0f} мс і ActiveSessions ≈ {tn_lr['медіана ActiveSessions']:.0f} у розпізнаного нормального трафіку) — це «стелс-атаки», які маскуються під нормальну швидкість. Основні ознаки швидкості в них не відрізняються від норми, тому виявити їх можна лише за слабкими сигналами (прапорці, розмір, вікно).

## 8. Важливість ознак

![Важливість ознак](reports/figures/04_feature_importance.png)
*Рис. 5 — Коефіцієнти Logistic Regression та важливість ознак Random Forest.*

За Random Forest найважливіші: {top_txt}. Тобто виявлення спирається насамперед на інтенсивність (кількість сесій, IAT), а потім на «відбитки» сканерів/флудерів (розмір вікна, розмір пакета, прапорець SYN). У логістичній регресії найсильніший позитивний внесок дають `TCP_Flags_SYN` та `ActiveSessions`, найсильніший негативний — `TCP_Flags_SYN-ACK` (SYN-ACK — відповідь сервера; у згенерованих атаках цього прапорця немає — це особливість синтетичних даних, у реальному трафіку такої закономірності може не бути).

## 9. Висновки

1. **Random Forest суттєво точніший:** Accuracy {rf['Accuracy']*100:.2f}% проти {lr['Accuracy']*100:.2f}%, F1 {rf['F1-Score']:.4f} проти {lr['F1-Score']:.4f}, ROC-AUC {rf['ROC-AUC']:.4f} проти {lr['ROC-AUC']:.4f}. Перевага підтверджується крос-валідацією (F1 {res.loc[RF, 'CV F1 (mean)']:.3f} проти {res.loc[LR, 'CV F1 (mean)']:.3f}) і значно перевищує розкид між фолдами (σ ≈ {min(res.loc[RF, 'CV F1 (std)'], res.loc[LR, 'CV F1 (std)']):.3f}–{max(res.loc[RF, 'CV F1 (std)'], res.loc[LR, 'CV F1 (std)']):.3f}).
2. **Для NIDS головний виграш — менше хибних тривог:** FP зменшилось з {int(lr['FP'])} до {int(rf['FP'])} (−{fp_cut:.0f}%), FPR — з {lr['FPR']*100:.1f}% до {rf['FPR']*100:.1f}%, при цьому Recall вищий ({rf['Recall']:.3f} проти {lr['Recall']:.3f}), а кількість пропущених атак зменшилась на {fn_cut:.0f}% ({int(lr['FN'])} → {int(rf['FN'])}). Для оператора, який перевіряє кожну тривогу вручну, це різниця між придатною та непридатною системою.
3. **Ціна точності:** Random Forest ресурсомісткіший за Logistic Regression: навчання ≈ ×{train_x:.0f}, передбачення ≈ ×{lat_x:.0f}, пам'ять ≈ ×{mem_x:.1f}. Проте затримка {rf['Затримка (мкс/запис)']:.1f} мкс/запис означає ≈ {f'{tp_rf:,.0f}'.replace(',', ' ')} записів/с на одному ядрі — для аналізу потоків (flow records) у реальному часі цього достатньо.
4. **Рекомендація:** для задачі NIDS обрати Random Forest. Logistic Regression доцільна як швидкий інтерпретований базовий рівень або для вбудованих пристроїв із жорсткими обмеженнями ресурсів.
5. **Куди рухатися далі:** обидві моделі пропускають «стелс-атаки», що імітують нормальну швидкість. Для їх виявлення потрібні додаткові ознаки (ентропія адрес/портів, кількість унікальних портів за вікно часу, часові ряди), а також налаштування порога класифікації під бажаний баланс Recall/FPR.

## 10. Обмеження

* Дані **синтетичні**, створені автором; ступінь перекриття класів задано параметрами генератора. Абсолютні значення метрик не переносяться на реальний трафік, показовим є **відносне порівняння** моделей на однаковому наборі.
* Час навчання, затримка та пам'ять залежать від апаратного забезпечення; значення в таблиці 2 виміряні на машині, зазначеній у розділі 6.
* Гіперпараметри не підбиралися (використано розумні значення за замовчуванням); підбір (`GridSearchCV`) міг би трохи покращити обидві моделі.

## 11. Структура проєкту та запуск

```
lab2_classical_ml_python/
├── data/
│   └── telemetry_dataset.csv      # вхідні дані (синтетичний NIDS-трафік)
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # генерація, завантаження, предобробка (Scaling, Encoding)
│   ├── models.py                  # Logistic Regression та Random Forest
│   ├── metrics.py                 # Accuracy/Precision/Recall/F1/ROC-AUC, час, затримка, пам'ять
│   └── plots.py                   # графіки для звіту
├── reports/                       # результати запуску: CSV, JSON, figures/
├── main.py                        # запуск і порівняння алгоритмів (варіант 2)
├── make_report.py                 # формує цей README.md із результатів main.py
├── requirements.txt
└── README.md                      # звіт
```

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python main.py                 # навчання, метрики, графіки
python make_report.py          # оновити README.md числами з вашого запуску
python main.py --regenerate    # (необов'язково) перегенерувати датасет
```
"""

(ROOT / "README.md").write_text(readme, encoding="utf-8")
print("README.md сформовано.")
