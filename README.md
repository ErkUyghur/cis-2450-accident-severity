# Predicting Severity of Traffic Accidents in the US

CIS 2450 Final Project — Erk & Brian

## Setup

```bash
python3 -m venv venv
source venv/bin/activate     # macOS/Linux
# venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

Drop `accidentsData.csv` into `data/raw/` (gitignored — too large to commit).

Open `notebooks/accident_severity_prediction.ipynb` in VS Code or Jupyter.

## Structure

```
├── data/
│   ├── raw/              # original CSV (gitignored)
│   └── processed/        # cleaned outputs (gitignored)
├── models/               # saved sklearn pipelines (.joblib)
├── modals/               # Dash dashboard components
├── notebooks/
│   └── accident_severity_prediction.ipynb   # full pipeline: EDA → 3 models
├── scripts/
│   └── balance_dataset.py          # one-time stratified sampling from raw 3 GB file
├── dashboard.py
├── requirements.txt
└── README.md
```

## Dashboard Demo

[Watch the demo on Loom](https://www.loom.com/share/9315227506034fc3b8f427c4a463ebc7)


## Presentation Video

[Watch the presentation on YouTube] https://www.youtube.com/watch?v=U7Awz0E0qz8
---

## Difficulty Concepts

Four concepts applied; the three graded ones are marked ✅. Full justification and results are in the notebook sections linked below.

| # | Concept | Notebook Section | Key evidence |
|---|---|---|---|
| ✅ 1 | **Feature Importance** | §6.1 | RF MDI ranking — Hour and weather variables dominate; directly informs deployment recommendation in §8 |
| ✅ 2 | **Imbalanced Data** | §3.2, §4 step 2 | Downsampling to training set only after the split; Severe recall improved from ~0% baseline to 0.75 |
| ✅ 3 | **Hyperparameter Tuning** | §5.1, §6.1, §7.1 | `RandomizedSearchCV` over continuous distributions for GBM (finds between-grid-point optima); `GridSearchCV` for LR and RF |
| ➕ 4 | **Ensemble Models** | §6, §7 | RF (bagging) + GBM (sequential residual correction) — two architecturally distinct ensemble strategies, each justified against the prior model's specific limitation |

---

## Application of Course Topics

This project applies the following course topics:

1. **Polars:** Used for efficient loading of the large accident dataset.
2. **SQL / DuckDB:** Used for SQL-based EDA queries on accident severity patterns.
3. **Joins:** Accident data was joined with Census population data to evaluate whether state-level severity patterns were confounded by population.
4. **Hypothesis Testing:** Chi-square, Mann-Whitney U, and Kruskal-Wallis tests were used to validate EDA findings.
5. **Supervised Learning Models:** Logistic Regression, Random Forest, and Gradient Boosting were trained for binary classification.
6. **Different Hyperparameter Tuning Methods:** GridSearchCV and RandomizedSearchCV were used to tune model parameters.
7. **Time-Series / Temporal Features:** Hour and day-of-week were extracted from accident start time.

---

## Key Results

| Model | Macro F1 | Severe Recall | Notes |
|---|---|---|---|
| Logistic Regression (baseline) | 0.678 | 0.66 | Linear boundary; C=10 via GridSearchCV |
| Random Forest | 0.722 | 0.75 | max_depth=30 via GridSearchCV; **recommended for deployment** |
| Gradient Boosting | ~0.720 | ~0.72 | RandomizedSearchCV over 3 parameters |

**Recommended model:** Random Forest — highest Severe recall (0.75), interpretable feature importances, and competitive macro F1.
