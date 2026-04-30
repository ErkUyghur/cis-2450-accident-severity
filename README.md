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

## Dashboard Demo Link: 
https://www.loom.com/share/9315227506034fc3b8f427c4a463ebc7

---

## Difficulty Concepts

This project applies four difficulty concepts. The three graded concepts are marked ✅.

---

### ✅ Concept 1 — Feature Importance (§6.1)

**What:** Random Forest Mean Decrease in Impurity (MDI) importances, computed after tuning and visualized as a ranked bar chart.

**Justification:** Feature importances answer a question that accuracy metrics cannot: *which variables did the model actually rely on?* For a city-planner stakeholder, knowing that Hour of day and Temperature rank above road infrastructure features has direct operational implications — patrol scheduling and weather-alert systems are higher-leverage interventions than traffic-signal upgrades.

**Results:**
- **Hour of day** is the single most informative feature, confirming the EDA finding (§3.3) that severe-accident rates peak overnight (25–30%) and dip during rush hours (~20%)
- **Weather variables** (Temperature, Visibility, Humidity) collectively dominate the numerical block, validating hypothesis tests from §3.10 (Mann-Whitney p ≈ 0, Kruskal-Wallis p ≈ 0)
- **Road features** (Junction, Crossing, Traffic_Signal) contribute marginal signal, ranking below all weather/temporal variables
- **Wind_Chill is correctly absent** — excluded pre-modeling due to r=0.98 with Temperature (§3.5); its low importance would have confirmed the exclusion decision retroactively

**Reflected in conclusion (§8):** Feature importances directly inform the final deployment recommendation. RF is chosen over GBM partly because MDI importances give stakeholders a transparent, actionable feature ranking without needing additional tooling (e.g., SHAP).

---

### ✅ Concept 2 — Imbalanced Data: Stratified Downsampling (§3.2, §4 step 2)

**What:** Stratified downsampling of the training set to equalize class representation, with a before/after visualization of the class distribution.

**Justification:** The raw Kaggle dataset (~3 million records) is ~87% Severity 2. A model trained naively on this distribution learns to predict "Severity 2" for every accident — achieving 87% accuracy while flagging zero high-risk conditions. This is the exact failure mode the project aims to prevent. The before/after chart in §3.2 shows the original distribution on a log scale (Severity 2 dwarfs all other classes by more than an order of magnitude) alongside the balanced working dataset.

**How it was applied:**
1. `scripts/balance_dataset.py` constructed a stratified 269k working CSV from the raw 3 GB file (dataset construction, not a pipeline step)
2. In §4, after the 80/20 train/test split, each severity class in the **training set only** is downsampled to the minority count (~67k per class)
3. The test set is never touched by the balancing logic — its class proportions reflect the natural post-split distribution

**Results:** Severe recall in the best model (RF) reaches **0.75**. Without balancing, the model would have learned to ignore the Severe minority class entirely, degrading Severe recall toward 0 while accuracy stayed superficially high. The 72% macro F1 reported across balanced classes is a genuine discriminative achievement, not an artifact of class skew.

---

### ✅ Concept 3 — Hyperparameter Tuning: RandomizedSearchCV (§5.1, §6.1, §7.1)

**What:** `RandomizedSearchCV` with continuous probability distributions for GBM hyperparameter search; `GridSearchCV` used for LR and RF where the parameter spaces are smaller and discrete.

**Why RandomizedSearchCV is smarter than GridSearch for GBM:**

`GridSearchCV` enumerates every combination in a fixed discrete set. A grid of `learning_rate ∈ [0.05, 0.1, 0.2]` never evaluates `0.07`, `0.13`, or any value between grid points — and GBM's optimal learning rate is rarely at a round number. For three simultaneously-tuned parameters, a grid with $k$ values each requires $k^3$ evaluations; random search scales linearly with `n_iter` regardless of dimensionality.

**Search configuration (§7.1):**

| Parameter | Distribution | Range |
|---|---|---|
| `learning_rate` | `uniform(0.01, 0.29)` | [0.01 – 0.30) |
| `max_depth` | `randint(3, 12)` | 3 – 11 |
| `min_samples_leaf` | `randint(10, 60)` | 10 – 59 |

25 random combinations × 3-fold CV = 75 fits. This covers a substantially wider joint parameter region than a comparable 9-combination GridSearch (27 fits) while adding the ability to find non-grid-point optima.

**Results:** Best parameters and test-set performance reported in §7.1. The random search simultaneously tuned three interacting parameters — a joint search that would require a prohibitively large grid to enumerate at equivalent resolution.

---

### ➕ Concept 4 — Ensemble Models: RF + GBM (§6, §7)

**What:** Two ensemble methods with architecturally distinct strategies: Random Forest (bagging) and Gradient Boosting (sequential residual correction).

**Justification:** Each model is introduced to address a specific limitation of the prior model:
- LR → RF: LR's linear boundary cannot express compound feature interactions (e.g., "low visibility AND junction AND night"). RF's tree branching naturally captures these AND-logic rules
- RF → GBM: RF trains trees independently — no tree learns from others' mistakes. GBM trains sequentially, fitting each new tree to the residual errors of the current ensemble, directly targeting hard examples

**Why GBM did not outperform RF:** The marginal gap (~0.2 pp macro F1) reflects a dataset-level ceiling: the "hard examples" for GBM to correct are Severity 2 vs. 3 accidents with nearly identical feature distributions — ambiguity that is irreducible given our feature set. RF's variance reduction via bagging already handles the reducible noise; little remains for sequential correction to improve.

---

## Key Results

| Model | Macro F1 | Severe Recall | Notes |
|---|---|---|---|
| Logistic Regression (baseline) | 0.678 | 0.66 | Linear boundary; C=10 via GridSearchCV |
| Random Forest | 0.722 | 0.75 | max_depth=30 via GridSearchCV; **recommended for deployment** |
| Gradient Boosting | ~0.720 | ~0.72 | RandomizedSearchCV over 3 parameters |

**Recommended model:** Random Forest — highest Severe recall (0.75), interpretable feature importances, and competitive macro F1.
