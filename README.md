# Predicting Severity of Traffic Accidents in the US

CIS 2450 Final Project — Erk & Brian

## Setup

```bash
python3 -m venv venv
source venv/bin/activate     # macOS/Linux
# venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

Drop `accidentsData.csv` into `data/raw/` (it's gitignored — too big to commit).

Open `notebooks/01_eda_and_baseline.ipynb` in VS Code.

## Structure

```
├── data/
│   ├── raw/              # original CSV (gitignored)
│   └── processed/        # cleaned/feature-engineered (gitignored)
├── notebooks/
│   └── 01_eda_and_baseline.ipynb
├── reports/figures/      # exported plots
├── requirements.txt
└── README.md
```
