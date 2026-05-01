"""One-time script to create a balanced sample of the full US Accidents CSV.

The raw Kaggle dataset is ~87% Severity 2. This script constructs a stratified
working CSV by keeping all Severity 1 records and sampling an equal number from
each of Severities 2, 3, and 4. Run once; the output is committed to the repo
(gitignored) and used as the input to the notebook and dashboard.
"""

import polars as pl
from pathlib import Path

INPUT = Path("data/raw/US_Accidents_March23.csv")
OUTPUT = Path("data/raw/accidentsData_balanced.csv")
SEED = 42

print(f"Loading {INPUT}...")
df = pl.read_csv(INPUT, truncate_ragged_lines=True, ignore_errors=True)
print(f"Full dataset: {df.shape}")

print("\nSeverity counts in full data:")
print(df.group_by("Severity").len().sort("Severity"))

# Use the minority class (Severity 1) count as the target per-class sample size.
sev1 = df.filter(pl.col("Severity") == 1)
target_n = sev1.height
print(f"\nUsing severity 1 count as target: {target_n:,}")

samples = [sev1]
for sev in [2, 3, 4]:
    subset = df.filter(pl.col("Severity") == sev)
    sampled = subset.sample(n=target_n, seed=SEED)
    samples.append(sampled)
    print(f"  severity {sev}: sampled {target_n:,} rows")

balanced = pl.concat(samples).sample(fraction=1.0, seed=SEED)
print(f"\nBalanced shape: {balanced.shape}")
print("Final distribution:")
print(balanced.group_by("Severity").len().sort("Severity"))

balanced.write_csv(OUTPUT)
print(f"\nWritten to {OUTPUT}")