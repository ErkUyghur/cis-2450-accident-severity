"""One-time script to create a balanced sample of the full US Accidents CSV
because there was a heavy imbalance in severity levels.
"""

import polars as pl
from pathlib import Path 
"""just for nicer path handling, since brian has windows and i have mac, 
but we can both use pathlib to make it work on both OSes without hardcoding slashes."""

INPUT = Path("data/raw/US_Accidents_March23.csv")   
OUTPUT = Path("data/raw/accidentsData_balanced.csv")
SEED = 42

print(f"Loading {INPUT}...")
df = pl.read_csv(INPUT, truncate_ragged_lines=True, ignore_errors=True)
#loading data here
print(f"Full dataset: {df.shape}")

# how many of each severity do we have?
print("\nSeverity counts in full data:")
print(df.group_by("Severity").len().sort("Severity"))

# Pull all severity 1 and use that count as the target
sev1 = df.filter(pl.col("Severity") == 1)
target_n = sev1.height
print(f"\nUsing severity 1 count as target: {target_n:,}")

# Sample target_n rows from each of severities 2, 3, 4
samples = [sev1]
for sev in [2, 3, 4]:
    subset = df.filter(pl.col("Severity") == sev) #grab all rows of this severity
    n_available = subset.height #how many rows of this severity?
    sampled = subset.sample(n= target_n, seed=SEED) #random sampling
    samples.append(sampled)
    print(f"  severity {sev}: sampled {target_n:,} rows")

balanced = pl.concat(samples).sample(fraction=1.0, seed=SEED)  #stacks the samples together
print(f"\nBalanced shape: {balanced.shape}") #checking shape
print("Final distribution:")
print(balanced.group_by("Severity").len().sort("Severity"))

balanced.write_csv(OUTPUT)
print(f"\nWritten to {OUTPUT}")