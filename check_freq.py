import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "polygon_pipeline"))

from pipeline.storage import read_panels
import pandas as pd

panels_1m = read_panels("polygon_pipeline/data/clean/", universe_only=False)
close = panels_1m["close"]

print("Data frequency analysis:")
print(f"  Shape: {close.shape}")
print(f"  Index: {close.index.min()} → {close.index.max()}")
print(f"  Index freq: {close.index.freq}")
print(f"  Index inferred freq: {pd.infer_freq(close.index)}")

# Check time differences in first 100 timestamps
diffs = close.index[1:100].to_series().diff().dt.total_seconds() / 60
print(f"\n  Time differences (minutes) in first 100 timestamps:")
print(f"    Min: {diffs.min()}")
print(f"    Max: {diffs.max()}")
print(f"    Unique values: {diffs.unique()}")
print(f"    Count of non-1min diffs: {(diffs != 1).sum()}")

# Show actual sample indices
print(f"\n  First 20 index values:")
for i, idx in enumerate(close.index[:20]):
    print(f"    {i:2d}: {idx}")
