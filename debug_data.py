import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "polygon_pipeline"))

from pipeline.storage import read_panels
from features.resampler import Resampler
import numpy as np

print("=" * 60)
print("DEBUG: Loading panels...")
print("=" * 60)

# Step 1: Load clean data
panels_1m = read_panels("polygon_pipeline/data/clean/", 
                        tickers=None,
                        start_date=None, 
                        end_date=None,
                        universe_only=False)

if not panels_1m:
    print("ERROR: No panels loaded!")
    sys.exit(1)

print(f"\nPanels loaded: {list(panels_1m.keys())}")
close = panels_1m["close"]
print(f"Close panel shape: {close.shape}")
print(f"Close panel index range: {close.index.min()} → {close.index.max()}")
print(f"Close panel columns: {close.columns.tolist()[:5]}... ({len(close.columns)} total)")

# Step 2: Resample to 15min
print("\n... Resampling to 15min ...")
resampler = Resampler(panels_1m, freq="15min", session_only=True)
panels = resampler.resample()
close_resampled = panels["close"]

print(f"Resampled close shape: {close_resampled.shape}")
if len(close_resampled) > 0:
    print(f"Resampled close index range: {close_resampled.index.min()} → {close_resampled.index.max()}")
else:
    print(f"Resampled close index range: EMPTY (NaT → NaT)")
print(f"Resampled close columns: {close_resampled.columns.tolist()[:5]}... ({len(close_resampled.columns)} total)")

# Check the _filter_session issue
print("\n... Debug _filter_session ...")
idx = close.index
print(f"Original index type: {type(idx)}")
print(f"Original index tz: {idx.tz}")
print(f"Index time sample: {idx[:5].time}")
t = idx.time
_SESSION_START = "09:30"
_SESSION_END = "16:00"
mask = (t >= pd.Timestamp(_SESSION_START).time()) & (t <= pd.Timestamp(_SESSION_END).time())
print(f"Session filter mask - True count: {mask.sum()}")
print(f"Session filter mask - False count: {(~mask).sum()}")
print(f"Time range in data: {t.min()} → {t.max()}")

# Step 3: Check forward returns
print("\n... Computing forward returns ...")
fwd_ret = close_resampled.pct_change(1).shift(-1)
print(f"Forward return shape: {fwd_ret.shape}")
print(f"Forward return non-null: {fwd_ret.notna().sum().sum()}")
print(f"Forward return all NaN rows: {fwd_ret.isna().all(axis=1).sum()}")

print(f"\nFwd ret sample values:")
print(fwd_ret.iloc[:5, :5])

# Check if last row is the problem
print(f"\nLast row of close: {close_resampled.iloc[-1, :5].values}")
print(f"Last row of fwd_ret (should be NaN): {fwd_ret.iloc[-1, :5].values}")
