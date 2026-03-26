import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "polygon_pipeline"))

import pandas as pd
import logging
logging.basicConfig(level=logging.DEBUG)

from pipeline.storage import read_panels
from features.resampler import Resampler

print("=" * 60)
print("DETAILED RESAMPLE DEBUGGING")
print("=" * 60)

panels_1m = read_panels("polygon_pipeline/data/clean/", universe_only=False)
print(f"\n1-min close panel shape: {panels_1m['close'].shape}")
print(f"1-min close index: {panels_1m['close'].index.min()} → {panels_1m['close'].index.max()}")
print(f"1-min close index type: {type(panels_1m['close'].index)}")
print(f"1-min close NaN count: {panels_1m['close'].isna().sum().sum()}")

# Try manual resampling
close = panels_1m["close"]
print(f"\nManual 15-min resample test:")
try:
    resampled = close.resample("15min", label="left", closed="left").first()
    print(f"  Resampled shape: {resampled.shape}")
    print(f"  Resampled index: {resampled.index.min()} → {resampled.index.max()}")
except Exception as e:
    print(f"  ERROR: {e}")

# Try the Resampler with session_only=False
print(f"\nResampler with session_only=False:")
try:
    r = Resampler(panels_1m, freq="15min", session_only=False, min_bars_pct=0.1)
    result = r.resample()
    print(f"  Result shape: {result['close'].shape}")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

# Try with even lower coverage requirement
print(f"\nResampler with min_bars_pct=0.01:")
try:
    r = Resampler(panels_1m, freq="15min", session_only=False, min__bars_pct=0.01)
    result = r.resample()
    print(f"  Result shape: {result['close'].shape}")
except Exception as e:
    print(f"  ERROR: {e}")
