import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "polygon_pipeline"))

from pipeline.storage import read_panels
from features.store import FeatureStore
import pandas as pd

print("=" * 60)
print("DEBUG: Data shape alignment check")
print("=" * 60)

# Load features
print("\n[1] Features:")
store = FeatureStore("data/features/")
features = store.load(tickers=None, start_date=None, end_date=None)
if features:
    first_feat = list(features.keys())[0]
    feat_df = features[first_feat]
    print(f"  Loaded {len(features)} features")
    print(f"  First feature '{first_feat}' shape: {feat_df.shape}")
    print(f"  First feature index: {feat_df.index.min()} → {feat_df.index.max()}")
    print(f"  First feature columns (tickers): {feat_df.columns.tolist()}")

# Load clean data
print("\n[2] Clean data panels:")
panels = read_panels("polygon_pipeline/data/clean/", tickers=None, 
                     start_date=None, end_date=None, universe_only=False)
close = panels["close"]
print(f"  Close shape: {close.shape}")
print(f"  Close index: {close.index.min()} → {close.index.max()}")
print(f"  Close columns (tickers): {close.columns.tolist()[:5]}... ({len(close.columns)} total)")

# Check forward returns
print("\n[3] Forward returns computation:")
fwd_ret = close.pct_change(1).shift(-1)
print(f"  fwd_ret shape: {fwd_ret.shape}")
print(f"  fwd_ret non-NaN count: {fwd_ret.notna().sum().sum()}")
print(f"  fwd_ret all-NaN rows: {fwd_ret.isna().all(axis=1).sum()}")
print(f"  fwd_ret values sample (first 5 rows, first 3 cols):")
print(fwd_ret.iloc[:5, :3])

# Check if indices match
print("\n[4] Index alignment:")
print(f"  Features index type: {type(feat_df.index)}")
print(f"  Close index type: {type(close.index)}")
print(f"  Are indices equal? {feat_df.index.equals(close.index)}")
print(f"  Index overlap:")
common_idx = feat_df.index.intersection(close.index)
print(f"    Common timestamps: {len(common_idx)} / {len(feat_df.index)} (feature)")
