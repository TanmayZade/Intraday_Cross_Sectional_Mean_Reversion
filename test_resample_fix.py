import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "polygon_pipeline"))

from pipeline.storage import read_panels
from features.resampler import Resampler

print("Testing with session_only=False...")
panels_1m = read_panels("polygon_pipeline/data/clean/", universe_only=False)
print(f"1-min panels: {panels_1m['close'].shape}")

resampler = Resampler(panels_1m, freq="15min", session_only=False)
panels = resampler.resample()
print(f"15-min panels: {panels['close'].shape}")
print(f"Success! Data available for alpha computation.")
