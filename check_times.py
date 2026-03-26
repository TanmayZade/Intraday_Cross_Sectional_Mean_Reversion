import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "polygon_pipeline"))

from pipeline.storage import read_panels

panels = read_panels("polygon_pipeline/data/clean/", universe_only=False)
close = panels["close"]

# Check time distribution
idx = close.index
times = idx.time

print("Time distribution in data:")
print(f"  Min time: {times.min()}")
print(f"  Max time: {times.max()}")
print(f"  Unique times: {len(set(times))}")

# Count how many bars are in regular hours (09:30-16:00)
from datetime import time
session_start = time(9, 30)
session_end = time(16, 0)
in_session = [(t >= session_start) and (t <= session_end) for t in times]
print(f"\nBars in session (09:30-16:00): {sum(in_session)} / {len(times)} ({100*sum(in_session)/len(times):.1f}%)")
print(f"Bars pre/post market: {len(times) - sum(in_session)}")

# Show time distribution
print(f"\nSample times:")
for h in [4, 5, 9, 10, 14, 15, 16, 19, 20]:
    count = sum(1 for t in times if t.hour == h)
    if count > 0:
        print(f"  {h:02d}:xx hours: {count} bars")
