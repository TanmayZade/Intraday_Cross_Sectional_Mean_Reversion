#!/usr/bin/env python3
"""Fix corruption in run_features.py"""

filepath = r"D:\Herbs magic\Int_Cross_Section_MR\NEW_APPROACH\run_features.py"

with open(filepath, 'r') as f:
    lines = f.readlines()

# Fix 1: Lines 158-159 (indices 157-158) - remove corrupted lines
# Remove the " universe_only=False," and ")" lines
if lines[157].strip().startswith('universe_only') and lines[158].strip() == ')':
    # Remove these lines and add blank line after panels_target = panels_5m
    lines = lines[:157] + ['\n'] + lines[159:]

# Fix 2: Line 160 (which is now 159 after removal) - change panels to panels_target  
# Change "close     = panels["close"]" to "close = panels_target["close"]"
for i, line in enumerate(lines):
    if 'close     = panels["close"]' in line:
        lines[i] = line.replace('close     = panels["close"]', 'close = panels_target["close"]')
        print(f"Fixed line {i+1}: close variable")
        break

# Fix 3: Change "Resampled:" to "Ready:" in log message
for i, line in enumerate(lines):
    if 'log.info("  Resampled: %d %s bars × %d tickers",' in line:
        lines[i] = line.replace('log.info("  Resampled: %d %s bars × %d tickers",', 'log.info("  Ready: %d %s bars × %d tickers",')
        print(f"Fixed line {i+1}: log message")
        break

# Fix 4: Line 167 (approx) - change panels to panels_target in FeatureEngine call
for i, line in enumerate(lines):
    if line.strip() == 'panels,' and 'FeatureEngine' in ''.join(lines[max(0, i-5):i]):
        lines[i] = line.replace('        panels,', '        panels_target,')
        print(f"Fixed line {i+1}: FeatureEngine panels argument")
        break

# Fix 5: Line 202 (approx) - change return statement
for i, line in enumerate(lines):
    if 'return {"features": features, "panels": panels}' in line:
        lines[i] = line.replace('return {"features": features, "panels": panels}', 'return {"features": features, "panels": panels_target}')
        print(f"Fixed line {i+1}: return statement")
        break

with open(filepath, 'w') as f:
    f.writelines(lines)

print(f"Fixed file: {filepath}")
