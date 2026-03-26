# Features Pipeline Logging Guide

## Overview

The features pipeline now integrates comprehensive logging using Python's standard `logging` library. Logs are stored in files while also being displayed to the console.

## Log File Location

By default, logs are stored in the `logs/` directory with timestamped filenames:
```
logs/
├── features_pipeline_20260321_143022.log
├── features_pipeline_20260321_150500.log
└── ...
```

## Configuration

### Basic Usage

Run the pipeline with default logging:
```bash
python run_features.py
```

Logs will be written to `logs/features_pipeline_YYYYMMDD_HHMMSS.log`

### Custom Log Directory

Specify a custom directory for logs:
```bash
python run_features.py --log-dir my_logs
```

### Log Level Control

Control verbosity with `--log-level`:
```bash
python run_features.py --log-level DEBUG    # Highly detailed
python run_features.py --log-level INFO     # Standard (default)
python run_features.py --log-level WARNING  # Only warnings & errors
python run_features.py --log-level ERROR    # Only errors
```

## What Gets Logged

### Pipeline Initialization (run_features.py)
- Pipeline configuration and parameters
- Data loading progress and row counts
- Resampling operations
- Feature computation stages (A→E categories)
- Storage operations with file sizes

### Feature Engine (engine.py)
- Engine initialization with data dimensions and date ranges
- Feature category progress (Reversal, VWAP, Volume, etc.)
- Computation completion with data quality metrics

### Feature Store (store.py)
- Store initialization
- Save operations with total sizes
- Load operations with filtering details
- Storage info queries

### Diagnostics (diagnostics.py)
- IC (Information Coefficient) computation progress
- Signal quality warnings:
  - Low-std signals (potential dead signals)
  - High bar-to-bar turnover (noisy signals)
  - Redundant feature pairs (correlation > 0.7)
- Best feature identification
- Report generation and file saving

## Log Format

```
2026-03-21 14:30:22  features.run_features     INFO      ────────────────────────────────
2026-03-21 14:30:22  features.run_features     INFO        Feature Engineering Pipeline
2026-03-21 14:30:22  features.run_features     INFO        Frequency    : 15min
```

Format: `TIMESTAMP  MODULE_NAME  LEVEL  MESSAGE`

## Example Output

```
2026-03-21 14:30:22  features.store            INFO      FeatureStore initialized at: data/features/
2026-03-21 14:30:23  features.run_features     INFO      [1/5] Loading 1-min clean panels ...
2026-03-21 14:30:25  features.engine           INFO      FeatureEngine: 10140 bars × 500 tickers | 2026-01-02 09:30:00 → 2026-03-20 16:00:00
2026-03-21 14:30:25  features.run_features     INFO      [2/5] Resampling 1-min → 15min ...
2026-03-21 14:30:30  features.run_features     INFO      [3/5] Computing 15 alpha features ...
2026-03-21 14:30:30  features.engine           INFO        [A] Reversal signals ...
2026-03-21 14:30:32  features.engine           INFO        [B] VWAP & microstructure ...
2026-03-21 14:30:34  features.engine           INFO        [C] Volume & flow ...
2026-03-21 14:30:36  features.engine           INFO        [D] Volatility regime ...
2026-03-21 14:30:37  features.engine           INFO        [E] Residual / factor ...
2026-03-21 14:30:38  features.engine           INFO      Features complete: 15 signals | avg valid=88.2%
2026-03-21 14:30:38  features.run_features     INFO      [4/5] Saving 15 features to data/features/ ...
2026-03-21 14:30:39  features.store            INFO      Saving 15 features to data/features/ with compression=snappy
2026-03-21 14:30:42  features.store            INFO      ✓ Saved 15 features | Total: 125.34 MB
2026-03-21 14:30:42  features.run_features     INFO      [5/5] Running signal quality diagnostics ...
2026-03-21 14:30:42  features.diagnostics      INFO      Running feature diagnostics ...
2026-03-21 14:30:43  features.diagnostics      INFO      ✓ Best IC feature: A2_short_rev (IC=0.0156, ICIR=2.340)
2026-03-21 14:30:43  features.diagnostics      INFO      Computing IC decay for: A2_short_rev
2026-03-21 14:30:43  features.diagnostics      INFO      Computing signal statistics...
2026-03-21 14:30:44  features.diagnostics      WARNING   ⚠ Low-std signals (potential dead signals): B3_open_gap, E2_sector_residual
2026-03-21 14:30:44  features.diagnostics      INFO      Computing feature correlation matrix...
2026-03-21 14:30:45  features.diagnostics      INFO      Saving diagnostic reports to polygon_pipeline/reports/
2026-03-21 14:30:45  features.diagnostics      INFO        ✓ feature_ic_summary.csv (3.2 KB, 15 rows)
2026-03-21 14:30:45  features.diagnostics      INFO        ✓ feature_signal_stats.csv (2.1 KB, 15 rows)
```

## Accessing Logs

### Recent logs
```bash
ls -lt logs/ | head -5
```

### View latest log
```bash
cat logs/$(ls -t logs/ | head -1)
```

### Search logs for specific features
```bash
grep "A1_bar_reversal" logs/*.log
```

### Monitor in real-time
```bash
tail -f logs/features_pipeline_*.log
```

## Debug Mode

For detailed debugging information:
```bash
python run_features.py --log-level DEBUG --log-dir debug_logs/
```

This will log:
- Detailed per-feature computation info
- Data shape information at each step
- File-level storage details
- Year partition information

## Integration with Your Code

If you need to add logging to your own code:

```python
import logging
log = logging.getLogger(__name__)

# The logger is automatically configured by setup_logging()
log.info("My message: %s", value)
log.warning("Warning: %s", value)
log.error("Error: %s", value)
log.debug("Debug info: %s", value)
```

## Log File Retention

Log files accumulate over time. To clean old logs:
```bash
find logs/ -name "*.log" -mtime +7 -delete  # Delete logs older than 7 days
```

## Troubleshooting

### Logs not appearing in file
- Check that `logs/` directory has write permissions
- Verify `--log-dir` path is correct
- Check console output for errors

### Too much/too little output
- Adjust `--log-level` (DEBUG for more, ERROR for less)
- Check if output is being buffered (add `-u` flag when running Python)

### Logs mixing with other output
- Logs use standard Python logger format
- Console and file use the same configuration
- Use `grep` or other tools to filter if needed
