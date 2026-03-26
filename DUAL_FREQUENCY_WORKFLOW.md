# Dual-Frequency Workflow Guide (Option A)

## Overview

Your pipeline now supports **parallel 1-minute and 5-minute data** storage and analysis:

```
polygon_pipeline/data/
├── clean_1min/          ← 1-minute bars (new)
│   ├── year=2024/
│   ├── year=2025/
│   └── year=2026/
│
└── clean_5min/          ← 5-minute bars (existing baseline)
    ├── year=2024/
    ├── year=2025/
    └── year=2026/
```

---

## Workflow: Fetch & Compare

### Step 1: Fetch 5-minute Data (Keep as Baseline)

```bash
# Fetch/update 5-min data using dedicated config
python -m polygon_pipeline.pipeline.orchestrator run \
  --config polygon_pipeline/configs/config_5min.yaml \
  --start 2024-03-22 \
  --end 2024-12-31
```

**Output**: `polygon_pipeline/data/clean_5min/year=*/data.parquet`

---

### Step 2: Fetch 1-minute Data (New)

```bash
# Fetch 1-min data using dedicated config
python -m polygon_pipeline.pipeline.orchestrator run \
  --config polygon_pipeline/configs/config_1min.yaml \
  --start 2024-03-22 \
  --end 2024-12-31
```

**Output**: `polygon_pipeline/data/clean_1min/year=*/data.parquet`

**Note**: This will take longer (~8-12 hours) due to 5× more data and API rate limits.

---

### Step 3: Compare IC (1-min vs 5-min)

```bash
# Run single-day comparison first (fast validation)
python compare_frequencies.py --start 2024-12-20 --end 2024-12-20

# Full comparison across all data
python compare_frequencies.py --save-report
```

**Expected Output**:
- IC should be **stable** (±5% difference)
- Signal half-lives should be similar
- Feature rankings preserved across frequencies

**Reports saved to**: `reports/frequency_comparison.csv`

---

## Workflow: Feature & Alpha Generation

### Option A: Run on 5-minute data (baseline)

```bash
# Features (5-min baseline, 60-bar IC window)
python run_features.py \
  --clean-dir polygon_pipeline/data/clean_5min/ \
  --freq 5min \
  --atr-window 60 \
  --vol-window 180 \
  --zscore-window 180 \
  --beta-window 1170 \
  --halflife 39

# Alpha (5-min baseline, 60-bar IC window)
python run_alpha.py \
  --clean-dir polygon_pipeline/data/clean_5min/ \
  --ic-window 60 \
  --vol-window 60 \
  --halflife 13
```

---

### Option B: Run on 1-minute data (new)

```bash
# Features (1-min, 300-bar IC window)
python run_features.py \
  --clean-dir polygon_pipeline/data/clean_1min/ \
  --freq 1min \
  --atr-window 300 \
  --vol-window 900 \
  --zscore-window 900 \
  --beta-window 5850 \
  --halflife 195

# Alpha (1-min, 300-bar IC window)
python run_alpha.py \
  --clean-dir polygon_pipeline/data/clean_1min/ \
  --ic-window 300 \
  --vol-window 900 \
  --halflife 195
```

---

### Option C: Run Both (Parallel Comparison)

```bash
# Terminal 1: Process 5-min
python run_features.py --clean-dir polygon_pipeline/data/clean_5min/
python run_alpha.py --clean-dir polygon_pipeline/data/clean_5min/

# Terminal 2: Process 1-min
python run_features.py --clean-dir polygon_pipeline/data/clean_1min/
python run_alpha.py --clean-dir polygon_pipeline/data/clean_1min/

# Then compare results
python compare_frequencies.py
```

---

## Configuration Files

### `polygon_pipeline/configs/config_5min.yaml`
```yaml
polygon:
  multiplier: 5
storage:
  raw_dir: polygon_pipeline/data/raw_5min/
  clean_dir: polygon_pipeline/data/clean_5min/
logging:
  file: polygon_pipeline/reports/pipeline_5min.log
```

### `polygon_pipeline/configs/config_1min.yaml`
```yaml
polygon:
  multiplier: 1
storage:
  raw_dir: polygon_pipeline/data/raw_1min/
  clean_dir: polygon_pipeline/data/clean_1min/
logging:
  file: polygon_pipeline/reports/pipeline_1min.log
```

---

## API Quota Implications

| Task | 5-min | 1-min | Combined |
|------|-------|-------|----------|
| API calls/month | ~55 | ~220 | ~275 |
| Storage/2yr (44 tickers) | ~150-200 MB | ~600-800 MB | ~800 MB - 1 GB |
| Feature compute time | ~1-2 min | ~10-15 min | — |
| Alpha compute time | ~10-20 sec | ~30-60 sec | — |

**Polygon Free Tier**: 
- Check your subscription supports 220+ calls/month
- Rate limit: 5 req/min (enforced with backoff)

---

## Troubleshooting

### "No data found for 1-minute"
→ Ensure `config_1min.yaml` fetch completed successfully
→ Check `polygon_pipeline/reports/pipeline_1min.log` for API errors

### "IC differs by >10% between frequencies"
→ Normal if bar counts differ significantly
→ Check that date ranges overlap
→ Verify feature parameters are scaled correctly (5× ratio)

### API rate limit (429 errors)
→ Pipeline handles this automatically with exponential backoff
→ May take 12-24 hours to fetch full 1-min dataset
→ Check rate limit with `tail -f polygon_pipeline/reports/pipeline_1min.log`

### "Features computed but IC=0"
→ Insufficient bars for IC estimation
→ Increase rolling IC window or date range
→ Run single-day test first to debug

---

## Quick Checklist

### Initial Setup ✓
- [x] Create `config_5min.yaml` and `config_1min.yaml`
- [x] Update `run_features.py` with `--clean-dir` support
- [x] Update `run_alpha.py` with `--clean-dir` support
- [x] Create `compare_frequencies.py`

### First Run
- [ ] Fetch 5-min data (baseline): `python -m polygon_pipeline.pipeline.orchestrator run --config polygon_pipeline/configs/config_5min.yaml --start 2024-03-22 --end 2024-03-29`
- [ ] Fetch 1-min data (new): `python -m polygon_pipeline.pipeline.orchestrator run --config polygon_pipeline/configs/config_1min.yaml --start 2024-03-22 --end 2024-03-29`
- [ ] Compare single day: `python compare_frequencies.py --start 2024-03-22 --end 2024-03-22`
- [ ] Check IC stability: Look for ±5% agreement in IC values
- [ ] Run features on both: `python run_features.py --clean-dir polygon_pipeline/data/clean_5min/` then `--clean-dir polygon_pipeline/data/clean_1min/`
- [ ] Run alpha on both: `python run_alpha.py --clean-dir polygon_pipeline/data/clean_5min/` then `--clean-dir polygon_pipeline/data/clean_1min/`
- [ ] Compare alpha results: `python compare_frequencies.py --save-report`

---

## Storage Layout After Setup

```
YOUR_PROJECT/
├── polygon_pipeline/
│   ├── data/
│   │   ├── raw_5min/          (raw API responses, 5-min)
│   │   ├── raw_1min/          (raw API responses, 1-min)
│   │   ├── clean_5min/
│   │   │   ├── year=2024/data.parquet
│   │   │   ├── year=2025/data.parquet
│   │   │   └── year=2026/data.parquet
│   │   ├── clean_1min/        (NEW)
│   │   │   ├── year=2024/data.parquet
│   │   │   ├── year=2025/data.parquet
│   │   │   └── year=2026/data.parquet
│   │   └── cache/
│   ├── configs/
│   │   ├── config_5min.yaml   (NEW)
│   │   ├── config_1min.yaml   (NEW)
│   │   └── config.yaml        (legacy, keep for reference)
│   └── reports/
│       ├── pipeline_5min.log
│       ├── pipeline_1min.log
│       ├── frequency_comparison.csv   (after compare_frequencies.py)
│       ├── ic_summary_5min.csv
│       └── ic_summary_1min.csv
│
├── data/features/             (features from both frequencies)
├── data/alpha/                (alpha signals from both frequencies)
├── reports/                   (alpha & feature reports)
├── compare_frequencies.py     (NEW)
├── run_features.py
├── run_alpha.py
└── ...
```

---

## Next Steps

1. **Initial validation**: Run single-day test on both frequencies
2. **Full fetch**: Once validated, fetch complete date ranges
3. **Compare results**: Run `compare_frequencies.py` to validate IC stability
4. **Backtest**: Run backtests on both 1-min and 5-min alphas
5. **Optimize**: If 1-min IC is comparable, consider full migration

---

## Questions?

- **Why 5× data volume?** Trading happens every minute, not every 5 minutes.
- **Why keep 5-min?** Baseline for comparison; easier to compute; still has decent signal.
- **How long to fetch 1-min?** ~8-12 hours for 44 stocks over 2 years due to API rate limits (5 req/min).
- **Can I run both in parallel?** Yes, use different configs and separate terminals.
- **What if IC differs significantly?** Check data quality, ensure bar counts align, verify parameters.
