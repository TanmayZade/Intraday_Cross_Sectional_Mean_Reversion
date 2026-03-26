# Quick Start: Dual-Frequency (1-min + 5-min)

## What You Have Now

✅ **Separate storage directories**:
- `polygon_pipeline/data/clean_5min/` → 5-minute bars
- `polygon_pipeline/data/clean_1min/` → 1-minute bars (new)

✅ **Two configs ready to use**:
- `polygon_pipeline/configs/config_5min.yaml`
- `polygon_pipeline/configs/config_1min.yaml`

✅ **Scripts support both frequencies**:
- `run_features.py --clean-dir polygon_pipeline/data/clean_5min/`
- `run_features.py --clean-dir polygon_pipeline/data/clean_1min/`
- `run_alpha.py --clean-dir polygon_pipeline/data/clean_5min/`
- `run_alpha.py --clean-dir polygon_pipeline/data/clean_1min/`

✅ **Comparison tool** to validate IC stability:
- `python compare_frequencies.py`

---

## Recommended Workflow

### 1️⃣ Single-Day Test (Fast Validation)
```bash
# Fetch 1 day of 1-min data
python -m polygon_pipeline.pipeline.orchestrator run \
  --config polygon_pipeline/configs/config_1min.yaml \
  --start 2024-12-20 \
  --end 2024-12-20

# Fetch 1 day of 5-min data (baseline)
python -m polygon_pipeline.pipeline.orchestrator run \
  --config polygon_pipeline/configs/config_5min.yaml \
  --start 2024-12-20 \
  --end 2024-12-20

# Compare IC
python compare_frequencies.py --start 2024-12-20 --end 2024-12-20
```

**Expected**: IC within ±5% of each other ✓

---

### 2️⃣ Full Data Fetch (if validation passes)
```bash
# Fetch full 1-min dataset (~8-12 hours, monitor rate limits)
python -m polygon_pipeline.pipeline.orchestrator run \
  --config polygon_pipeline/configs/config_1min.yaml \
  --start 2024-03-22 \
  --end 2024-12-31

# Update 5-min baseline (much faster)
python -m polygon_pipeline.pipeline.orchestrator run \
  --config polygon_pipeline/configs/config_5min.yaml \
  --start 2024-03-22 \
  --end 2024-12-31
```

---

### 3️⃣ Feature Generation (Both Frequencies)
```bash
# 5-minute baseline features
python run_features.py \
  --clean-dir polygon_pipeline/data/clean_5min/ \
  --start 2024-12-20 --end 2024-12-31

# 1-minute features
python run_features.py \
  --clean-dir polygon_pipeline/data/clean_1min/ \
  --start 2024-12-20 --end 2024-12-31
```

---

### 4️⃣ Alpha Generation (Both Frequencies)
```bash
# 5-minute baseline alpha
python run_alpha.py \
  --clean-dir polygon_pipeline/data/clean_5min/ \
  --start 2024-12-20 --end 2024-12-31

# 1-minute alpha
python run_alpha.py \
  --clean-dir polygon_pipeline/data/clean_1min/ \
  --start 2024-12-20 --end 2024-12-31
```

---

### 5️⃣ Validate IC Across Full Dataset
```bash
python compare_frequencies.py --save-report
```

This generates:
- `reports/frequency_comparison.csv` (IC comparison table)
- `reports/ic_summary_5min.csv` (5-min IC per feature)
- `reports/ic_summary_1min.csv` (1-min IC per feature)

---

## Parameter Quick Reference

### 5-Minute Parameters
```bash
python run_features.py \
  --clean-dir polygon_pipeline/data/clean_5min/ \
  --freq 5min \
  --atr-window 60 \
  --vol-window 180 \
  --zscore-window 180 \
  --beta-window 1170 \
  --halflife 39
```

### 1-Minute Parameters
```bash
python run_features.py \
  --clean-dir polygon_pipeline/data/clean_1min/ \
  --freq 1min \
  --atr-window 300 \
  --vol-window 900 \
  --zscore-window 900 \
  --beta-window 5850 \
  --halflife 195
```

---

## Files Created/Updated

| File | Purpose |
|------|---------|
| `configs/config_5min.yaml` | 5-min pipeline config |
| `configs/config_1min.yaml` | 1-min pipeline config |
| `run_features.py` | Updated to support `--clean-dir` |
| `run_alpha.py` | Updated to support `--clean-dir` |
| `compare_frequencies.py` | NEW: IC comparison tool |
| `DUAL_FREQUENCY_WORKFLOW.md` | NEW: Full workflow guide |

---

## Storage Overview

```
polygon_pipeline/data/
├── clean_5min/              ← 5-minute bars (existing)
│   ├── year=2024/data.parquet
│   ├── year=2025/data.parquet
│   └── year=2026/data.parquet
│
└── clean_1min/              ← 1-minute bars (new)
    ├── year=2024/data.parquet
    ├── year=2025/data.parquet
    └── year=2026/data.parquet
```

Total storage: ~900 MB - 1.3 GB for 44 stocks, 2 years

---

## Key Design Decisions

✅ **Separate configs** (`config_5min.yaml` vs `config_1min.yaml`)
- Clear intent: one config per frequency
- Easy to maintain: no confusion about which multiplier is active
- Simple to swap: just change `--config` argument

✅ **Separate clean directories** (`clean_5min/` vs `clean_1min/`)
- No schema changes needed
- Both use identical Parquet format (frequency-agnostic)
- Easy to debug: each directory is self-contained
- Simple filtering: just change `--clean-dir` argument

✅ **Updated scripts** to accept `--clean-dir` parameter
- Backward compatible
- Explicit: user specifies which frequency to use
- Consistent defaults: both scripts default to `clean_1min/`

✅ **Comparison tool** (`compare_frequencies.py`)
- Validates IC stability across frequencies
- Identifies potential issues early
- Generates reports for analysis

---

## Next Actions

1. **Test single day** (5-10 min): `python compare_frequencies.py --start 2024-12-20 --end 2024-12-20`
2. **If IC stable** (±5%): Proceed to full fetch
3. **Fetch full data** (8-12 hours): `python -m polygon_pipeline.pipeline.orchestrator run --config polygon_pipeline/configs/config_1min.yaml --start 2024-03-22 --end 2024-12-31`
4. **Generate features** (15-30 min): `python run_features.py --clean-dir polygon_pipeline/data/clean_1min/`
5. **Build alpha** (1-2 min): `python run_alpha.py --clean-dir polygon_pipeline/data/clean_1min/`
6. **Compare results**: `python compare_frequencies.py --save-report`

---

## FAQ

**Q: Can I keep using just 5-min?**
A: Yes! Configs are independent. Just use `config_5min.yaml` and ignore 1-min setup.

**Q: How long to fetch 1-min data?**
A: ~8-12 hours for 44 stocks, 2 years (API rate limit: 5 req/min).

**Q: Will IC differ between frequencies?**
A: Should be stable (±5%) because IC is rank-based, not frequency-dependent.

**Q: Can I run both fetches in parallel?**
A: No, they'll hit same API rate limit. Run sequentially or in separate sessions.

**Q: What if IC differs by >10%?**
A: Check data quality, ensure dates overlap, verify parameters. See DUAL_FREQUENCY_WORKFLOW.md troubleshooting.

**Q: How much disk space do I need?**
A: ~900 MB - 1.3 GB total (5-min ~200MB + 1-min ~700MB for 44 stocks, 2 years).
