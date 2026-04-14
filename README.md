# 📈 Intraday Cross-Sectional Mean Reversion (NSE India)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A professional-grade, quantitative **intraday mean reversion trading strategy** built specifically for the National Stock Exchange (NSE) of India. 

This repository provides a complete, autonomous pipeline that dynamically screens the NSE universe, calculates alpha features from 5-minute OHLCV data, models signals using Information Coefficient (IC) weighting, and constructs a dollar-neutral, risk-managed portfolio.

---

## ✨ Features

- **Dynamic NSE Universe Selection:** Auto-screens 100 stocks (50 volatile, 50 non-volatile) from a master seed pool using rolling ATR% rankings.
- **Robust Data Pipeline:** Built-in `yfinance` integration for NSE tickers (with `.NS` suffix). Automatically filters out after-hours noise and aligns strictly to NSE market hours (9:15 AM - 3:30 PM IST).
- **7-Factor Alpha Engine:** Computes sophisticated cross-sectional features including Bar Reversals, VWAP Deviations, Volume Shocks, and an **NSE-specific Circuit Limit Proximity** signal.
- **Smart Portfolio Construction:** Generates beta-neutral, volatility-scaled position weights while factoring in 1 bps transaction costs and target leverage.
- **Strict Risk Management:** Detects NSE circuit breakers (5%, 10%, 20%) and market-wide stress (NIFTY crashes) to dynamically scale down exposure.

---

## 🚀 Quick Start (UX Friendly Guide)

### 1. Prerequisites
You only need Python 3.9+ and pip installed. We recommend using a virtual environment (`.venv`).

### 2. Installation
Clone the repo and install the required dependencies:
```bash
git clone https://github.com/yourusername/nse-mean-reversion.git
cd nse-mean-reversion
pip install -r requirements.txt
```

### 3. Run the Strategy

**A. The "Test Drive" (Fastest)**
Want to see how it works instantly? Run it on just 3 blue-chip stocks for the last 30 days:
```bash
python run_pipeline.py --tickers RELIANCE TCS INFY --days 30
```

**B. The "Standard Run" (No dynamic screening)**
Run the strategy using a default seed pool of stocks without waiting to dynamically screen thousands of tickers:
```bash
python run_pipeline.py --skip-universe 
```

**C. The "Full Production Run"**
Execute the entire pipeline: dynamically screen the universe, fetch data, compute alpha, and simulate portfolio returns:
```bash
python run_pipeline.py
```

---

## 🧠 Architecture Overview

The codebase is modular and designed for easy extension by quants and developers:

- **`config/config.yaml`**: The brain of the operation. Change leverage, capital (default ₹10L), transaction costs, and signal windows here.
- **`nse_pipeline/`**: The data layer. Handles downloading `yfinance` data, cleaning bad ticks, handling splits, and storing to fast Parquet formats.
- **`features/`**: The math layer. Transforms raw OHLCV into cross-sectional Z-scores. 
- **`alpha/`**: The portfolio layer. Weights the 7 features based on their rolling predictive power (IC), and outputs target trade weights.

---

## 🔮 Next Version: Agentic A2A Architecture

In upcoming versions, this monolithic pipeline will evolve into an **Agent-to-Agent (A2A)** collaborative AI system. 

Instead of sequential scripts, the strategy will be driven by specialized **Personal AI Agents** acting over the MCP (Model Context Protocol):
1. **The Data Agent**: Autonomously monitors NSE feeds and triggers downstream events.
2. **The Quant Agent**: Continuously researches and optimizes feature windows.
3. **The Risk Agent**: Oversees portfolio health, acting as an independent kill-switch for circuit breakers or extreme drawdowns.

These agents will negotiate and communicate via A2A protocols, allowing for a fully autonomous, self-healing trading operation.

---

## 🤝 Contributing

We welcome contributions from the open-source community! Whether you are a quant researcher wanting to add a new alpha feature, or a Python dev looking to optimize Pandas operations:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingAlpha`)
3. Commit your changes (`git commit -m 'Add AmazingAlpha feature'`)
4. Push to the branch (`git push origin feature/AmazingAlpha`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

> **Disclaimer:** This project is for educational and research purposes only. It is not financial advice. Trading Indian equities on margin involves significant risk.