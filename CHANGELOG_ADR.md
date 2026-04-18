# Architecture & Version Decision Record

This document maintains a chronological formal record of significant changes, architectural decisions (ADRs), and version updates. This structure ensures that we capture the "why" (context) and the "when" behind our design choices, not just the "what."

## 📋 Version Index

| Version | Date | Time | Change Summary | Status | Details |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v1.2.0** | 2026-04-18 | 23:54:00 | Documentation Sync & ADR Formalization | `Active` | [View Details](#v120---documentation-sync--adr-formalization) |
| **v1.1.0** | 2026-04-16 | 19:54:29 | Optimized NASDAQ pipeline execution parameters | `Active` | [View Details](#v110---optimized-nasdaq-pipeline-execution-parameters) |
| **v1.0.1** | 2026-04-15 | 15:35:17 | Established version tracking and ADR template | `Active` | [View Details](#v101---established-version-tracking-and-adr-template) |
| **v1.0.0** | 2026-04-15 | 10:00:00 | Migrated trading pipeline from NSE to US Market | `Active` | [View Details](#v100---migrated-trading-pipeline-from-nse-to-us-market) |
| **v0.2.0** | 2026-04-14 | 20:10:55 | Enforced single day trades constraint | `Deprecated` | [View Details](#v020---enforced-single-day-trades-constraint) |
| **v0.1.0** | 2026-03-26 | 19:52:38 | Initial implementation & 15-minute strategy baseline | `Deprecated` | [View Details](#v010---initial-implementation--15-minute-strategy-baseline) |

---

## 📝 Detailed Version Logs

### v1.1.0 - Optimized NASDAQ pipeline execution parameters

* **Date:** 2026-04-16
* **Time:** 19:54:29 (+05:30)

#### 1. Context & Problem Statement
Following the migration to the US market, the strategy was experiencing frequent "Time Exit" trades and suboptimal profit capture. Execution parameters needed calibration to maximize daily profitability and improve the balance between risk management and profit capture.

#### 2. Decision / Changes Implemented
* Calibrated execution parameters including stop-loss, profit-taking, and trailing stop triggers for the NASDAQ market.
* Refined intraday mean-reversion logic to better capture market anomalies.
* Maintained a robust, automated logging and backtesting workflow.

#### 3. Consequences
* **Positive:** Intraday trades now have an optimized exit strategy, reducing "Time Exit" frequency. Increased daily profitability margins through improved handling of risk parameters.
* **Risks:** The calibrated parameters are specifically tuned for the current market regime and might need periodic adjustments depending on volatility.

<br/><hr/><br/>

### v1.0.1 - Established version tracking and ADR template

* **Date:** 2026-04-15
* **Time:** 15:35:17 (+05:30)

#### 1. Context & Problem Statement
As the project scales and multiple trading strategies are iteratively improved, a professional, centralized document is required to maintain the history of all changes. It is crucial to have a source of truth—inspired by Arc42 and standard ADRs—that efficiently tracks version names, exact timestamps, and detailed rationales.

#### 2. Decision / Changes Implemented
* Created the `CHANGELOG_ADR.md` document at the project root.
* Implemented a tabular index at the top, equipped with local markdown references linking directly to the comprehensive version elaborations.
* Mandated a structured layout containing **Context**, **Decision**, and **Consequences** for future architectural updates.

#### 3. Consequences
* **Positive:** Greatly improves maintainability, simplifies onboarding, and clearly documents the evolution of the quantitative strategies.
* **Risks:** The log requires manual updating, so it must be added to the standard development lifecycle following any major milestone.

<br/><hr/><br/>

### v1.0.0 - Migrated trading pipeline from NSE to US Market

* **Date:** 2026-04-15
* **Time:** 10:00:00 (+05:30)

#### 1. Context & Problem Statement
The original intraday mean-reversion trading pipeline was tailored exclusively for the Indian Stock Exchange (NSE). In order to capture different market anomalies, the decision was made to reconfigure the system entirely for the US market.

#### 2. Decision / Changes Implemented
* Systematically removed all NSE-specific logic including session times, local circuit breaker rules, and currency symbols.
* Reconfigured the target portfolio for a 80-stock US tech/growth strategy, starting with $100,000 capital.
* Switched the primary hedging component to QQQ.

#### 3. Consequences
* The pipeline is fully equipped to backtest against US market hours correctly.
* Any lingering dependencies on the previous feature engine meant strictly for NSE have been deprecated.

<br/><hr/><br/>

### v0.2.0 - Enforced single day trades constraint

* **Date:** 2026-04-14
* **Time:** 20:10:55 (+05:30)

#### 1. Context & Problem Statement
The strategy previously held positions across multiple sessions which introduced overnight gap risks. It was necessary to strictly restrict the pipeline to execute and close all trades within a single trading day to validate specific mean reversion hypothesis.

#### 2. Decision / Changes Implemented
* Implemented strict intraday constraints ensuring all positions are squared off by the end of the trading session.
* Refactored trading logic to prioritize single day trades.

#### 3. Consequences
* **Positive:** Reduced exposure to overnight market risk. Transformed the algorithm into a pure intraday trading system.
* **Risks:** Missing out on potential multi-day continued momentum.

<br/><hr/><br/>

### v0.1.0 - Initial implementation & 15-minute strategy baseline

* **Date:** 2026-03-26
* **Time:** 19:52:38 (+05:30)

#### 1. Context & Problem Statement
Initial conceptualization and implementation of the Intraday Cross-Sectional Mean Reversion strategy.

#### 2. Decision / Changes Implemented
* Established the initial quantitative strategy operating on 15-minute interval data.
* Set up the fundamental backtesting pipeline for evaluating strategy performance.
* Implemented baseline logic for signal generation and position sizing.

#### 3. Consequences
* **Positive:** Provided a functional baseline for subsequent metric tracking and logical improvements. Required infrastructure for execution created.
* **Risks:** Baseline strategy needed significant tuning since early assumptions heavily affect future behavior.
