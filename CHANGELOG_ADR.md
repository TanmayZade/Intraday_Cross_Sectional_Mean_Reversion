# Architecture & Version Decision Record

This document maintains a chronological formal record of significant changes, architectural decisions (ADRs), and version updates. This structure ensures that we capture the "why" (context) and the "when" behind our design choices, not just the "what."

## 📋 Version Index

| Version | Date | Time | Change Summary | Status | Details |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v1.0.1** | 2026-04-15 | 15:35:17 | Established version tracking and ADR template | `Active` | [View Details](#v101---established-version-tracking-and-adr-template) |
| **v1.0.0** | 2026-04-15 | 10:00:00 | Migrated trading pipeline from NSE to US Market | `Active` | [View Details](#v100---migrated-trading-pipeline-from-nse-to-us-market) |

---

## 📝 Detailed Version Logs

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
