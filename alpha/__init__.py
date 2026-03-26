"""
alpha/
======
Phase 2: Alpha Signal Construction + Portfolio Construction.

Modules
-------
signal.py    — IC-weighted composite alpha from 15 features
portfolio.py — dollar-neutral, vol-scaled position weights

Quick start
-----------
    from alpha.signal    import AlphaModel
    from alpha.portfolio import PortfolioBuilder

    # Build composite alpha
    model   = AlphaModel(features, close)
    alpha   = model.composite_alpha()
    summary = model.ic_summary_table()
    print(summary)

    # Build portfolio weights
    builder = PortfolioBuilder(alpha, close, volume)
    weights = builder.build()
"""
from alpha.signal    import AlphaModel, compute_ic_series, compute_ic_decay
from alpha.portfolio import PortfolioBuilder