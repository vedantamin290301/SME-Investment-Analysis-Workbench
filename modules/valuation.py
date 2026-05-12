from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import latest, module_summary, result, safe_financial_div, score_lower_better


def ps_ratio(df: pd.DataFrame, benchmarks: dict, market_cap: float) -> dict:
    value = safe_financial_div(market_cap, latest(df["revenue"].tolist()))
    benchmark = benchmarks.get("ps", 2)
    return result("P/S Ratio", score_lower_better(value, benchmark, benchmark * 2), {"Latest x": value, "Sector Median x": benchmark})


def ev_ebitda(df: pd.DataFrame, benchmarks: dict, market_cap: float) -> dict:
    enterprise_value = market_cap + latest(df["total_debt"].tolist()) - latest(df["cash"].tolist())
    value = safe_financial_div(enterprise_value, latest(df["ebitda"].tolist()))
    benchmark = benchmarks.get("ev_ebitda", 12)
    flags = ["EV/EBITDA significantly above sector median; possible overvaluation."] if value > benchmark * 1.5 else []
    if latest(df["ebitda"].tolist()) <= 0:
        flags.append("EBITDA is zero or negative; EV/EBITDA is not meaningful and should be treated as high risk.")
    return result("EV/EBITDA", score_lower_better(value, benchmark, benchmark * 2), {"Latest x": value, "Sector Median x": benchmark}, flags)


def pb_ratio(df: pd.DataFrame, benchmarks: dict, market_cap: float) -> dict:
    value = safe_financial_div(market_cap, latest(df["equity"].tolist()))
    roe = safe_financial_div(latest(df["net_profit"].tolist()), latest(df["equity"].tolist())) * 100
    benchmark = benchmarks.get("pb", 3)
    score = score_lower_better(value, benchmark, benchmark * 2)
    if roe > benchmarks.get("roe", 16):
        score += 10
    flags = ["High P/B with low ROE; valuation looks expensive."] if value > benchmark and roe < benchmarks.get("roe", 16) else []
    if latest(df["equity"].tolist()) <= 0:
        flags.append("Book equity is zero or negative; P/B valuation is not meaningful.")
    return result("P/B Ratio", score, {"Latest x": value, "ROE %": roe, "Sector Median x": benchmark}, flags)


def analyze_valuation(df: pd.DataFrame, benchmarks: dict, market_cap: float) -> dict:
    metrics = {
        "ps_ratio": ps_ratio(df, benchmarks, market_cap),
        "ev_ebitda": ev_ebitda(df, benchmarks, market_cap),
        "pb_ratio": pb_ratio(df, benchmarks, market_cap),
    }
    return module_summary(metrics)
