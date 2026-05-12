from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import latest, module_summary, ratio_series, result, safe_financial_div, score_higher_better, score_lower_better


def current_ratio(df: pd.DataFrame, benchmarks: dict) -> dict:
    values = ratio_series(df["current_assets"], df["current_liabilities"])
    flags = ["Current ratio below 1; liquidity stress."] if latest(values) < 1 else []
    return result("Current Ratio", score_higher_better(latest(values), 1, 2), {"Latest x": latest(values), "Trend": values}, flags)


def quick_ratio(df: pd.DataFrame, benchmarks: dict) -> dict:
    values = ratio_series(df["current_assets"] - df["inventory"], df["current_liabilities"])
    flags = ["Quick ratio below 1; inventory-dependent liquidity."] if latest(values) < 1 else []
    return result("Quick Ratio", score_higher_better(latest(values), 0.7, 1.5), {"Latest x": latest(values), "Trend": values}, flags)


def interest_coverage(df: pd.DataFrame, benchmarks: dict) -> dict:
    values = ratio_series(df["ebit"], df["interest_expense"])
    flags = []
    if latest(values) < 2:
        flags.append("Interest coverage below 2x in a high-interest environment.")
    if latest(values) < 1:
        flags.append("Interest coverage below 1x; critical debt service risk.")
    return result("Interest Coverage", score_higher_better(latest(values), 1, 4), {"Latest x": latest(values), "Trend": values}, flags)


def dscr(df: pd.DataFrame, benchmarks: dict) -> dict:
    service = df["interest_expense"] + df["principal_repayments"]
    values = ratio_series(df["operating_income"], service)
    flags = ["DSCR below 1; insufficient cash generation for debt service."] if latest(values) < 1 else []
    return result("DSCR", score_higher_better(latest(values), 1, 1.5), {"Latest x": latest(values), "Trend": values}, flags)


def cash_debt_coverage(df: pd.DataFrame, benchmarks: dict) -> dict:
    values = ratio_series(df["operating_cash_flow"], df["total_debt"])
    flags = ["OCF covers less than 10% of debt."] if latest(values) < 0.1 else []
    return result("Cash Debt Coverage", score_higher_better(latest(values), 0.05, 0.4), {"Latest x": latest(values), "Trend": values}, flags)


def net_debt_to_ebitda(df: pd.DataFrame, benchmarks: dict) -> dict:
    values = ratio_series(df["total_debt"] - df["cash"], df["ebitda"])
    flags = ["Net debt / EBITDA above 4x; high leverage warning."] if latest(values) > 4 else []
    return result("Net Debt-to-EBITDA", score_lower_better(latest(values), 2, 4), {"Latest x": latest(values), "Trend": values}, flags)


def debt_ratio(df: pd.DataFrame, benchmarks: dict) -> dict:
    values = ratio_series(df["total_debt"], df["total_assets"])
    flags = ["Debt ratio above 0.6; heavily debt-financed balance sheet."] if latest(values) > 0.6 else []
    return result("Debt Ratio", score_lower_better(latest(values), 0.4, 0.6), {"Latest x": latest(values), "Trend": values}, flags)


def long_term_debt_cap(df: pd.DataFrame, benchmarks: dict) -> dict:
    capitalization = df["long_term_debt"] + df["equity"]
    values = ratio_series(df["long_term_debt"], capitalization)
    return result("Long-Term Debt-to-Capitalization", score_lower_better(latest(values), 0.25, 0.6), {"Latest x": latest(values), "Trend": values})


def goodwill_assets(df: pd.DataFrame, benchmarks: dict) -> dict:
    values = ratio_series(df["goodwill"], df["total_assets"], 100)
    flags = ["Goodwill exceeds 50% of assets; impairment can wipe equity."] if latest(values) > 50 else []
    score = score_lower_better(latest(values), 10, 40)
    return result("Goodwill as % of Assets", score, {"Latest %": latest(values), "Trend": values}, flags)


def analyze_balance_sheet(df: pd.DataFrame, benchmarks: dict) -> dict:
    metrics = {
        "current_ratio": current_ratio(df, benchmarks),
        "quick_ratio": quick_ratio(df, benchmarks),
        "interest_coverage": interest_coverage(df, benchmarks),
        "dscr": dscr(df, benchmarks),
        "cash_debt_coverage": cash_debt_coverage(df, benchmarks),
        "net_debt_to_ebitda": net_debt_to_ebitda(df, benchmarks),
        "debt_ratio": debt_ratio(df, benchmarks),
        "long_term_debt_cap": long_term_debt_cap(df, benchmarks),
        "goodwill_assets": goodwill_assets(df, benchmarks),
    }
    return module_summary(metrics)
