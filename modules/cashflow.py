from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import latest, module_summary, ratio_series, result, score_higher_better, score_lower_better, trend


def operating_cash_flow(df: pd.DataFrame, benchmarks: dict) -> dict:
    values = df["operating_cash_flow"].tolist()
    score = score_higher_better(latest(values), 0, max(latest(df["net_profit"].tolist()), 1))
    if trend(values) == "rising":
        score += 10
    flags = ["OCF negative for 2+ consecutive years."] if sum(v < 0 for v in values[-2:]) >= 2 else []
    if latest(values) < 0 < latest(df["net_profit"].tolist()):
        flags.append("Negative OCF despite accounting profits.")
    return result("Operating Cash Flow", score, {"Latest": latest(values), "Trend": values}, flags)


def cash_conversion_ratio(df: pd.DataFrame, benchmarks: dict) -> dict:
    values = ratio_series(df["operating_cash_flow"], df["net_profit"], allow_negative_denominator=True)
    flags = ["Cash conversion ratio below 0.5; profit quality concern."] if latest(values) < 0.5 else []
    return result("Cash Conversion Ratio", score_higher_better(latest(values), 0.5, 1), {"Latest x": latest(values), "Trend": values}, flags)


def free_cash_flow(df: pd.DataFrame, benchmarks: dict) -> dict:
    values = (df["operating_cash_flow"] - df["capex"]).tolist()
    margin = ratio_series(df["operating_cash_flow"] - df["capex"], df["revenue"], 100)
    flags = ["FCF is negative in recent years; business may be cash-burning."] if sum(v < 0 for v in values[-2:]) >= 2 else []
    return result("Free Cash Flow", score_higher_better(latest(values), 0, max(latest(df["operating_cash_flow"].tolist()), 1)), {"Latest": latest(values), "FCF Margin %": latest(margin), "Trend": values}, flags)


def capex_intensity(df: pd.DataFrame, benchmarks: dict) -> dict:
    values = []
    for capex, ocf in zip(df["capex"].tolist(), df["operating_cash_flow"].tolist()):
        if ocf <= 0 and capex > 0:
            values.append(999.0)
        else:
            values.extend(ratio_series([capex], [ocf]))
    flags = ["Capex consistently exceeds OCF; likely funded by debt or equity issuance."] if all(v > 1 for v in values[-2:]) else []
    return result("Capex Intensity", score_lower_better(latest(values), 0.5, 1.2), {"Latest x": latest(values), "Trend": values}, flags)


def financing_debt_trend(df: pd.DataFrame, benchmarks: dict) -> dict:
    debt_raised = df["debt_raised"].tolist()
    profits = df["net_profit"].tolist()
    repeated = sum(v > 0 for v in debt_raised) == len(debt_raised)
    weak_profit = latest(profits) <= profits[0]
    score = 35 if repeated and weak_profit else 80 if repeated else 95
    flags = ["Debt raised every year with no profitability improvement."] if repeated and weak_profit else []
    return result("Financing Cash Flow - Debt Trend", score, {"Debt Raised Trend": debt_raised}, flags)


def analyze_cash_flow(df: pd.DataFrame, benchmarks: dict) -> dict:
    metrics = {
        "operating_cash_flow": operating_cash_flow(df, benchmarks),
        "cash_conversion_ratio": cash_conversion_ratio(df, benchmarks),
        "free_cash_flow": free_cash_flow(df, benchmarks),
        "capex_intensity": capex_intensity(df, benchmarks),
        "financing_debt_trend": financing_debt_trend(df, benchmarks),
    }
    return module_summary(metrics)
