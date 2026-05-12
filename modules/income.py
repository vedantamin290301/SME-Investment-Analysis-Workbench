from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import average_pair, cagr, latest, module_summary, pct_change, result, safe_div, score_higher_better, score_lower_better, trend


def revenue_growth(df: pd.DataFrame, benchmarks: dict) -> dict:
    revenue = df["revenue"].tolist()
    yoy = [pct_change(revenue[i], revenue[i - 1]) for i in range(1, len(revenue))]
    growth_cagr = cagr(revenue)
    score = score_higher_better(growth_cagr, 0, 20)
    flags = []
    if len(yoy) >= 2 and max(yoy) > 30 and latest(yoy) < 10:
        flags.append("Single-year revenue spike was not sustained across 3 years.")
    if growth_cagr <= 0:
        flags.append("Revenue is flat or declining.")
    return result("Revenue Growth", score, {"3Y CAGR %": growth_cagr, "YoY %": yoy}, flags)


def gross_margin(df: pd.DataFrame, benchmarks: dict) -> dict:
    margins = ((df["revenue"] - df["cogs"]) / df["revenue"] * 100).replace([np.inf, -np.inf], 0).fillna(0).tolist()
    benchmark = benchmarks.get("gross_margin", 35)
    score = score_higher_better(latest(margins), benchmark * 0.6, benchmark)
    if trend(margins) == "rising":
        score += 10
    flags = ["Gross margin is falling, suggesting pricing power erosion."] if trend(margins) == "falling" else []
    return result("Gross Margin", score, {"Latest %": latest(margins), "Trend": margins, "Sector Benchmark %": benchmark}, flags)


def ebitda_margin(df: pd.DataFrame, benchmarks: dict, sector: str) -> dict:
    margins = (df["ebitda"] / df["revenue"] * 100).replace([np.inf, -np.inf], 0).fillna(0).tolist()
    benchmark = benchmarks.get("ebitda_margin", 12)
    score = score_higher_better(latest(margins), 0, benchmark)
    if trend(margins) == "rising":
        score += 10
    if trend(margins) == "falling":
        score -= 20
    floor = 10 if sector == "Manufacturing" else 5 if sector == "Trading" else 8
    flags = []
    if latest(margins) < floor:
        flags.append(f"EBITDA margin below {floor}% floor for {sector}.")
    if trend(margins) == "falling":
        flags.append("EBITDA margin is falling over the 3-year period.")
    return result("EBITDA Margin", score, {"Latest %": latest(margins), "Trend": margins}, flags)


def pat_margin(df: pd.DataFrame, benchmarks: dict) -> dict:
    margins = (df["net_profit"] / df["revenue"] * 100).replace([np.inf, -np.inf], 0).fillna(0).tolist()
    score = score_higher_better(latest(margins), -5, 12)
    if min(margins) < 0 and trend(margins) in {"rising", "improving"}:
        score += 15
    flags = ["Losses are widening as a percentage of revenue."] if latest(margins) < 0 and trend(margins) == "falling" else []
    return result("PAT Margin", score, {"Latest %": latest(margins), "Trend": margins}, flags)


def roce(df: pd.DataFrame, benchmarks: dict, cost_of_debt: float = 12) -> dict:
    values = (df["ebit"] / (df["total_assets"] - df["current_liabilities"]) * 100).replace([np.inf, -np.inf], 0).fillna(0).tolist()
    score = score_higher_better(latest(values), cost_of_debt, max(cost_of_debt + 10, 25))
    flags = []
    if latest(values) < cost_of_debt:
        flags.append("ROCE is below cost of debt, indicating value destruction.")
    if latest(values) < 10:
        flags.append("ROCE below 10% for a non-early-stage SME.")
    return result("ROCE", score, {"Latest %": latest(values), "Trend": values, "Cost of Debt %": cost_of_debt}, flags)


def asset_turnover(df: pd.DataFrame, benchmarks: dict) -> dict:
    assets = df["total_assets"].tolist()
    values = [safe_div(df["revenue"].iloc[i], average_pair(assets, i)) for i in range(len(df))]
    benchmark = benchmarks.get("asset_turnover", 1.2)
    score = score_higher_better(latest(values), benchmark * 0.5, benchmark)
    flags = ["Asset turnover is below sector median."] if latest(values) < benchmark else []
    return result("Asset Turnover", score, {"Latest x": latest(values), "Trend": values, "Sector Median x": benchmark}, flags)


def fixed_asset_turnover(df: pd.DataFrame, benchmarks: dict) -> dict:
    values = (df["revenue"] / df["net_fixed_assets"]).replace([np.inf, -np.inf], 0).fillna(0).tolist()
    score = score_higher_better(latest(values), 0.5, benchmarks.get("fixed_asset_turnover", 3))
    if trend(values) == "rising":
        score += 15
    flags = ["Fixed asset turnover is falling; recent capex may not be paying off."] if trend(values) == "falling" else []
    return result("Fixed Asset Turnover", score, {"Latest x": latest(values), "Trend": values}, flags)


def roa(df: pd.DataFrame, benchmarks: dict) -> dict:
    assets = df["total_assets"].tolist()
    values = [safe_div(df["net_profit"].iloc[i], average_pair(assets, i)) * 100 for i in range(len(df))]
    score = score_higher_better(latest(values), 5, 15)
    flags = ["ROA below 5%; capital-heavy warning."] if latest(values) < 5 else []
    return result("ROA", score, {"Latest %": latest(values), "Trend": values}, flags)


def debtor_days(df: pd.DataFrame, benchmarks: dict) -> dict:
    values = (df["receivables"] / df["revenue"] * 365).replace([np.inf, -np.inf], 0).fillna(0).tolist()
    benchmark = benchmarks.get("dso", 75)
    score = score_lower_better(latest(values), benchmark, 120)
    flags = ["DSO above 90 days; cash crunch risk."] if latest(values) > 90 else []
    return result("Debtor Days", score, {"Latest Days": latest(values), "Trend": values, "Sector Norm": benchmark}, flags)


def inventory_days(df: pd.DataFrame, benchmarks: dict) -> dict:
    values = (df["inventory"] / df["cogs"] * 365).replace([np.inf, -np.inf], 0).fillna(0).tolist()
    benchmark = benchmarks.get("dio", 90)
    score = score_lower_better(latest(values), benchmark, benchmark * 1.8)
    flags = ["DIO rising 3 years in a row; unsold stock warning."] if trend(values) == "rising" else []
    return result("Inventory Days", score, {"Latest Days": latest(values), "Trend": values}, flags)


def creditor_days(df: pd.DataFrame, benchmarks: dict) -> dict:
    values = (df["payables"] / df["cogs"] * 365).replace([np.inf, -np.inf], 0).fillna(0).tolist()
    benchmark = benchmarks.get("dpo", 60)
    score = 90 if benchmark <= latest(values) <= benchmark * 1.6 else score_lower_better(abs(latest(values) - benchmark), 0, benchmark)
    flags = ["Extremely high DPO may signal inability to pay suppliers on time."] if latest(values) > benchmark * 2 else []
    return result("Creditor Days", score, {"Latest Days": latest(values), "Trend": values}, flags)


def inventory_turnover(df: pd.DataFrame, benchmarks: dict) -> dict:
    inventory = df["inventory"].tolist()
    values = [safe_div(df["cogs"].iloc[i], average_pair(inventory, i)) for i in range(len(df))]
    benchmark = benchmarks.get("inventory_turnover", 5)
    score = score_higher_better(latest(values), benchmark * 0.5, benchmark)
    return result("Inventory Turnover", score, {"Latest x": latest(values), "Trend": values, "Sector Peer x": benchmark})


def cash_conversion_cycle(df: pd.DataFrame, benchmarks: dict) -> dict:
    dso = (df["receivables"] / df["revenue"] * 365).replace([np.inf, -np.inf], 0).fillna(0)
    dio = (df["inventory"] / df["cogs"] * 365).replace([np.inf, -np.inf], 0).fillna(0)
    dpo = (df["payables"] / df["cogs"] * 365).replace([np.inf, -np.inf], 0).fillna(0)
    values = (dso + dio - dpo).tolist()
    score = 100 if latest(values) < 0 else score_lower_better(latest(values), benchmarks.get("ccc", 60), 180)
    flags = ["CCC worsening year over year."] if trend(values) == "rising" else []
    return result("Cash Conversion Cycle", score, {"Latest Days": latest(values), "Trend": values}, flags)


def analyze_income_and_working_capital(df: pd.DataFrame, benchmarks: dict, sector: str, cost_of_debt: float) -> tuple[dict, dict]:
    income = {
        "revenue_growth": revenue_growth(df, benchmarks),
        "gross_margin": gross_margin(df, benchmarks),
        "ebitda_margin": ebitda_margin(df, benchmarks, sector),
        "pat_margin": pat_margin(df, benchmarks),
        "roce": roce(df, benchmarks, cost_of_debt),
        "asset_turnover": asset_turnover(df, benchmarks),
        "fixed_asset_turnover": fixed_asset_turnover(df, benchmarks),
        "roa": roa(df, benchmarks),
    }
    working_capital = {
        "debtor_days": debtor_days(df, benchmarks),
        "inventory_days": inventory_days(df, benchmarks),
        "creditor_days": creditor_days(df, benchmarks),
        "inventory_turnover": inventory_turnover(df, benchmarks),
        "cash_conversion_cycle": cash_conversion_cycle(df, benchmarks),
    }
    return module_summary(income), module_summary(working_capital)

