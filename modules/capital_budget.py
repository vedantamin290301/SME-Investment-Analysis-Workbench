from __future__ import annotations

import numpy as np

from .utils import module_summary, result, safe_div, safe_financial_div, score_higher_better, score_lower_better


def npv(initial_investment: float, cash_flows: list[float], discount_rate: float) -> dict:
    rate = discount_rate / 100
    value = sum(cf / ((1 + rate) ** (idx + 1)) for idx, cf in enumerate(cash_flows)) - initial_investment
    score = 80 if value > 0 else 50 if abs(value) < initial_investment * 0.05 else 25
    return result("NPV", score, {"Value": value, "Discount Rate %": discount_rate})


def irr(initial_investment: float, cash_flows: list[float], target_irr: float) -> dict:
    flows = [-initial_investment] + cash_flows
    try:
        import numpy_financial as npf

        value = float(npf.irr(flows) * 100)
    except Exception:
        value = _irr_bisection(flows) * 100
    if not np.isfinite(value):
        value = -100.0
    score = score_higher_better(value, 15, 30)
    if value >= 40:
        score = 100
    flags = ["IRR below fund target."] if value < target_irr else []
    if sum(cf > 0 for cf in cash_flows) == 0:
        flags.append("Projected cash flows never turn positive; IRR is not investable.")
    return result("IRR", score, {"IRR %": value, "Target IRR %": target_irr}, flags)


def _irr_bisection(flows: list[float]) -> float:
    low, high = -0.95, 5.0
    for _ in range(120):
        mid = (low + high) / 2
        val = sum(cf / ((1 + mid) ** idx) for idx, cf in enumerate(flows))
        if val > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def payback_period(initial_investment: float, cash_flows: list[float]) -> dict:
    recovered = 0.0
    period = float("inf")
    for idx, cf in enumerate(cash_flows, start=1):
        if recovered + cf >= initial_investment:
            period = idx - 1 + safe_financial_div(initial_investment - recovered, cf)
            break
        recovered += cf
    score = score_lower_better(period, 2, 5) if period != float("inf") else 10
    return result("Payback Period", score, {"Years": period, "Warning": "Ignores time value of money."})


def discounted_payback_period(initial_investment: float, cash_flows: list[float], discount_rate: float) -> dict:
    rate = discount_rate / 100
    discounted = [cf / ((1 + rate) ** idx) for idx, cf in enumerate(cash_flows, start=1)]
    recovered = 0.0
    period = float("inf")
    for idx, cf in enumerate(discounted, start=1):
        if recovered + cf >= initial_investment:
            period = idx - 1 + safe_financial_div(initial_investment - recovered, cf)
            break
        recovered += cf
    score = score_lower_better(period, 3, 6) if period != float("inf") else 10
    return result("Discounted Payback Period", score, {"Years": period})


def profitability_index(initial_investment: float, cash_flows: list[float], discount_rate: float) -> dict:
    rate = discount_rate / 100
    pv = sum(cf / ((1 + rate) ** (idx + 1)) for idx, cf in enumerate(cash_flows))
    value = safe_financial_div(pv, initial_investment)
    score = score_higher_better(value, 1, 1.5)
    return result("Profitability Index", score, {"PI": value})


def analyze_capital_budget(initial_investment: float, cash_flows: list[float], discount_rate: float, target_irr: float) -> dict:
    metrics = {
        "npv": npv(initial_investment, cash_flows, discount_rate),
        "irr": irr(initial_investment, cash_flows, target_irr),
        "payback_period": payback_period(initial_investment, cash_flows),
        "discounted_payback_period": discounted_payback_period(initial_investment, cash_flows, discount_rate),
        "profitability_index": profitability_index(initial_investment, cash_flows, discount_rate),
    }
    return module_summary(metrics)
