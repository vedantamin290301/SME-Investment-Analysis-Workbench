from __future__ import annotations

import numpy as np

from .utils import module_summary, result, safe_div, score_lower_better


def expected_value(cash_flows: list[float], probabilities: list[float]) -> float:
    total_p = sum(probabilities) or 1
    probs = [p / total_p for p in probabilities]
    return float(sum(cf * p for cf, p in zip(cash_flows, probs)))


def variance(cash_flows: list[float], probabilities: list[float]) -> dict:
    ev = expected_value(cash_flows, probabilities)
    total_p = sum(probabilities) or 1
    probs = [p / total_p for p in probabilities]
    value = float(sum(((cf - ev) ** 2) * p for cf, p in zip(cash_flows, probs)))
    score = score_lower_better(safe_div(value, abs(ev) ** 2 if ev else 1), 0.05, 0.5)
    return result("Variance", score, {"Variance": value, "Expected Cash Flow": ev})


def standard_deviation(cash_flows: list[float], probabilities: list[float]) -> dict:
    var = variance(cash_flows, probabilities)["values"]["Variance"]
    value = float(np.sqrt(var))
    ev = expected_value(cash_flows, probabilities)
    score = score_lower_better(safe_div(value, abs(ev) if ev else 1), 0.2, 0.8)
    return result("Standard Deviation", score, {"Std Dev": value})


def coefficient_variation(cash_flows: list[float], probabilities: list[float]) -> dict:
    ev = expected_value(cash_flows, probabilities)
    sd = standard_deviation(cash_flows, probabilities)["values"]["Std Dev"]
    value = safe_div(sd, abs(ev))
    return result("Coefficient of Variation", score_lower_better(value, 0.25, 1.0), {"CV": value})


def analyze_risk(cash_flows: list[float], probabilities: list[float]) -> dict:
    metrics = {
        "variance": variance(cash_flows, probabilities),
        "standard_deviation": standard_deviation(cash_flows, probabilities),
        "coefficient_variation": coefficient_variation(cash_flows, probabilities),
    }
    return module_summary(metrics)

