from __future__ import annotations

from math import isfinite
from typing import Iterable

import numpy as np
import pandas as pd


RAG_COLORS = {"good": "#16a34a", "watch": "#d97706", "bad": "#dc2626", "neutral": "#64748b"}


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    try:
        if denominator in (0, None) or pd.isna(denominator):
            return default
        value = numerator / denominator
        return float(value) if isfinite(value) else default
    except Exception:
        return default


def latest(series: Iterable[float]) -> float:
    values = list(series)
    return float(values[-1]) if values else 0.0


def average_pair(values: list[float], idx: int) -> float:
    if idx <= 0:
        return float(values[idx])
    return (float(values[idx]) + float(values[idx - 1])) / 2


def pct_change(current: float, previous: float) -> float:
    return safe_div(current - previous, previous) * 100


def cagr(values: list[float]) -> float:
    if len(values) < 2 or values[0] <= 0:
        return 0.0
    years = len(values) - 1
    return ((values[-1] / values[0]) ** (1 / years) - 1) * 100


def trend(values: list[float]) -> str:
    if len(values) < 2:
        return "flat"
    diffs = np.diff(values)
    if np.all(diffs > 0):
        return "rising"
    if np.all(diffs < 0):
        return "falling"
    if values[-1] > values[0]:
        return "improving"
    if values[-1] < values[0]:
        return "deteriorating"
    return "flat"


def score_higher_better(value: float, low: float, high: float) -> float:
    if value <= low:
        return 20.0
    if value >= high:
        return 100.0
    return 20 + (value - low) / (high - low) * 80


def score_lower_better(value: float, good: float, bad: float) -> float:
    if value <= good:
        return 100.0
    if value >= bad:
        return 20.0
    return 100 - (value - good) / (bad - good) * 80


def clamp_score(value: float) -> float:
    return round(max(0.0, min(100.0, float(value))), 1)


def rag_from_score(score: float) -> str:
    if score >= 75:
        return "good"
    if score >= 50:
        return "watch"
    return "bad"


def result(name: str, score: float, values: dict, flags: list[str] | None = None) -> dict:
    score = clamp_score(score)
    return {"name": name, "score": score, "values": values, "flags": flags or [], "status": rag_from_score(score)}


def module_summary(results: dict[str, dict]) -> dict:
    scores = [item["score"] for item in results.values()]
    flags = [flag for item in results.values() for flag in item.get("flags", [])]
    return {"score": clamp_score(np.mean(scores) if scores else 0), "metrics": results, "flags": flags}


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col != "Metric"]

