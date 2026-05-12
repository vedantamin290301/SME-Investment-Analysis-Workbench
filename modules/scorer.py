from __future__ import annotations

from .utils import clamp_score


WEIGHTS = {
    "Income Statement": 0.20,
    "Working Capital": 0.10,
    "Balance Sheet": 0.20,
    "Cash Flow": 0.15,
    "Valuation": 0.15,
    "Capital Budgeting": 0.10,
    "Risk Analysis": 0.05,
    "IPO-Specific": 0.05,
}


def final_rating(score: float) -> tuple[str, str]:
    if score >= 75:
        return "INVEST", "#16a34a"
    if score >= 50:
        return "WATCH", "#d97706"
    return "AVOID", "#dc2626"


def score_investment(modules: dict[str, dict]) -> dict:
    total = 0.0
    for name, weight in WEIGHTS.items():
        total += modules.get(name, {}).get("score", 0) * weight
    score = clamp_score(total)
    rating, color = final_rating(score)
    flags = [flag for module in modules.values() for flag in module.get("flags", [])]
    return {"score": score, "rating": rating, "color": color, "module_scores": {k: v.get("score", 0) for k, v in modules.items()}, "flags": flags}
