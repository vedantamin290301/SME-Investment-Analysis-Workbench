from __future__ import annotations

import re

from .utils import module_summary, result, score_higher_better


RED_FLAG_PATTERNS = {
    "Qualified audit opinion": r"qualified opinion|qualification by auditor|basis for qualified",
    "Related party transactions": r"related party transaction|associate company|promoter group transaction",
    "High promoter pledging": r"pledge|encumbered shares|promoter pledg",
    "Aggressive revenue recognition": r"revenue recognition|unbilled revenue|contract asset",
}


def promoter_holding(post_ipo_holding: float) -> dict:
    flags = ["Promoter holding below 40% post-IPO."] if post_ipo_holding < 40 else []
    return result("Promoter Holding", score_higher_better(post_ipo_holding, 35, 60), {"Post-IPO %": post_ipo_holding}, flags)


def gmp_tracker(gmp_percent: float) -> dict:
    score = 70 if -5 <= gmp_percent <= 20 else 55 if gmp_percent > 20 else 40
    flags = ["Very high GMP can indicate speculative demand; do not anchor valuation on it."] if gmp_percent > 30 else []
    return result("Grey Market Premium", score, {"GMP %": gmp_percent}, flags)


def proceeds_score(use_of_proceeds: str) -> dict:
    text = use_of_proceeds.lower()
    score = 85
    flags = []
    if "expansion" in text or "capacity" in text or "growth" in text:
        score += 10
    if "debt repayment" in text and not any(word in text for word in ["expansion", "capacity", "working capital"]):
        score = 45
        flags.append("IPO proceeds appear focused only on debt repayment.")
    return result("Use of IPO Proceeds", score, {"Use": use_of_proceeds}, flags)


def drhp_red_flag_scanner(text: str) -> dict:
    found = [name for name, pattern in RED_FLAG_PATTERNS.items() if re.search(pattern, text or "", flags=re.I)]
    score = max(20, 100 - len(found) * 20)
    flags = [f"DRHP scanner hit: {item}." for item in found]
    return result("DRHP Red-Flag Scanner", score, {"Hits": found}, flags)


def analyze_ipo_checks(post_ipo_holding: float, gmp_percent: float, use_of_proceeds: str, drhp_text: str) -> dict:
    metrics = {
        "promoter_holding": promoter_holding(post_ipo_holding),
        "gmp_tracker": gmp_tracker(gmp_percent),
        "proceeds_score": proceeds_score(use_of_proceeds),
        "drhp_scanner": drhp_red_flag_scanner(drhp_text),
    }
    return module_summary(metrics)

