from __future__ import annotations

import json
import os


MODEL = "claude-sonnet-4-20250514"


def build_prompt(metrics: dict) -> str:
    return (
        "You are a fund manager's assistant. Given these SME IPO financial metrics, "
        "generate a structured 300-word investment memo: Bull case (3 points), "
        "Bear case (3 points), Key risks, and a final Invest/Watch/Avoid recommendation with reasoning.\n\n"
        f"Metrics JSON:\n{json.dumps(metrics, indent=2, default=str)}"
    )


def generate_memo(metrics: dict, api_key: str | None = None) -> str:
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    prompt = build_prompt(metrics)
    if not api_key:
        return fallback_memo(metrics)
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        message = client.messages.create(
            model=MODEL,
            max_tokens=800,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return "\n".join(block.text for block in message.content if getattr(block, "type", "") == "text")
    except Exception as exc:
        return f"{fallback_memo(metrics)}\n\nClaude API unavailable: {exc}"


def fallback_memo(metrics: dict) -> str:
    score = metrics.get("overall", {}).get("score", 0)
    rating = metrics.get("overall", {}).get("rating", "WATCH")
    flags = metrics.get("overall", {}).get("flags", [])[:5]
    modules = metrics.get("overall", {}).get("module_scores", {})
    best = sorted(modules.items(), key=lambda item: item[1], reverse=True)[:2]
    weak = sorted(modules.items(), key=lambda item: item[1])[:2]
    return (
        f"### Investment Memo\n\n"
        f"**Bull case**\n"
        f"1. Overall investment score is {score}/100, resulting in a {rating} framework rating.\n"
        f"2. Strongest modules: {', '.join(f'{k} ({v})' for k, v in best) or 'not available'}.\n"
        f"3. The model rewards sustained growth, cash conversion, prudent leverage, and positive capital budgeting metrics.\n\n"
        f"**Bear case**\n"
        f"1. Weakest modules: {', '.join(f'{k} ({v})' for k, v in weak) or 'not available'}.\n"
        f"2. SME IPO disclosures may be limited and restated financials should be verified against DRHP filings.\n"
        f"3. Valuation sensitivity is high where earnings quality, OCF, or promoter alignment is weak.\n\n"
        f"**Key risks**\n"
        f"{'; '.join(flags) if flags else 'No major automated red flags were triggered.'}\n\n"
        f"**Recommendation**\n"
        f"{rating}. Treat INVEST as an investable screening outcome, then validate management quality, auditor notes, order book, and peer valuation before allocation."
    )
