from __future__ import annotations

import json
import os
from typing import Any


MODEL = "claude-sonnet-4-20250514"


REQUIRED_SCHEMA = {
    "company": "Company or issuer name if available",
    "rows": [
        {
            "year": 2025,
            "revenue": 0,
            "cogs": 0,
            "ebitda": 0,
            "ebit": 0,
            "net_profit": 0,
            "total_assets": 0,
            "current_assets": 0,
            "current_liabilities": 0,
            "net_fixed_assets": 0,
            "receivables": 0,
            "inventory": 0,
            "payables": 0,
            "interest_expense": 0,
            "principal_repayments": 0,
            "operating_income": 0,
            "operating_cash_flow": 0,
            "total_debt": 0,
            "long_term_debt": 0,
            "cash": 0,
            "equity": 0,
            "goodwill": 0,
            "capex": 0,
            "debt_raised": 0,
        }
    ],
    "confidence": 0.0,
    "notes": "Short notes on assumptions, missing fields, units, and source quality",
}


def extract_financials_with_ai(report_text: str, api_key: str | None = None) -> dict[str, Any]:
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "Claude API key not supplied.", "data": None}

    prompt = build_extraction_prompt(report_text)
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        message = client.messages.create(
            model=MODEL,
            max_tokens=3500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "\n".join(block.text for block in message.content if getattr(block, "type", "") == "text")
        return {"error": None, "data": parse_json_response(text), "raw": text}
    except Exception as exc:
        return {"error": str(exc), "data": None}


def build_extraction_prompt(report_text: str) -> str:
    clipped = report_text[:120000]
    return f"""
You are a financial statement extraction agent for an SME IPO fund manager.

Task:
Extract the latest 3 years of restated financial data from the report text and return only valid JSON.

Rules:
- Use the exact JSON schema and field names shown below.
- Values must be numeric and in the same units used by the report, preferably INR crore or INR lakh if that is the report unit.
- If a field is not available, use 0 and explain it in notes.
- Map "revenue from operations" or "total income from operations" to revenue.
- Map "PAT", "profit after tax", or "profit for the year" to net_profit.
- Map "finance cost" to interest_expense when interest is not separately disclosed.
- Map "borrowings" to total_debt, and non-current borrowings to long_term_debt.
- Map "cash flow from operating activities" to operating_cash_flow.
- Map "purchase of property, plant and equipment" or fixed asset purchase to capex.
- Return years as calendar year integers matching the financial year end. For FY 2024-25, use 2025.
- Return exactly JSON. Do not wrap it in markdown.

Schema:
{json.dumps(REQUIRED_SCHEMA, indent=2)}

Report text:
{clipped}
"""


def parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        stripped = stripped[start : end + 1]
    return json.loads(stripped)
