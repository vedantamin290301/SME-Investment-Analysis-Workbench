from __future__ import annotations

import base64
import json
import os
from typing import Any


MODEL = "gpt-4.1-mini"


FIELD_SCHEMA = {
    "company": "Company or issuer name",
    "unit": "Reported unit, for example INR crore or INR lakh",
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
    "notes": "Assumptions, missing fields, negative values, units, and source quality",
}


def extract_financials_with_openai(
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    report_text: str = "",
    api_key: str | None = None,
) -> dict[str, Any]:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OpenAI API key not supplied.", "data": None}

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        content: list[dict[str, Any]] = [{"type": "input_text", "text": extraction_prompt(report_text)}]
        encoded = base64.b64encode(file_bytes).decode("utf-8")
        if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
            content.append({
                "type": "input_file",
                "filename": filename,
                "file_data": f"data:application/pdf;base64,{encoded}",
            })
        elif mime_type.startswith("image/"):
            content.append({
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{encoded}",
                "detail": "high",
            })
        else:
            return {"error": f"Unsupported file type for OpenAI extraction: {mime_type or filename}", "data": None}

        response = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": content}],
            temperature=0,
        )
        return {"error": None, "data": parse_json_response(response.output_text), "raw": response.output_text}
    except Exception as exc:
        return {"error": str(exc), "data": None}


def extraction_prompt(report_text: str) -> str:
    clipped = report_text[:60000]
    return f"""
You are a financial data extraction agent for an SME investment analysis app.

Extract the latest available single-year or 3-year financial statement data from the supplied document or image.

Return only valid JSON using this exact schema:
{json.dumps(FIELD_SCHEMA, indent=2)}

Rules:
- Use numeric values only. Preserve negative values when the source reports losses, negative cash flow, repayments, or outflows.
- Use financial year end as the year. FY 2024-25 should be 2025.
- Map revenue from operations / sales / total operating income to revenue.
- Map cost of materials, purchases, cost of goods sold, or cost of sales to cogs when available.
- Map PAT / profit after tax / profit for the year to net_profit.
- Map finance cost to interest_expense if interest is not separated.
- Map borrowings / debt to total_debt; non-current borrowings to long_term_debt.
- Map cash flow from operating activities to operating_cash_flow.
- Map purchase of PPE/fixed assets to capex as a positive capex number.
- If a field is missing, use 0 and explain in notes.
- If units differ across tables, normalize to one unit and state it in unit/notes.
- Prefer consolidated financials if both consolidated and standalone are shown; mention that choice in notes.

Locally extracted text, if any:
{clipped}
"""


def parse_json_response(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        stripped = stripped[start : end + 1]
    return json.loads(stripped)
