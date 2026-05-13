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
    "missing_fields": ["field names that were not found"],
    "field_sources": {"revenue": "short source line or table name"},
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
You are a strict financial statement extraction agent for an SME investment analysis app.

Extract ONLY the financial fields required by the app from the supplied document or image.
The app needs these exact fields:
revenue, cogs, ebitda, ebit, net_profit, total_assets, current_assets, current_liabilities,
net_fixed_assets, receivables, inventory, payables, interest_expense, principal_repayments,
operating_income, operating_cash_flow, total_debt, long_term_debt, cash, equity, goodwill,
capex, debt_raised.

Return only valid JSON using this exact schema:
{json.dumps(FIELD_SCHEMA, indent=2)}

Rules:
- First identify the company/issuer and the reporting unit.
- Prefer audited/restated/consolidated financial statements. If only standalone data is available, use it and say so in notes.
- Extract from P&L, balance sheet, cash flow statement, notes to borrowings, and fixed-asset/capex notes.
- Do not use narrative percentages or ratio tables as source values.
- Use numeric values only. Preserve negative values when the source reports losses, negative cash flow, repayments, or outflows.
- Use financial year end as the year. FY 2024-25 should be 2025.
- Map revenue from operations / sales / total operating income to revenue.
- Map cost of materials consumed + purchases + changes in inventory / cost of goods sold / cost of sales to cogs when available.
- If EBITDA is not directly reported, calculate EBITDA = EBIT + depreciation + amortisation when those values are available. Otherwise use 0 and list it missing.
- If EBIT is not directly reported, calculate EBIT = profit before tax + finance cost when possible.
- Map PAT / profit after tax / profit for the year to net_profit.
- Map finance cost to interest_expense if interest is not separated.
- Map borrowings / debt to total_debt; non-current borrowings to long_term_debt.
- Map trade receivables to receivables, inventories to inventory, trade payables to payables.
- Map total equity / net worth / shareholders' funds to equity.
- Map cash and cash equivalents plus bank balances to cash when appropriate.
- Map cash flow from operating activities to operating_cash_flow.
- Map purchase of PPE/fixed assets to capex as a positive capex number.
- Map proceeds from borrowings to debt_raised. Principal repayment should be positive when it represents repayment amount.
- If a field is missing, use 0 and explain in notes.
- If units differ across tables, normalize to one unit and state it in unit/notes.
- For every field you fill, add a short source clue in field_sources, e.g. "P&L: Revenue from operations".
- Put every unavailable field in missing_fields.
- Return up to the latest 3 years if present. If only one year is visible, return one row.

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
