from __future__ import annotations

import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.ai_narrative import generate_memo
from modules.ai_extractor import extract_financials_with_ai
from modules.balance import analyze_balance_sheet
from modules.capital_budget import analyze_capital_budget
from modules.cashflow import analyze_cash_flow
from modules.income import analyze_income_and_working_capital
from modules.ipo_checks import analyze_ipo_checks
from modules.openai_extractor import extract_financials_with_openai
from modules.report import build_pdf_report
from modules.risk import analyze_risk
from modules.scorer import score_investment
from modules.valuation import analyze_valuation


DATA_DIR = Path(__file__).parent / "data"
REQUIRED_COLUMNS = [
    "year",
    "revenue",
    "cogs",
    "ebitda",
    "ebit",
    "net_profit",
    "total_assets",
    "current_assets",
    "current_liabilities",
    "net_fixed_assets",
    "receivables",
    "inventory",
    "payables",
    "interest_expense",
    "principal_repayments",
    "operating_income",
    "operating_cash_flow",
    "total_debt",
    "long_term_debt",
    "cash",
    "equity",
    "goodwill",
    "capex",
    "debt_raised",
]
COMPANY_COLUMNS = ["company", "firm", "company_name", "issuer", "name"]
PDF_FINANCIAL_FIELD_ALIASES = {
    "revenue": ["revenue", "total revenue", "income from operations", "revenue from operations", "sales", "net sales", "operating income"],
    "cogs": ["cost of goods sold", "cost of materials consumed", "purchase of stock", "cost of sales", "raw material consumed"],
    "ebitda": ["ebitda", "earnings before interest tax depreciation amortisation", "earnings before interest tax depreciation amortization"],
    "ebit": ["ebit", "profit before interest and tax", "operating profit"],
    "net_profit": ["net profit", "profit after tax", "pat", "profit for the year", "profit for period"],
    "total_assets": ["total assets"],
    "current_assets": ["current assets", "total current assets"],
    "current_liabilities": ["current liabilities", "total current liabilities"],
    "net_fixed_assets": ["net fixed assets", "property plant and equipment", "ppe", "fixed assets"],
    "receivables": ["trade receivables", "accounts receivable", "receivables"],
    "inventory": ["inventories", "inventory"],
    "payables": ["trade payables", "accounts payable", "payables"],
    "interest_expense": ["finance costs", "interest expense", "interest cost"],
    "principal_repayments": ["principal repayment", "repayment of borrowings", "repayment of debt"],
    "operating_income": ["operating income", "ebit", "profit before interest and tax"],
    "operating_cash_flow": ["net cash generated from operating activities", "cash flow from operating activities", "operating cash flow", "cash generated from operations"],
    "total_debt": ["total debt", "borrowings", "total borrowings", "debt"],
    "long_term_debt": ["long term borrowings", "non-current borrowings", "long term debt"],
    "cash": ["cash and cash equivalents", "cash equivalents", "cash"],
    "equity": ["equity", "shareholders funds", "net worth", "total equity"],
    "goodwill": ["goodwill"],
    "capex": ["capital expenditure", "purchase of property plant and equipment", "purchase of fixed assets", "capex"],
    "debt_raised": ["proceeds from borrowings", "debt raised", "borrowings availed", "proceeds from long term borrowings"],
}


st.set_page_config(page_title="SME Investment Analysis Workbench", layout="wide", initial_sidebar_state="expanded")


def main() -> None:
    inject_css()
    benchmarks = load_benchmarks()

    with st.sidebar:
        st.title("Fund Parameters")
        target_irr = st.slider("Target IRR (%)", 10.0, 50.0, 25.0, 0.5)
        horizon = st.slider("Investment horizon (years)", 1, 7, 4)
        sector = st.selectbox("Sector filter", list(benchmarks.keys()))
        max_de = st.number_input("Max Debt/Equity threshold", min_value=0.0, value=1.5, step=0.1)
        min_roce = st.number_input("Min ROCE floor (%)", min_value=0.0, value=12.0, step=0.5)
        discount_rate = st.slider("Discount rate (%)", 5.0, 35.0, 15.0, 0.5)
        cost_of_debt = st.slider("Cost of debt (%)", 5.0, 24.0, 12.0, 0.5)

        st.divider()
        company_scope = st.radio("Company analysis", ["Single company", "Multi company"])
        period_mode = st.radio("Financial period", ["Multi year", "Single year"])
        analysis_years = 1 if period_mode == "Single year" else 3
        if company_scope == "Single company":
            input_method = st.radio("Input source", ["Manual / sample", "CSV upload", "PDF upload", "Ticker lookup"])
        else:
            input_method = st.radio("Input source", ["Manual / sample portfolio", "CSV/PDF upload"])
        anthropic_key = st.text_input("Claude API key", type="password", help="Optional. If blank, the app shows a deterministic memo.")
        openai_key = st.text_input("OpenAI API key", type="password", help="Optional. Used for ChatGPT document/photo extraction.")
        extraction_mode = st.selectbox("Document/photo extraction mode", ["Heuristic autoscan", "AI agent autoscan", "Hybrid autoscan"], index=2)

    st.title("SME Investment Analysis Workbench")
    st.caption("For professional pre-IPO and SME IPO screening across growth, margins, working capital, leverage, cash flow, valuation, capital budgeting, risk, and IPO-specific red flags.")

    firms = collect_inputs(company_scope, input_method, horizon, anthropic_key, openai_key, extraction_mode, analysis_years)
    assumptions = firm_assumptions_editor(firms, sector)
    common_growth = st.slider("Portfolio projected annual FCF growth (%)", -20.0, 60.0, 18.0, 1.0)

    analyses = analyze_firms(
        firms=firms,
        assumptions=assumptions,
        benchmarks_by_sector=benchmarks,
        default_sector=sector,
        analysis_years=analysis_years,
        horizon=horizon,
        common_growth=common_growth,
        cost_of_debt=cost_of_debt,
        target_irr=target_irr,
        discount_rate=discount_rate,
        max_de=max_de,
        min_roce=min_roce,
    )

    portfolio_df = portfolio_summary(analyses)
    st.subheader("Multi-Firm Investment Ranking")
    st.dataframe(portfolio_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download multi-firm scorecard CSV",
        data=portfolio_df.to_csv(index=False).encode("utf-8"),
        file_name="sme_ipo_multi_firm_scorecard.csv",
        mime="text/csv",
    )

    selected_company = st.selectbox("Select firm for detailed analysis", portfolio_df["Company"].tolist())
    selected = analyses[selected_company]
    source_firm = next((firm for firm in firms if firm["company"] == selected_company), {})
    if source_firm.get("extraction_note"):
        st.info(source_firm["extraction_note"])
    with st.expander(f"Restated financials: {selected_company}", expanded=False):
        st.caption("Review autoscan or AI-agent extracted values here. In single-year mode, only the latest year is scored; in multi-year mode, up to three years are scored.")
        edited = st.data_editor(selected["financials"], use_container_width=True, num_rows="fixed", key=f"financials_{selected_company}")
        edited_financials = ensure_columns(edited, analysis_years)
        if not edited_financials.equals(selected["financials"]):
            updated = analyze_firms(
                firms=[{"company": selected_company, "financials": edited_financials, "drhp_text": source_firm.get("drhp_text", "")}],
                assumptions=assumptions[assumptions["Company"] == selected_company],
                benchmarks_by_sector=benchmarks,
                default_sector=sector,
                analysis_years=analysis_years,
                horizon=horizon,
                common_growth=common_growth,
                cost_of_debt=cost_of_debt,
                target_irr=target_irr,
                discount_rate=discount_rate,
                max_de=max_de,
                min_roce=min_roce,
            )[selected_company]
            analyses[selected_company] = updated
            selected = updated
            portfolio_df = portfolio_summary(analyses)

    render_dashboard(
        selected_company,
        selected["financials"],
        selected["analysis"],
        anthropic_key,
        portfolio_df=portfolio_df,
    )


def collect_inputs(
    company_scope: str,
    input_method: str,
    horizon: int,
    anthropic_key: str | None,
    openai_key: str | None,
    extraction_mode: str,
    analysis_years: int,
) -> list[dict[str, Any]]:
    drhp_text = ""
    if company_scope == "Multi company" and input_method == "CSV/PDF upload":
        return collect_multi_firm_inputs(anthropic_key, openai_key, extraction_mode, analysis_years)
    if company_scope == "Multi company":
        return [trim_firm_years(firm, analysis_years) for firm in sample_firms()]

    if input_method == "CSV upload":
        upload = st.file_uploader("Upload financial CSV", type=["csv"])
        if upload:
            return firms_from_csv(pd.read_csv(upload), upload.name.rsplit(".", 1)[0], analysis_years)
        st.info("Upload a CSV with the required financial columns. Sample data is loaded until then.")
    elif input_method == "PDF upload":
        upload = st.file_uploader("Upload DRHP, financial statement PDF, or financial photo", type=["pdf", "png", "jpg", "jpeg", "webp"])
        if upload:
            drhp_text = parse_pdf_text(upload) if is_pdf(upload) else ""
            extracted = extract_financials_from_document(upload, drhp_text, anthropic_key, openai_key, extraction_mode)
            st.success(f"Parsed {len(drhp_text):,} characters from PDF for red-flag scanning.")
            st.text_area("Parsed PDF text preview", drhp_text[:4000], height=160)
            if extracted["confidence"] > 0:
                st.info(f"Autoscan extracted {extracted['matched_fields']} financial fields across {extracted['matched_years']} year(s). Review the table before relying on the score.")
                st.dataframe(extracted["financials"], use_container_width=True, hide_index=True)
                return [{
                    "company": upload.name.rsplit(".", 1)[0],
                    "financials": ensure_columns(extracted["financials"], analysis_years),
                    "drhp_text": drhp_text,
                    "extraction_note": extracted["note"],
                }]
        st.info("PDF extraction is used for red-flag scanning and financial autoscan. If tables are not machine-readable, sample rows are loaded for manual editing.")
        return [{"company": "PDF Uploaded Firm", "financials": ensure_columns(sample_financials(), analysis_years), "drhp_text": drhp_text}]
    elif input_method == "Ticker lookup":
        ticker = st.text_input("NSE/BSE ticker", value="RELIANCE.NS")
        if st.button("Fetch public-market context"):
            fetched = fetch_ticker_snapshot(ticker)
            st.json(fetched)
        st.info("Ticker lookup gives market context via yfinance where available. SME IPO or pre-IPO financials still need manual restated entries.")

    return [trim_firm_years(firm, analysis_years) for firm in sample_firms()[:1]]


def collect_multi_firm_inputs(anthropic_key: str | None, openai_key: str | None, extraction_mode: str, analysis_years: int) -> list[dict[str, Any]]:
    st.markdown("Upload multiple firm documents. CSV files provide financial rows; PDF files add DRHP text and optional financial autoscan.")
    csv_uploads = st.file_uploader(
        "Upload one combined CSV or multiple firm CSVs",
        type=["csv"],
        accept_multiple_files=True,
        help="A combined CSV can include a company/firm column. Separate CSV files use the file name as the firm name.",
    )
    pdf_uploads = st.file_uploader("Upload DRHP / financial statement PDFs or financial photos", type=["pdf", "png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

    firms: list[dict[str, Any]] = []
    if csv_uploads:
        for upload in csv_uploads:
            firms.extend(firms_from_csv(pd.read_csv(upload), upload.name.rsplit(".", 1)[0], analysis_years))

    pdf_text_by_name = {}
    pdf_financials_by_name = {}
    if pdf_uploads:
        for upload in pdf_uploads:
            name = upload.name.rsplit(".", 1)[0]
            text = parse_pdf_text(upload) if is_pdf(upload) else ""
            pdf_text_by_name[name] = text
            extracted = extract_financials_from_document(upload, text, anthropic_key, openai_key, extraction_mode)
            if extracted["confidence"] > 0:
                pdf_financials_by_name[name] = extracted
        st.success(f"Parsed {len(pdf_text_by_name)} PDF document(s) for DRHP red-flag scanning.")
        if pdf_financials_by_name:
            st.info(f"Autoscan extracted financial tables from {len(pdf_financials_by_name)} PDF document(s).")

    if not firms and pdf_text_by_name:
        firms = []
        for name, text in pdf_text_by_name.items():
            extracted = pdf_financials_by_name.get(name)
            financials = extracted["financials"] if extracted else sample_financials()
            firms.append({
                "company": name,
                "financials": ensure_columns(financials, analysis_years),
                "drhp_text": text,
                "extraction_note": extracted["note"] if extracted else "No machine-readable financial table was detected; sample rows loaded.",
            })
        if any(name not in pdf_financials_by_name for name in pdf_text_by_name):
            st.info("Some PDFs did not expose machine-readable financial tables. Edit those generated financial rows in the firm detail table.")

    for firm in firms:
        matched_text = pdf_text_by_name.get(firm["company"], "")
        if not matched_text:
            matched_text = first_matching_pdf_text(firm["company"], pdf_text_by_name)
        firm["drhp_text"] = f"{firm.get('drhp_text', '')}\n{matched_text}".strip()

    if not firms:
        st.info("Upload firm CSV/PDF documents. Sample multi-firm data is loaded until then.")
        return [trim_firm_years(firm, analysis_years) for firm in sample_firms()]
    return dedupe_firms(firms, analysis_years)


def firms_from_csv(df: pd.DataFrame, fallback_name: str, analysis_years: int = 3) -> list[dict[str, Any]]:
    company_col = next((col for col in df.columns if col.lower().strip() in COMPANY_COLUMNS), None)
    if company_col:
        firms = []
        for company, group in df.groupby(company_col):
            firms.append({"company": str(company), "financials": ensure_columns(group.drop(columns=[company_col]), analysis_years), "drhp_text": ""})
        return firms
    return [{"company": fallback_name, "financials": ensure_columns(df, analysis_years), "drhp_text": ""}]


def first_matching_pdf_text(company: str, pdf_text_by_name: dict[str, str]) -> str:
    company_key = company.lower().replace(" ", "")
    for name, text in pdf_text_by_name.items():
        if company_key in name.lower().replace(" ", "") or name.lower().replace(" ", "") in company_key:
            return text
    return ""


def dedupe_firms(firms: list[dict[str, Any]], analysis_years: int = 3) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    output = []
    for firm in firms:
        name = firm["company"].strip() or "Unnamed Firm"
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name} ({seen[name]})"
        output.append({
            "company": name,
            "financials": ensure_columns(firm["financials"], analysis_years),
            "drhp_text": firm.get("drhp_text", ""),
            "extraction_note": firm.get("extraction_note", ""),
        })
    return output


def trim_firm_years(firm: dict[str, Any], analysis_years: int) -> dict[str, Any]:
    return {
        **firm,
        "financials": ensure_columns(firm["financials"], analysis_years),
    }


def parse_pdf_text(uploaded_file) -> str:
    try:
        import pdfplumber

        text = []
        with pdfplumber.open(BytesIO(uploaded_file.getvalue())) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)
    except Exception as exc:
        st.warning(f"PDF parsing failed: {exc}")
        return ""


def extract_financials_from_pdf(uploaded_file) -> dict[str, Any]:
    if not is_pdf(uploaded_file):
        return empty_extraction("Heuristic autoscan supports PDFs only. Use OpenAI extraction for photos/images.")
    raw_rows = extract_pdf_table_rows(uploaded_file)
    extracted = extract_values_from_rows(raw_rows)
    financials = financials_from_extracted_values(extracted)
    matched_fields = sum(1 for col in REQUIRED_COLUMNS if col != "year" and financials[col].abs().sum() > 0)
    matched_years = int(financials["year"].nunique()) if not financials.empty else 0
    confidence = min(1.0, matched_fields / 16) if matched_years else 0.0
    note = (
        f"Autoscan matched {matched_fields} required fields across {matched_years} year(s). "
        "Review values because DRHP PDFs vary in table structure, units, and row labels."
    )
    return {
        "financials": financials,
        "matched_fields": matched_fields,
        "matched_years": matched_years,
        "confidence": confidence,
        "note": note,
    }


def extract_financials_from_document(uploaded_file, report_text: str, anthropic_key: str | None, openai_key: str | None, extraction_mode: str) -> dict[str, Any]:
    heuristic = extract_financials_from_pdf(uploaded_file)
    if extraction_mode == "Heuristic autoscan":
        return heuristic

    openai_result = extract_financials_with_openai(
        uploaded_file.getvalue(),
        uploaded_file.name,
        uploaded_file.type or guess_mime_type(uploaded_file.name),
        report_text,
        openai_key,
    )
    if not openai_result.get("error"):
        openai_extraction = extraction_from_ai_data(openai_result.get("data") or {}, "ChatGPT/OpenAI")
        if extraction_mode == "AI agent autoscan":
            return openai_extraction
        if openai_extraction["matched_fields"] >= heuristic["matched_fields"]:
            openai_extraction["note"] = f"Hybrid autoscan used ChatGPT/OpenAI extraction. {openai_extraction['note']}"
            return openai_extraction

    ai_result = extract_financials_with_ai(report_text, anthropic_key)
    if ai_result.get("error"):
        if extraction_mode == "AI agent autoscan":
            st.warning(f"AI extraction unavailable. OpenAI: {openai_result.get('error')}; Claude: {ai_result['error']}. Falling back to heuristic autoscan.")
        return heuristic

    ai_data = ai_result.get("data") or {}
    ai_financials = financials_from_ai_data(ai_data)
    matched_fields = sum(1 for col in REQUIRED_COLUMNS if col != "year" and ai_financials[col].abs().sum() > 0)
    matched_years = int(ai_financials["year"].nunique()) if not ai_financials.empty else 0
    ai_confidence = float(ai_data.get("confidence") or 0)
    ai_note = (
        f"AI agent extracted {matched_fields} fields across {matched_years} year(s). "
        f"Model confidence: {ai_confidence:.0%}. Notes: {ai_data.get('notes', 'Review extracted values.')}"
    )

    if extraction_mode == "AI agent autoscan":
        return {
            "financials": ai_financials,
            "matched_fields": matched_fields,
            "matched_years": matched_years,
            "confidence": ai_confidence or min(1.0, matched_fields / 16),
            "note": ai_note,
        }

    if matched_fields >= heuristic["matched_fields"]:
        return {
            "financials": ai_financials,
            "matched_fields": matched_fields,
            "matched_years": matched_years,
            "confidence": ai_confidence or min(1.0, matched_fields / 16),
            "note": f"Hybrid autoscan used AI extraction. {ai_note}",
        }
    heuristic["note"] = f"Hybrid autoscan used heuristic extraction because it matched more fields. {heuristic['note']}"
    return heuristic


def extraction_from_ai_data(ai_data: dict[str, Any], provider: str) -> dict[str, Any]:
    financials = financials_from_ai_data(ai_data)
    matched_fields = sum(1 for col in REQUIRED_COLUMNS if col != "year" and financials[col].abs().sum() > 0)
    matched_years = int(financials["year"].nunique()) if not financials.empty else 0
    confidence = float(ai_data.get("confidence") or min(1.0, matched_fields / 16))
    note = (
        f"{provider} extracted {matched_fields} fields across {matched_years} year(s). "
        f"Model confidence: {confidence:.0%}. Unit: {ai_data.get('unit', 'not specified')}. "
        f"Notes: {ai_data.get('notes', 'Review extracted values.')}"
    )
    return {
        "financials": financials,
        "matched_fields": matched_fields,
        "matched_years": matched_years,
        "confidence": confidence,
        "note": note,
    }


def empty_extraction(note: str) -> dict[str, Any]:
    return {"financials": ensure_columns(pd.DataFrame()), "matched_fields": 0, "matched_years": 0, "confidence": 0.0, "note": note}


def is_pdf(uploaded_file) -> bool:
    return (uploaded_file.type == "application/pdf") or uploaded_file.name.lower().endswith(".pdf")


def guess_mime_type(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return "application/pdf"
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower.endswith(".webp"):
        return "image/webp"
    return "application/octet-stream"


def financials_from_ai_data(ai_data: dict[str, Any]) -> pd.DataFrame:
    rows = ai_data.get("rows") or []
    if not isinstance(rows, list) or not rows:
        return ensure_columns(pd.DataFrame())
    return ensure_columns(pd.DataFrame(rows))


def extract_pdf_table_rows(uploaded_file) -> list[list[str]]:
    try:
        import pdfplumber

        rows: list[list[str]] = []
        with pdfplumber.open(BytesIO(uploaded_file.getvalue())) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables() or []
                for table in tables:
                    for row in table or []:
                        cleaned = [clean_cell(cell) for cell in row if clean_cell(cell)]
                        if len(cleaned) >= 2:
                            rows.append(cleaned)
                text = page.extract_text() or ""
                rows.extend(rows_from_plain_text(text))
        return rows
    except Exception as exc:
        st.warning(f"Financial autoscan failed: {exc}")
        return []


def rows_from_plain_text(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        if not re.search(r"\d", line):
            continue
        parts = re.split(r"\s{2,}|\t+", line.strip())
        if len(parts) >= 2:
            rows.append([clean_cell(part) for part in parts if clean_cell(part)])
    return rows


def extract_values_from_rows(rows: list[list[str]]) -> dict[str, dict[int, float]]:
    extracted: dict[str, dict[int, float]] = {field: {} for field in REQUIRED_COLUMNS if field != "year"}
    active_years: list[int] = []
    for row in rows:
        row_text = " ".join(row)
        years = extract_years(row)
        if len(years) >= 2:
            active_years = years[-3:]
            continue

        field = match_financial_field(row_text)
        if not field:
            continue

        numbers = extract_numbers(row)
        if not numbers:
            continue

        row_years = years or active_years
        if not row_years:
            row_years = infer_recent_years(len(numbers))

        values = align_numbers_to_years(numbers, row_years)
        for year, value in values.items():
            extracted[field][year] = value
    return extracted


def financials_from_extracted_values(extracted: dict[str, dict[int, float]]) -> pd.DataFrame:
    years = sorted({year for values in extracted.values() for year in values})
    if not years:
        years = [2023, 2024, 2025]
    years = years[-3:]
    rows = []
    for year in years:
        row = {"year": year}
        for col in REQUIRED_COLUMNS:
            if col != "year":
                row[col] = extracted.get(col, {}).get(year, 0.0)
        if row["operating_income"] == 0:
            row["operating_income"] = row["ebit"]
        if row["long_term_debt"] == 0:
            row["long_term_debt"] = row["total_debt"]
        if row["cogs"] == 0 and row["revenue"] and row["ebitda"]:
            row["cogs"] = max(row["revenue"] - row["ebitda"], 0)
        rows.append(row)
    return ensure_columns(pd.DataFrame(rows))


def match_financial_field(text: str) -> str | None:
    normalized = normalize_label(text)
    for field, aliases in PDF_FINANCIAL_FIELD_ALIASES.items():
        for alias in aliases:
            if normalize_label(alias) in normalized:
                return field
    return None


def align_numbers_to_years(numbers: list[float], years: list[int]) -> dict[int, float]:
    if not years:
        return {}
    values = numbers[-len(years):]
    if len(values) < len(years):
        values = ([0.0] * (len(years) - len(values))) + values
    return dict(zip(years[-len(values):], values))


def infer_recent_years(count: int) -> list[int]:
    base = 2025
    count = min(max(count, 1), 3)
    return list(range(base - count + 1, base + 1))


def extract_years(row: list[str]) -> list[int]:
    years = []
    for cell in row:
        for match in re.findall(r"(?:FY\s*)?(20\d{2})|(?:31\s+Mar(?:ch)?\s+)(20\d{2})", cell, flags=re.I):
            year = next((part for part in match if part), None)
            if year:
                years.append(int(year))
    return sorted(dict.fromkeys(years))


def extract_numbers(row: list[str]) -> list[float]:
    numbers = []
    for cell in row:
        cell_without_years = re.sub(r"(?:FY\s*)?20\d{2}|31\s+Mar(?:ch)?\s+20\d{2}", "", cell, flags=re.I)
        for token in re.findall(r"\(?-?\d[\d,]*(?:\.\d+)?\)?", cell_without_years):
            numbers.append(parse_number(token))
    return numbers


def parse_number(token: str) -> float:
    token = token.strip()
    negative = token.startswith("(") and token.endswith(")")
    token = token.strip("()").replace(",", "")
    try:
        value = float(token)
    except ValueError:
        value = 0.0
    return -value if negative else value


def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def clean_cell(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def fetch_ticker_snapshot(ticker: str) -> dict[str, Any]:
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "ticker": ticker,
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "trailing_pe": info.get("trailingPE"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "sector": info.get("sector"),
            "long_name": info.get("longName"),
        }
    except Exception as exc:
        return {"ticker": ticker, "error": str(exc)}


def projected_cash_flow_inputs(financials: pd.DataFrame, horizon: int) -> list[float]:
    base_fcf = max(float(financials["operating_cash_flow"].iloc[-1] - financials["capex"].iloc[-1]), 1.0)
    with st.expander("Capital budgeting cash flows", expanded=False):
        growth = st.slider("Projected annual FCF growth (%)", -20.0, 60.0, 18.0, 1.0)
        cash_flows = [base_fcf * ((1 + growth / 100) ** year) for year in range(1, horizon + 1)]
        cols = st.columns(min(horizon, 5))
        edited = []
        for idx, cf in enumerate(cash_flows):
            with cols[idx % len(cols)]:
                edited.append(st.number_input(f"Year {idx + 1} cash flow", value=float(round(cf, 2)), step=10.0))
    return edited


def scenario_inputs(projected_cash_flows: list[float]) -> tuple[list[float], list[float]]:
    expected = float(np.mean(projected_cash_flows)) if projected_cash_flows else 0.0
    with st.expander("Risk scenario cash flows", expanded=False):
        bear = st.number_input("Bear case cash flow", value=float(round(expected * 0.65, 2)), step=10.0)
        base = st.number_input("Base case cash flow", value=float(round(expected, 2)), step=10.0)
        bull = st.number_input("Bull case cash flow", value=float(round(expected * 1.35, 2)), step=10.0)
        p_bear = st.slider("Bear probability", 0.0, 1.0, 0.25, 0.05)
        p_base = st.slider("Base probability", 0.0, 1.0, 0.50, 0.05)
        p_bull = max(0.0, 1.0 - p_bear - p_base)
        st.metric("Bull probability", f"{p_bull:.0%}")
    return [bear, base, bull], [p_bear, p_base, p_bull]


def firm_assumptions_editor(firms: list[dict[str, Any]], default_sector: str) -> pd.DataFrame:
    rows = []
    for firm in firms:
        financials = ensure_columns(firm["financials"])
        latest_revenue = float(financials["revenue"].iloc[-1])
        rows.append({
            "Company": firm["company"],
            "Sector": default_sector,
            "Market Cap": round(latest_revenue * 2.2, 2),
            "Investment Amount": 100.0,
            "Promoter Holding %": 52.0,
            "GMP %": 8.0,
            "Use of Proceeds": "Capacity expansion, working capital, and partial debt repayment.",
        })
    st.subheader("Firm-Level IPO Assumptions")
    st.caption("Edit valuation and IPO fields for each firm before running the batch score.")
    return st.data_editor(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "Sector": st.column_config.SelectboxColumn("Sector", options=list(load_benchmarks().keys())),
            "Use of Proceeds": st.column_config.TextColumn("Use of Proceeds", width="large"),
        },
        key="firm_assumptions",
    )


def analyze_firms(
    firms: list[dict[str, Any]],
    assumptions: pd.DataFrame,
    benchmarks_by_sector: dict,
    default_sector: str,
    analysis_years: int,
    horizon: int,
    common_growth: float,
    cost_of_debt: float,
    target_irr: float,
    discount_rate: float,
    max_de: float,
    min_roce: float,
) -> dict[str, dict[str, Any]]:
    assumptions_by_company = assumptions.set_index("Company").to_dict("index") if "Company" in assumptions else {}
    output = {}
    for firm in firms:
        company = firm["company"]
        financials = ensure_columns(firm["financials"], analysis_years)
        row = assumptions_by_company.get(company, {})
        firm_sector = row.get("Sector", default_sector) or default_sector
        projected_cash_flows = auto_projected_cash_flows(financials, horizon, common_growth)
        scenario_cash_flows, scenario_probs = auto_scenarios(projected_cash_flows)
        analysis = run_analysis(
            financials=financials,
            benchmarks=benchmarks_by_sector.get(firm_sector, benchmarks_by_sector[default_sector]),
            sector=firm_sector,
            cost_of_debt=cost_of_debt,
            target_irr=target_irr,
            discount_rate=discount_rate,
            initial_investment=float(row.get("Investment Amount", 100.0) or 100.0),
            projected_cash_flows=projected_cash_flows,
            scenario_cash_flows=scenario_cash_flows,
            scenario_probs=scenario_probs,
            market_cap=float(row.get("Market Cap", financials["revenue"].iloc[-1] * 2.2) or 0),
            post_ipo_holding=float(row.get("Promoter Holding %", 52.0) or 0),
            gmp_percent=float(row.get("GMP %", 8.0) or 0),
            use_of_proceeds=str(row.get("Use of Proceeds", "")),
            drhp_text=firm.get("drhp_text", ""),
        )
        apply_fund_threshold_flags(analysis, financials, max_de, min_roce)
        output[company] = {
            "company": company,
            "sector": firm_sector,
            "financials": financials,
            "analysis": analysis,
            "projected_cash_flows": projected_cash_flows,
            "extraction_note": firm.get("extraction_note", ""),
        }
    return output


def auto_projected_cash_flows(financials: pd.DataFrame, horizon: int, growth: float) -> list[float]:
    base_fcf = max(float(financials["operating_cash_flow"].iloc[-1] - financials["capex"].iloc[-1]), 1.0)
    return [round(base_fcf * ((1 + growth / 100) ** year), 2) for year in range(1, horizon + 1)]


def auto_scenarios(projected_cash_flows: list[float]) -> tuple[list[float], list[float]]:
    expected = float(np.mean(projected_cash_flows)) if projected_cash_flows else 0.0
    return [expected * 0.65, expected, expected * 1.35], [0.25, 0.5, 0.25]


def apply_fund_threshold_flags(analysis: dict, financials: pd.DataFrame, max_de: float, min_roce: float) -> None:
    de_ratio = financials["total_debt"].iloc[-1] / max(financials["equity"].iloc[-1], 1)
    if de_ratio > max_de:
        analysis["overall"]["flags"].append(f"Debt/equity {de_ratio:.2f}x exceeds fund threshold of {max_de:.2f}x.")
    latest_roce = analysis["modules"]["Income Statement"]["metrics"]["roce"]["values"]["Latest %"]
    if latest_roce < min_roce:
        analysis["overall"]["flags"].append(f"ROCE {latest_roce:.1f}% is below fund floor of {min_roce:.1f}%.")


def portfolio_summary(analyses: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for company, bundle in analyses.items():
        financials = bundle["financials"]
        analysis = bundle["analysis"]
        overall = analysis["overall"]
        modules = overall["module_scores"]
        rows.append({
            "Company": company,
            "Sector": bundle["sector"],
            "Investment Score": overall["score"],
            "Rating": overall["rating"],
            "Revenue": round(float(financials["revenue"].iloc[-1]), 2),
            "PAT": round(float(financials["net_profit"].iloc[-1]), 2),
            "OCF": round(float(financials["operating_cash_flow"].iloc[-1]), 2),
            "Debt/Equity": round(float(financials["total_debt"].iloc[-1] / max(financials["equity"].iloc[-1], 1)), 2),
            "Red Flags": len(overall["flags"]),
            **{name: score for name, score in modules.items()},
        })
    return pd.DataFrame(rows).sort_values(["Investment Score", "Revenue"], ascending=[False, False]).reset_index(drop=True)


def run_analysis(**kwargs) -> dict:
    financials = kwargs["financials"]
    income, working = analyze_income_and_working_capital(financials, kwargs["benchmarks"], kwargs["sector"], kwargs["cost_of_debt"])
    modules = {
        "Income Statement": income,
        "Working Capital": working,
        "Balance Sheet": analyze_balance_sheet(financials, kwargs["benchmarks"]),
        "Cash Flow": analyze_cash_flow(financials, kwargs["benchmarks"]),
        "Valuation": analyze_valuation(financials, kwargs["benchmarks"], kwargs["market_cap"]),
        "Capital Budgeting": analyze_capital_budget(kwargs["initial_investment"], kwargs["projected_cash_flows"], kwargs["discount_rate"], kwargs["target_irr"]),
        "Risk Analysis": analyze_risk(kwargs["scenario_cash_flows"], kwargs["scenario_probs"]),
        "IPO-Specific": analyze_ipo_checks(kwargs["post_ipo_holding"], kwargs["gmp_percent"], kwargs["use_of_proceeds"], kwargs["drhp_text"]),
    }
    return {"modules": modules, "overall": score_investment(modules)}


def render_dashboard(company: str, financials: pd.DataFrame, analysis: dict, anthropic_key: str | None, portfolio_df: pd.DataFrame | None = None) -> None:
    overall = analysis["overall"]
    top_cols = st.columns([1.1, 1.3, 2.2])
    with top_cols[0]:
        st.markdown(f"<div class='score-badge' style='background:{overall['color']}'>{overall['rating']}</div>", unsafe_allow_html=True)
    with top_cols[1]:
        st.plotly_chart(gauge_chart(overall["score"], overall["color"]), use_container_width=True)
    with top_cols[2]:
        st.plotly_chart(radar_chart(overall["module_scores"]), use_container_width=True)

    st.subheader("Metric Cards")
    render_metric_cards(analysis)

    render_extracted_data_calculations(financials, analysis)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(line_chart(financials, "revenue", "Revenue Trend"), use_container_width=True)
    with chart_cols[1]:
        margins = financials.assign(
            gross_margin=(financials["revenue"] - financials["cogs"]) / financials["revenue"] * 100,
            ebitda_margin=financials["ebitda"] / financials["revenue"] * 100,
            pat_margin=financials["net_profit"] / financials["revenue"] * 100,
        )
        st.plotly_chart(multi_line_chart(margins, ["gross_margin", "ebitda_margin", "pat_margin"], "Margin Trends"), use_container_width=True)

    cf_cols = st.columns(2)
    with cf_cols[0]:
        st.plotly_chart(line_chart(financials, "operating_cash_flow", "Operating Cash Flow"), use_container_width=True)
    with cf_cols[1]:
        fcf = financials.assign(free_cash_flow=financials["operating_cash_flow"] - financials["capex"])
        st.plotly_chart(line_chart(fcf, "free_cash_flow", "Free Cash Flow"), use_container_width=True)

    st.subheader("Red Flag Log")
    if overall["flags"]:
        for flag in overall["flags"]:
            st.warning(flag)
    else:
        st.success("No automated red flags triggered.")

    render_interpretation(company, financials, analysis)

    st.subheader("Peer Comparison")
    st.dataframe(peer_comparison(company, analysis, portfolio_df), use_container_width=True, hide_index=True)

    payload = {"overall": overall, "modules": analysis["modules"]}
    with st.expander("AI Narrative", expanded=True):
        memo = generate_memo(payload, anthropic_key)
        st.markdown(memo)

    pdf = build_pdf_report(company, analysis, financials, memo)
    st.download_button("Download PDF report", data=pdf, file_name=f"{company.lower().replace(' ', '_')}_sme_ipo_report.pdf", mime="application/pdf")


def render_extracted_data_calculations(financials: pd.DataFrame, analysis: dict) -> None:
    st.subheader("Extracted Data & Calculations")
    with st.expander("Review source data and computed formula outputs", expanded=True):
        st.markdown("**Extracted / entered financial data**")
        st.dataframe(financials, use_container_width=True, hide_index=True)

        st.markdown("**Calculated financial indicators**")
        rows = []
        for module_name, module in analysis["modules"].items():
            for metric in module["metrics"].values():
                rows.append({
                    "Module": module_name,
                    "Indicator": metric["name"],
                    "Formula / Logic": formula_description(metric["name"]),
                    "Latest Value": compact_metric_value(metric),
                    "Score": metric["score"],
                    "Status": metric["status"].upper(),
                    "Warnings": "; ".join(metric.get("flags", [])),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def formula_description(name: str) -> str:
    formulas = {
        "Revenue Growth": "YoY growth and CAGR from revenue trend; handles negative prior-year base conservatively.",
        "Gross Margin": "(Revenue - COGS) / Revenue x 100",
        "EBITDA Margin": "EBITDA / Revenue x 100",
        "PAT Margin": "Net Profit / Revenue x 100",
        "ROCE": "EBIT / (Total Assets - Current Liabilities) x 100",
        "Asset Turnover": "Revenue / Average Total Assets",
        "Fixed Asset Turnover": "Revenue / Net Fixed Assets",
        "ROA": "Net Profit / Average Total Assets x 100",
        "Debtor Days": "Receivables / Revenue x 365",
        "Inventory Days": "Inventory / COGS x 365",
        "Creditor Days": "Payables / COGS x 365",
        "Inventory Turnover": "COGS / Average Inventory",
        "Cash Conversion Cycle": "DSO + DIO - DPO",
        "Current Ratio": "Current Assets / Current Liabilities",
        "Quick Ratio": "(Current Assets - Inventory) / Current Liabilities",
        "Interest Coverage": "EBIT / Interest Expense",
        "DSCR": "Operating Income / (Interest + Principal Repayments)",
        "Cash Debt Coverage": "Operating Cash Flow / Total Debt",
        "Net Debt-to-EBITDA": "(Total Debt - Cash) / EBITDA",
        "Debt Ratio": "Total Debt / Total Assets",
        "Long-Term Debt-to-Capitalization": "Long-Term Debt / (Long-Term Debt + Equity)",
        "Goodwill as % of Assets": "Goodwill / Total Assets x 100",
        "Operating Cash Flow": "Reported OCF trend; negative OCF is penalized.",
        "Cash Conversion Ratio": "Operating Cash Flow / Net Profit",
        "Free Cash Flow": "Operating Cash Flow - Capex; margin = FCF / Revenue x 100",
        "Capex Intensity": "Capex / Operating Cash Flow",
        "Financing Cash Flow - Debt Trend": "Repeated debt raised versus profit improvement",
        "P/S Ratio": "Market Cap / Revenue",
        "EV/EBITDA": "(Market Cap + Debt - Cash) / EBITDA",
        "P/B Ratio": "Market Cap / Book Equity; cross-checked with ROE",
        "NPV": "Sum of discounted cash flows - initial investment",
        "IRR": "Discount rate that sets NPV to zero",
        "Payback Period": "Time for undiscounted cash flows to recover investment",
        "Discounted Payback Period": "Time for discounted cash flows to recover investment",
        "Profitability Index": "PV of future cash flows / initial investment",
        "Variance": "Probability-weighted squared deviation from expected cash flow",
        "Standard Deviation": "Square root of variance",
        "Coefficient of Variation": "Standard deviation / expected cash flow",
        "Promoter Holding": "Post-IPO promoter holding threshold scoring",
        "Grey Market Premium": "Manual GMP input, treated as sentiment not valuation proof",
        "Use of IPO Proceeds": "Expansion/growth uses score better than debt-only uses",
        "DRHP Red-Flag Scanner": "Keyword scan for audit, RPT, pledging, and revenue recognition risks",
    }
    return formulas.get(name, "Model threshold scoring based on latest value and trend.")


def render_interpretation(company: str, financials: pd.DataFrame, analysis: dict) -> None:
    overall = analysis["overall"]
    module_scores = overall["module_scores"]
    best_modules = sorted(module_scores.items(), key=lambda item: item[1], reverse=True)[:3]
    weak_modules = sorted(module_scores.items(), key=lambda item: item[1])[:3]
    drivers = key_metric_drivers(analysis)

    st.subheader("Detailed Interpretation")
    with st.expander("Investment interpretation", expanded=True):
        if overall["rating"] == "INVEST":
            st.success(
                f"{company} clears the model's investable threshold with a score of {overall['score']}/100. "
                "The result indicates the financial profile is strong enough for deeper diligence and potential allocation, subject to valuation, governance, and document verification."
            )
        elif overall["rating"] == "WATCH":
            st.warning(
                f"{company} is in the watch zone with a score of {overall['score']}/100. "
                "The company has investable elements, but the model sees enough weakness or uncertainty to require more evidence before committing capital."
            )
        else:
            st.error(
                f"{company} is currently avoid-rated with a score of {overall['score']}/100. "
                "The financial profile does not clear the fund's screening threshold based on the uploaded data."
            )

        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Strongest areas**")
            for name, score in best_modules:
                st.write(f"- {name}: {score}/100 - {module_interpretation(name, score)}")
        with cols[1]:
            st.markdown("**Weakest areas**")
            for name, score in weak_modules:
                st.write(f"- {name}: {score}/100 - {module_interpretation(name, score)}")

        st.markdown("**Key ratio interpretation**")
        for item in drivers:
            st.write(f"- {item}")

        st.markdown("**Trend reading**")
        st.write(trend_interpretation(financials))

    render_module_logic_tabs(analysis)


def render_module_logic_tabs(analysis: dict) -> None:
    st.subheader("Analysis Logic")
    module_names = list(analysis["modules"].keys())
    tabs = st.tabs(module_names)
    for tab, module_name in zip(tabs, module_names):
        module = analysis["modules"][module_name]
        with tab:
            st.markdown(f"**{module_name} interpretation**")
            st.write(module_logic_summary(module_name, module))

            metrics_df = module_metric_table(module)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            st.markdown("**Metric-by-metric logic**")
            for metric in module["metrics"].values():
                st.write(f"- **{metric['name']}**: {metric_logic(metric)}")

            if module.get("flags"):
                st.markdown("**Triggered warnings**")
                for flag in module["flags"]:
                    st.warning(flag)
            else:
                st.success("No red flags triggered inside this module.")


def module_metric_table(module: dict) -> pd.DataFrame:
    rows = []
    for metric in module["metrics"].values():
        rows.append({
            "Metric": metric["name"],
            "Score": metric["score"],
            "Status": metric["status"].upper(),
            "Latest / Main Value": compact_metric_value(metric),
            "Flags": len(metric.get("flags", [])),
        })
    return pd.DataFrame(rows)


def compact_metric_value(metric: dict) -> str:
    values = metric.get("values", {})
    if not values:
        return "n/a"
    key, value = next(iter(values.items()))
    if isinstance(value, list):
        value = value[-1] if value else "n/a"
    if isinstance(value, float):
        value = round(value, 2)
    return f"{key}: {value}"


def module_logic_summary(module_name: str, module: dict) -> str:
    score = module["score"]
    flags = len(module.get("flags", []))
    if score >= 75:
        posture = "This module supports an investable outcome."
    elif score >= 50:
        posture = "This module is mixed and should be treated as a diligence area."
    else:
        posture = "This module is a material concern for allocation."

    explanations = {
        "Income Statement": "It evaluates whether the company is growing profitably, improving margins, and earning enough return on capital to justify expansion.",
        "Working Capital": "It checks whether reported growth is converting into cash or getting trapped in receivables, inventory, and supplier credit.",
        "Balance Sheet": "It measures liquidity, leverage, interest service capacity, and balance-sheet fragility.",
        "Cash Flow": "It tests whether accounting profit is backed by operating cash flow and whether capex is internally funded.",
        "Valuation": "It compares the asking valuation with sales, EBITDA, and book value to detect overpayment risk.",
        "Capital Budgeting": "It translates the investment into NPV, IRR, payback, and profitability index to judge fund-level return attractiveness.",
        "Risk Analysis": "It measures volatility and downside dispersion in expected cash-flow scenarios.",
        "IPO-Specific": "It checks promoter alignment, GMP context, use of proceeds, and DRHP red flags.",
    }
    return f"{posture} {explanations.get(module_name, '')} Module score: {score}/100. Triggered warnings: {flags}."


def metric_logic(metric: dict) -> str:
    score = metric["score"]
    status = metric["status"]
    value = compact_metric_value(metric)
    flags = metric.get("flags", [])

    if status == "good":
        base = "The metric is favorable versus the model threshold and strengthens the case."
    elif status == "watch":
        base = "The metric is acceptable but not decisive; it needs comparison with peers and management commentary."
    else:
        base = "The metric is weak versus the model threshold and reduces confidence in the investment case."

    flag_text = f" Warning: {'; '.join(flags)}" if flags else ""
    return f"{base} {value}. Score: {score}/100.{flag_text}"


def key_metric_drivers(analysis: dict) -> list[str]:
    modules = analysis["modules"]
    income = modules["Income Statement"]["metrics"]
    balance = modules["Balance Sheet"]["metrics"]
    cash = modules["Cash Flow"]["metrics"]
    valuation = modules["Valuation"]["metrics"]
    capital = modules["Capital Budgeting"]["metrics"]

    drivers = [
        describe_metric("Revenue Growth", income["revenue_growth"], "growth durability"),
        describe_metric("EBITDA Margin", income["ebitda_margin"], "operating profitability"),
        describe_metric("ROCE", income["roce"], "capital efficiency"),
        describe_metric("Net Debt-to-EBITDA", balance["net_debt_to_ebitda"], "leverage capacity"),
        describe_metric("Cash Conversion Ratio", cash["cash_conversion_ratio"], "earnings quality"),
        describe_metric("Free Cash Flow", cash["free_cash_flow"], "cash generation after capex"),
        describe_metric("EV/EBITDA", valuation["ev_ebitda"], "relative valuation"),
        describe_metric("IRR", capital["irr"], "fund return potential"),
    ]
    return drivers


def describe_metric(label: str, metric: dict, lens: str) -> str:
    status = metric["status"]
    score = metric["score"]
    values = metric.get("values", {})
    latest_value = next(iter(values.values()), "n/a")
    if isinstance(latest_value, list):
        latest_value = latest_value[-1] if latest_value else "n/a"
    if isinstance(latest_value, float):
        latest_value = round(latest_value, 2)

    if status == "good":
        tone = "supports the investment case"
    elif status == "watch":
        tone = "is acceptable but needs diligence"
    else:
        tone = "weakens the investment case"
    return f"{label} ({lens}) {tone}; latest value {latest_value}, score {score}/100."


def module_interpretation(module_name: str, score: float) -> str:
    if score >= 75:
        return "clear strength versus the model thresholds"
    if score >= 50:
        return "mixed but not disqualifying"
    return "material diligence concern"


def trend_interpretation(financials: pd.DataFrame) -> str:
    if len(financials) < 2:
        return "Single-year mode is active, so trend interpretation is limited. Treat the output as a snapshot and rely more heavily on document quality and peer comparison."

    revenue_start = financials["revenue"].iloc[0]
    revenue_end = financials["revenue"].iloc[-1]
    pat_start = financials["net_profit"].iloc[0]
    pat_end = financials["net_profit"].iloc[-1]
    ocf_start = financials["operating_cash_flow"].iloc[0]
    ocf_end = financials["operating_cash_flow"].iloc[-1]
    fcf = financials["operating_cash_flow"] - financials["capex"]

    revenue_text = "expanded" if revenue_end > revenue_start else "contracted or stayed flat"
    pat_text = "improved" if pat_end > pat_start else "weakened or stayed flat"
    ocf_text = "improved" if ocf_end > ocf_start else "weakened or stayed flat"
    fcf_text = "positive in the latest year" if fcf.iloc[-1] > 0 else "negative in the latest year"
    return (
        f"Revenue {revenue_text} across the uploaded period, PAT {pat_text}, and operating cash flow {ocf_text}. "
        f"Free cash flow is {fcf_text}, which is important because SME IPOs with weak FCF can require repeated external funding."
    )


def render_metric_cards(analysis: dict) -> None:
    metrics = []
    for module_name, module in analysis["modules"].items():
        for metric in module["metrics"].values():
            metrics.append((module_name, metric))
    for row_start in range(0, len(metrics), 4):
        cols = st.columns(4)
        for col, (module_name, metric) in zip(cols, metrics[row_start : row_start + 4]):
            color = {"good": "#dcfce7", "watch": "#fef3c7", "bad": "#fee2e2", "neutral": "#e2e8f0"}[metric["status"]]
            latest_value = next(iter(metric["values"].values()), "n/a")
            if isinstance(latest_value, list):
                latest_value = latest_value[-1] if latest_value else "n/a"
            if isinstance(latest_value, float):
                latest_value = round(latest_value, 2)
            col.markdown(
                f"""
                <div class='metric-card' style='background:{color}'>
                    <div class='metric-module'>{module_name}</div>
                    <div class='metric-name'>{metric['name']}</div>
                    <div class='metric-value'>{latest_value}</div>
                    <div class='metric-score'>Score {metric['score']}/100</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def gauge_chart(score: float, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}, "steps": [
            {"range": [0, 49], "color": "#fee2e2"},
            {"range": [50, 74], "color": "#fef3c7"},
            {"range": [75, 100], "color": "#dcfce7"},
        ]},
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def radar_chart(module_scores: dict):
    labels = list(module_scores.keys())
    values = list(module_scores.values())
    fig = go.Figure(go.Scatterpolar(r=values + values[:1], theta=labels + labels[:1], fill="toself", name="Module score"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=300, margin=dict(l=30, r=30, t=30, b=30))
    return fig


def line_chart(df: pd.DataFrame, column: str, title: str):
    fig = go.Figure(go.Scatter(x=df["year"], y=df[column], mode="lines+markers", name=column))
    fig.update_layout(title=title, height=280, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def multi_line_chart(df: pd.DataFrame, columns: list[str], title: str):
    fig = go.Figure()
    for col in columns:
        fig.add_trace(go.Scatter(x=df["year"], y=df[col], mode="lines+markers", name=col.replace("_", " ").title()))
    fig.update_layout(title=title, height=280, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def peer_comparison(company: str, analysis: dict, portfolio_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if portfolio_df is not None and not portfolio_df.empty:
        columns = ["Company", "Investment Score", "Rating", "Revenue", "PAT", "OCF", "Debt/Equity", "Red Flags"]
        return portfolio_df[[col for col in columns if col in portfolio_df.columns]]
    latest = {metric["name"]: metric["score"] for module in analysis["modules"].values() for metric in module["metrics"].values()}
    rows = [{"Company": company, "Investment Score": analysis["overall"]["score"], "Rating": analysis["overall"]["rating"], **latest}]
    for idx in range(1, 6):
        rows.append({"Company": f"Peer {idx}", "Investment Score": np.nan, "Rating": "Manual input", **{key: np.nan for key in latest}})
    return pd.DataFrame(rows)


def ensure_columns(df: pd.DataFrame, analysis_years: int = 3) -> pd.DataFrame:
    df = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    df = df[REQUIRED_COLUMNS]
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    analysis_years = max(1, min(int(analysis_years or 3), 3))
    return df.sort_values("year").tail(analysis_years).reset_index(drop=True)


def sample_financials() -> pd.DataFrame:
    return pd.DataFrame([
        {"year": 2023, "revenue": 420, "cogs": 285, "ebitda": 54, "ebit": 43, "net_profit": 24, "total_assets": 330, "current_assets": 150, "current_liabilities": 90, "net_fixed_assets": 120, "receivables": 72, "inventory": 70, "payables": 58, "interest_expense": 9, "principal_repayments": 18, "operating_income": 50, "operating_cash_flow": 28, "total_debt": 115, "long_term_debt": 72, "cash": 18, "equity": 130, "goodwill": 5, "capex": 24, "debt_raised": 22},
        {"year": 2024, "revenue": 515, "cogs": 338, "ebitda": 74, "ebit": 60, "net_profit": 36, "total_assets": 385, "current_assets": 178, "current_liabilities": 102, "net_fixed_assets": 145, "receivables": 82, "inventory": 76, "payables": 66, "interest_expense": 10, "principal_repayments": 20, "operating_income": 70, "operating_cash_flow": 48, "total_debt": 125, "long_term_debt": 78, "cash": 24, "equity": 162, "goodwill": 5, "capex": 30, "debt_raised": 18},
        {"year": 2025, "revenue": 650, "cogs": 412, "ebitda": 102, "ebit": 84, "net_profit": 53, "total_assets": 455, "current_assets": 218, "current_liabilities": 116, "net_fixed_assets": 170, "receivables": 96, "inventory": 82, "payables": 74, "interest_expense": 11, "principal_repayments": 22, "operating_income": 98, "operating_cash_flow": 68, "total_debt": 132, "long_term_debt": 80, "cash": 32, "equity": 210, "goodwill": 5, "capex": 36, "debt_raised": 12},
    ])


def sample_firms() -> list[dict[str, Any]]:
    base = sample_financials()
    challenger = base.copy()
    challenger["revenue"] = [360, 470, 610]
    challenger["cogs"] = [255, 318, 402]
    challenger["ebitda"] = [38, 62, 96]
    challenger["ebit"] = [30, 50, 79]
    challenger["net_profit"] = [16, 28, 49]
    challenger["operating_cash_flow"] = [18, 38, 64]
    challenger["total_debt"] = [150, 142, 124]
    challenger["equity"] = [95, 126, 176]

    leveraged = base.copy()
    leveraged["revenue"] = [500, 560, 590]
    leveraged["cogs"] = [380, 432, 466]
    leveraged["ebitda"] = [48, 46, 42]
    leveraged["ebit"] = [35, 31, 25]
    leveraged["net_profit"] = [18, 12, 6]
    leveraged["operating_cash_flow"] = [12, -4, -8]
    leveraged["total_debt"] = [210, 245, 285]
    leveraged["equity"] = [110, 118, 122]
    leveraged["receivables"] = [115, 145, 172]

    return [
        {"company": "Sample SME IPO", "financials": base, "drhp_text": ""},
        {"company": "Growth Components Ltd", "financials": challenger, "drhp_text": "Objects of the issue include capacity expansion and working capital."},
        {"company": "Leveraged Manufacturing Ltd", "financials": leveraged, "drhp_text": "Related party transactions and revenue recognition policies are described in the DRHP."},
    ]


def load_benchmarks() -> dict:
    with open(DATA_DIR / "sector_benchmarks.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { background: #0f172a; color: white; }
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color: #e5e7eb; }
        .main .block-container { background: #ffffff; padding-top: 1.6rem; }
        .score-badge {
            color: white;
            font-size: 2.2rem;
            font-weight: 800;
            text-align: center;
            border-radius: 8px;
            padding: 1.35rem 0.5rem;
            margin-top: 1.25rem;
        }
        .metric-card {
            border: 1px solid rgba(15, 23, 42, 0.1);
            border-radius: 8px;
            padding: 0.85rem;
            min-height: 142px;
            margin-bottom: 0.75rem;
        }
        .metric-module { color: #475569; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; }
        .metric-name { color: #0f172a; font-size: 0.96rem; font-weight: 700; margin-top: 0.25rem; min-height: 2.4rem; }
        .metric-value { color: #111827; font-size: 1.35rem; font-weight: 800; margin-top: 0.4rem; overflow-wrap: anywhere; }
        .metric-score { color: #475569; font-size: 0.82rem; margin-top: 0.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
