# SME Investment Analysis Workbench

Python + Streamlit workbench for professional pre-IPO and SME IPO screening.

## Features

- Manual single-year or multi-year restated financial input with editable Streamlit table
- CSV upload for structured financials
- Multi-file PDF/photo upload, DRHP red-flag scanning, and AI-assisted financial extraction
- Public ticker context through `yfinance`
- Sidebar fund parameters: target IRR, horizon, sector, discount rate, max D/E, ROCE floor
- Separate sidebar controls for single vs multi-company analysis and single-year vs multi-year analysis
- 38 financial indicators across income statement, working capital, balance sheet, cash flow, valuation, capital budgeting, and risk
- IPO-specific checks for promoter holding, GMP, proceeds, and DRHP red flags
- Weighted INVEST / WATCH / AVOID investment score
- Plotly gauge, radar chart, metric cards, trend charts, red flag log, peer comparison scaffold
- Claude investment memo integration with deterministic fallback when no API key is supplied
- OpenAI/ChatGPT document and image extraction for financial reports and statement photos
- Financial Extraction Review section showing uploaded files, missing fields, source clues, extracted rows, formulas, scores, and warnings
- ReportLab PDF export with ratio table, charts, and memo

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

On this Windows machine you can also double-click:

- `install_app.bat` to install dependencies
- `launch_sme_ipo_app.bat` to start the local application

Or run the launcher with Python:

```bash
python run_app.py
```

## Expected Financial Columns

CSV uploads should include:

`year, revenue, cogs, ebitda, ebit, net_profit, total_assets, current_assets, current_liabilities, net_fixed_assets, receivables, inventory, payables, interest_expense, principal_repayments, operating_income, operating_cash_flow, total_debt, long_term_debt, cash, equity, goodwill, capex, debt_raised`

Missing columns are filled with zero so fund managers can paste partial pre-IPO data and complete it in the editable table.

## Multi-Firm Analysis

Choose **Multi company** in the sidebar to analyze several companies together.

Supported upload patterns:

- One combined CSV with a `company`, `firm`, `company_name`, `issuer`, or `name` column
- Multiple CSV files, one per company
- Multiple PDFs for DRHP red-flag scanning; edit the generated financial rows in the app
- PDF autoscan for machine-readable financial tables. The app maps common labels such as revenue, PAT, assets, borrowings, receivables, inventory, payables, OCF, capex, and cash into the required scoring model.

The app creates a firm-level assumptions table for sector, IPO valuation, investment amount, promoter holding, GMP, and use of proceeds. It then ranks every firm by the weighted investment score and lets you drill into any one firm for the full dashboard, memo, and PDF report.

PDF autoscan is heuristic because DRHP and annual-report table formats vary. Always review the extracted financial table in the app before using the final score.

## AI Agent Extraction

The sidebar has a **Financial document extraction** selector:

- `Heuristic autoscan`: uses local table parsing only
- `AI agent autoscan`: sends parsed report text to Claude and asks it to fill the scoring schema
- `Hybrid autoscan`: uses AI when it extracts at least as many fields as the heuristic parser

To use AI extraction, paste a Claude/Anthropic API key into the sidebar field. If no key is supplied, the app falls back to local heuristic extraction.

For ChatGPT/OpenAI extraction from PDFs and photos, paste an OpenAI API key into the sidebar. The app accepts PDF, PNG, JPG, JPEG, and WEBP uploads. You can upload multiple files at once; for single-company mode the app merges them into one company, and for multi-company mode it groups files by company-like filename prefix.

Always review the extracted table before scoring. Financial reports may use different units, consolidated/standalone sections, or repeated tables.
