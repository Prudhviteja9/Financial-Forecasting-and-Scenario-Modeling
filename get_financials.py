import pandas as pd
import yfinance as yf
from pathlib import Path

# -------- settings --------
TICKER = "AAPL"   # <--- change to your company
OUT_CSV = Path("data/financials_quarterly.csv")
# --------------------------

tk = yf.Ticker(TICKER)

# yfinance returns statements with metrics as rows, dates as columns
fin = tk.quarterly_financials.fillna(0)
cf  = tk.quarterly_cashflow.fillna(0)
inc = tk.quarterly_financials.fillna(0)  # same as fin; kept for clarity

# Helper to grab a row by any of several possible names
def pick_row(df, choices):
    for name in choices:
        if name in df.index:
            return df.loc[name]
    return pd.Series(dtype="float64")

# Revenue (a.k.a. Total Revenue / TotalRevenue)
revenue = pick_row(fin, ["Total Revenue", "TotalRevenue"])

# Net income (profit)
net_income = pick_row(fin, ["Net Income", "NetIncome"])

# Operating cash flow
ocf = pick_row(cf, [
    "Total Cash From Operating Activities",
    "Operating Cash Flow",
    "TotalCashFromOperatingActivities"
])

# Capital expenditures (usually negative)
capex = pick_row(cf, ["Capital Expenditures", "CapitalExpenditures", "Capital Expenditure"])

# Build tidy dataframe (quarters as rows)
df = pd.DataFrame({
    "date": revenue.index,   # quarter end date
    "revenue": revenue.values,
    "net_income": net_income.reindex(revenue.index, fill_value=0).values,
    "operating_cf": ocf.reindex(revenue.index, fill_value=0).values,
    "capex": capex.reindex(revenue.index, fill_value=0).values
})

# Free cash flow = OCF - CapEx
df["free_cf"] = df["operating_cf"] - df["capex"]

# Keep latest 20 quarters (5 years) if available, sort by date ascending
df = df.sort_values("date").tail(20).reset_index(drop=True)

# Basic sanity checks
for col in ["revenue","net_income","operating_cf","capex","free_cf"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Save & preview
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)

print(f"{TICKER} quarterly financials → {OUT_CSV}")
print(df.head(3))
print("\nRows:", len(df))
