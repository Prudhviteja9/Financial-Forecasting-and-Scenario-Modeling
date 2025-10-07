import numpy as np
import pandas as pd
from pathlib import Path

# Try to import Prophet, but don't fail if not used
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

from sklearn.linear_model import LinearRegression

# ====== settings ======
DATA_CSV = Path("data/financials_quarterly.csv")
OUT_CSV  = Path("data/financials_forecast.csv")

MIN_PROPHET_POINTS = 8      # rule of thumb for stable Prophet fit
Z80 = 1.2816                # ~80% interval z-score (one-sided ~0.9), close enough

def lr_forecast_next(y: np.ndarray):
    """
    Simple linear trend forecast with 80% interval using residual std.
    y: 1D numpy array (chronological).
    """
    x = np.arange(len(y)).reshape(-1, 1)
    lr = LinearRegression().fit(x, y)
    y_hat = lr.predict([[len(y)]])[0]
    resid = y - lr.predict(x)
    s = np.std(resid, ddof=1) if len(resid) > 1 else 0.0
    return y_hat, y_hat - Z80 * s, y_hat + Z80 * s

# ====== load data ======
df = pd.read_csv(DATA_CSV)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

metrics = ["revenue", "net_income", "free_cf"]
rows = []

for metric in metrics:
    series = df[["date", metric]].dropna().copy()
    series = series.sort_values("date")  # ensure chronological

    print(f"\nTraining points for {metric}: {len(series)}")

    if len(series) >= MIN_PROPHET_POINTS and HAS_PROPHET:
        # Prophet path
        data = series.rename(columns={"date": "ds", metric: "y"})
        m = Prophet(
            growth="linear",
            interval_width=0.80,
            weekly_seasonality=False,
            daily_seasonality=False,
            yearly_seasonality=False,
            n_changepoints=3
        )
        m.fit(data)
        future = m.make_future_dataframe(periods=1, freq="Q")
        fc = m.predict(future).tail(1).iloc[0]
        yhat, lo, hi = fc["yhat"], fc["yhat_lower"], fc["yhat_upper"]
        print(f"Prophet → {metric}: yhat={yhat:,.0f}, [{lo:,.0f}, {hi:,.0f}]")

    elif len(series) >= 3:
        # Linear regression fallback
        y = series[metric].astype(float).values
        yhat, lo, hi = lr_forecast_next(y)
        print(f"LR → {metric}: yhat={yhat:,.0f}, [{lo:,.0f}, {hi:,.0f}]")

    else:
        raise ValueError(
            f"Not enough points for {metric}. Need ≥3, got {len(series)}. "
            "Fetch a ticker with more history or use a longer dataset."
        )

    rows.append({
        "metric": metric,
        "next_quarter": series["date"].iloc[-1] + pd.offsets.QuarterEnd(1),
        "yhat": yhat,
        "yhat_lower": lo,
        "yhat_upper": hi
    })

out = pd.DataFrame(rows)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_CSV, index=False)
print(f"\nSaved → {OUT_CSV}\n")
print(out)
