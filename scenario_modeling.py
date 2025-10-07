import pandas as pd
from pathlib import Path

# ====== load forecast ======
df = pd.read_csv("data/financials_forecast.csv")

# define planning assumptions (% changes relative to base)
assumptions = {
    "best": {"revenue": 1.05, "net_income": 1.10, "free_cf": 1.08},   # optimistic growth
    "base": {"revenue": 1.00, "net_income": 1.00, "free_cf": 1.00},   # neutral
    "worst": {"revenue": 0.90, "net_income": 0.85, "free_cf": 0.88},  # downturn
}

rows = []
for scenario, mult in assumptions.items():
    for _, row in df.iterrows():
        metric = row["metric"]
        rows.append({
            "Scenario": scenario,
            "Metric": metric,
            "Forecast_Date": row["next_quarter"],
            "Base_Value": row["yhat"],
            "Adjusted_Value": row["yhat"] * mult[metric],
        })

out = pd.DataFrame(rows)
out.to_csv("data/scenario_output.csv", index=False)
print("\nSaved → data/scenario_output.csv\n")
print(out)
