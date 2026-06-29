'''
DIPLOMA THESIS — EXPECTED VALUE (EV) CALCULATION
----------------------------------------------------------------------------
The deterministic problem is solved once using the Expected Value forecast,
which is the probability-weighted average of all scenario price forecasts:
     EV_forecast[h] = Σ p_w * price_w[h]   for each hour h
The optimal profit obtained is the EV solution.
'''



import pandas as pd
from deterministic_problem_code import solve_deterministic


# load scenario data from excel file
input_file = "stochastic_data.xlsx"

df_scenarios = pd.read_excel(input_file, sheet_name="Scenarios")
scenarios = df_scenarios["scenario_id"].tolist()
pscenarios = dict(zip(df_scenarios["scenario_id"], df_scenarios["probability"]))
hour_cols = [f"hour_{h}" for h in range(24)]
scenario_prices = {
    row["scenario_id"]: [row[h] for h in hour_cols]
    for _, row in df_scenarios.iterrows()
}


# calculate expected value forecast (weighted average across scenarios)
EV_forecast = [
    sum(pscenarios[w] * scenario_prices[w][h] for w in scenarios)
    for h in range(24)
]

print("============================================================================")
print("                 EXPECTED VALUE (EV) CALCULATION")
print("============================================================================")
print("\nEV Forecast (weighted average price per hour):")
for h, price in enumerate(EV_forecast):
    print(f"  Hour {h:02d}: {price:.2f} €/MWh")


# solve deterministic problem using the EV forecast
_, EV, EV_xji_decisions, _, _, _ = solve_deterministic(forecast_en=EV_forecast, print_output=False)

print("\nEV Order Acceptance Decisions (Stage 1):")
for (j, i), val in EV_xji_decisions.items():
    if val is not None and val > 0.5:
        print(f"  Order {i} → accepted & assigned to Factory {j}")

print("\n============================================================================")
print(f"               EV (Expected Value Solution) = {EV:.1f} €")
print("============================================================================")