'''
DIPLOMA THESIS — WAIT-AND-SEE (WS) CALCULATION
----------------------------------------------------------------------------
The deterministic problem is solved separately for each scenario,
as if that scenario were known with certainty (probability = 1).
WS = weighted average of the optimal profits across all scenarios:
     WS = Σ p_w * profit_w
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


# solve deterministic problem for each scenario independently
print("============================================================================")
print("                   WAIT-AND-SEE (WS) CALCULATION")
print("============================================================================")

profit_per_scenario = {}

for w in scenarios:
    forecast = scenario_prices[w]
    _, profit, _, _, _, _ = solve_deterministic(forecast_en=forecast, print_output=False)
    profit_per_scenario[w] = profit
    print(f"\nScenario {w} (p={pscenarios[w]}):")
    print(f"  Optimal Profit (if known with certainty) = {profit:.1f} €")


# calculation of the WS metric
WS = sum(pscenarios[w] * profit_per_scenario[w] for w in scenarios)

print("\n============================================================================")
print(f"                   WS (Wait-and-See Value) = {WS:.1f} €")
print("============================================================================")