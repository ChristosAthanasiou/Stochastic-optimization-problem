'''
DIPLOMA THESIS — SCENARIO GENERATION VIA K-MEANS CLUSTERING
----------------------------------------------------------------------------
Loads historical hourly electricity price data of Greece (2014-2025),
clusters the daily price profiles into K representative scenarios,
and saves the results to stochastic_data.xlsx (sheet: "Scenarios").
'''

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os

# settings
input_file  = "hourly_electricity_prices_GR.xlsx"       # historical price data file
output_file = "stochastic_data.xlsx"                    # output file (must already exist with other sheets)
n_clusters  = 3                                         # number of scenarios (K)
random_seed = 42                                        # for reproducibility


# load data
print("Loading data...")
df = pd.read_excel(input_file, header=0)
df.columns = ["Datetime", "Price"]


# seperate the "Datetime" column into "Date" and "Hour" columns for easier processing
df["Datetime"] = pd.to_datetime(df["Datetime"])
df["Date"] = df["Datetime"].dt.date
df["Hour"] = df["Datetime"].dt.hour

# pivot to wide format (one row per day, 24 columns for hours)
print("Reshaping data to wide format...")
df_wide = df.pivot_table(index="Date", columns="Hour", values="Price", aggfunc="mean")

# rename columns to hour_0, hour_1, ..., hour_23
df_wide.columns = [f"hour_{h}" for h in range(24)]

# drop days with missing data
df_wide.dropna(inplace=True)

print(f"Total valid days: {len(df_wide)}")


# apply K-Means clustering to the daily price profiles
print(f"Applying K-Means clustering with K={n_clusters}...")
kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
kmeans.fit(df_wide.values)

# cluster labels for each day
labels = kmeans.labels_

# extract the cluster centroids (representative daily price profiles for each scenario)
centroids = kmeans.cluster_centers_  # shape: (n_clusters, 24)

# calculate the probability of each scenario based on the number of days assigned to each cluster
total_days = len(df_wide)
probabilities = [
    round(np.sum(labels == k) / total_days, 4)
    for k in range(n_clusters)
]

# make sure probabilities sum to exactly 1.0 (fix rounding errors)
probabilities[-1] = round(1.0 - sum(probabilities[:-1]), 4)

print("\nScenario probabilities:")
for k in range(n_clusters):
    print(f"  Scenario {k}: {np.sum(labels == k)} days → p = {probabilities[k]:.4f}")


# build a dataframe with scenario_id, probability & hourly prices for each scenario
hour_cols = [f"hour_{h}" for h in range(24)]

scenarios_df = pd.DataFrame(centroids, columns=hour_cols)
scenarios_df.insert(0, "probability", probabilities)
scenarios_df.insert(0, "scenario_id", range(n_clusters))

# round centroid prices to 2 decimal places
scenarios_df[hour_cols] = scenarios_df[hour_cols].round(2)

print("\nGenerated scenarios:")
print(scenarios_df[["scenario_id", "probability"]].to_string(index=False))


# save to the output excel file stochastic_data.xlsx (sheet: "Scenarios")
print(f"\nSaving scenarios to '{output_file}' (sheet: Scenarios)...")

if os.path.exists(output_file):
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        scenarios_df.to_excel(writer, sheet_name='Scenarios', index=False)
    print("Done! Scenarios saved successfully (Replaced existing sheet).")
else:
    raise FileNotFoundError(f"The file {output_file} was not found.")

print("============================================================================")
print(f"  Scenarios generated : {n_clusters}")
print(f"  Valid days used     : {total_days}")
print(f"  Probability check   : {sum(probabilities):.2f} (should be 1.00)")
print("============================================================================")