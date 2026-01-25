# STOCHASTIC PROBLEM - CODE


# import libraries
import pulp
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import matplotlib.patheffects as path_effects
# import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# DEFINING PROBLEM DATA
# ----------------------------------------------------------------------------

scenario = "two_stage_problem"

# orders setup
orders_type_a = [0, 1]
orders_type_b = [2, 3]
orders = sorted(orders_type_a + orders_type_b)  # all orders in ascending order
orders_profit = {0: 4000, 1:2000, 2:5000, 3:4500}   # orders profit
duration = [2, 4, 5, 3]  # duration of execution of orders (in hours)
consumption = [4, 1, 2, 3]  # energy consumption of each order (Mwhr)

# factories setup
factories_type_a = [0]
factories_type_b = [1]
factories_type_c = [2]
factories = sorted(factories_type_a + factories_type_b + factories_type_c)  # all factories in ascending order

M = 1000  # M variable for the Big M Method
hours = list(range(0, 24))

# scenarios setup
scenarios = [0, 1]
pscenarios = {0: 0.3, 1: 0.7} # probability of each scenario

# energy price list setup
energy_price_0 = [0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 250, 250, 250, 250, 250]  # energy price forecasting - scenario 1 (€/Mwhr)
energy_price_1 = [250, 250, 250, 250, 250, 250, 250, 250,
 250, 250, 250, 250, 250, 250, 250, 250,
 250, 250, 250, 0, 0, 0, 0, 0]  # energy price forecasting - scenario 2 (€/Mwhr)

fixed_energy_price = 50    # fixed energy price (€)

all_price_lists = [energy_price_0, energy_price_1]
scenario_prices = {w: all_price_lists[w] for w in scenarios}


# ----------------------------------------------------------------------------
# CREATING COMBINATIONS OF FACTORIES - ORDERS
# ----------------------------------------------------------------------------

# every combination of factories-orders and sorting the orders
factories_orders = [(j, i) for i in orders for j in factories]

# dictionary for every feasible order-factory
orders_2_factories = {
    i: [ j for j in factories if
        (
            (i in orders_type_a and j in factories_type_a)
            or (i in orders_type_b and (j in factories_type_b or j in factories_type_c))
        )
    ]
    for i in orders
}

# every feasible combination of factories-orders
feasible_factories_orders = [
    (j, i) for i in orders for j in orders_2_factories[i]
]


# ----------------------------------------------------------------------------
# CREATING VARIABLES, LISTS & DICTIONARIES FOR THE SOLUTION OF THE PROBLEM
# ----------------------------------------------------------------------------

# creation of Eji list for every order
Eji = {}
for w in scenarios:
    current_forecast = scenario_prices[w]
    for j, i in feasible_factories_orders:
        d = duration[i]
        cons = consumption[i]
        Eji[(w, j, i)] = []
        for si in range(24):
            if si + d <= 24:
                sum_prices = sum(current_forecast[si : si + d])
                cost = sum_prices * cons
                Eji[(w, j, i)].append(cost)


# defining variables x, p, q, s_hour, s_i & energy_bought
energy_bought = pulp.LpVariable("Energy_bought", lowBound=0, cat="Continuous")

x_ji = pulp.LpVariable.dicts("x", [(w, j, i) for w in scenarios for (j,i) in factories_orders], cat="Binary")
p_jik = pulp.LpVariable.dicts(
    "p",
    [
        (w, j, i, k)
        for w in scenarios
        for i in orders
        for k in orders
        if i != k
        for j in set(orders_2_factories[i]).intersection(set(orders_2_factories[k]))
    ],
    cat="Binary",
)
q_jik = pulp.LpVariable.dicts(
    "q",
    [
        (w, j, i, k)
        for w in scenarios
        for i in orders
        for k in orders
        if i != k
        for j in set(orders_2_factories[i]).intersection(set(orders_2_factories[k]))
    ],
    cat="Binary",
)

# defining the starting time variable for the orders
s_i = pulp.LpVariable.dicts(
    "startingtime", [(w, i) for w in scenarios for i in orders], lowBound=0, upBound=23, cat="Integer"
)

# creating a custom variable for the calculation of the starting time
s_hour = {}
for w in scenarios:
    for j, i in feasible_factories_orders:
        possible_starts = len(Eji[(w, j, i)])
        s_hour[(w, j, i)] = [
            pulp.LpVariable(f"s_hour_scen{w}_{j}_{i}_{h:02d}", cat="Binary")
            for h in range(possible_starts)
        ]


# ----------------------------------------------------------------------------
# DEFINING THE OPTIMIZATION PROBLEM
# ----------------------------------------------------------------------------

# define the linear problem
twostageprob = pulp.LpProblem("Two-Stage Problem", pulp.LpMaximize)

# ----------------------------------------------------------------------------
# SETTING THE CONSTRAINTS OF THE OPTIMIZATION PROBLEM & DEFINE THE OBJECTIVE FUNCTION
# ----------------------------------------------------------------------------

# all the constraints in a loop for each scenario
for w in scenarios:
    # an order can be assigned to a factory or not
    order_x = {}
    for o, f in orders_2_factories.items():
        # ensures that each order is assigned to a maximum of one factory
        order_x[o] = pulp.lpSum([x_ji[(w, f_, o)] for f_ in f])
        twostageprob += (
            order_x[o] <= 1,
            f"sum(x_f{o}_scen{w})<=1",
        )

    for i in orders:
        # factories that can execute order i
        valid_factories_for_i = orders_2_factories[i]
        # sum of s_i[i] for each factory (only 1 is selected - the others are 0)
        hours_list = []
        for j in valid_factories_for_i:
            possible_starts = len(Eji[(w, j, i)])
            hours_list.extend([h * s_hour[(w, j, i)][h] for h in range(possible_starts)])
        # calculation of the starting time of each order i
        twostageprob += (
            s_i[w, i] == pulp.lpSum(hours_list),
            f"calculate_start_time_for_order_{i}_scen{w}",
        )

    for j, i in feasible_factories_orders:
        # ensuring the proper function of the custom variable s_hour
        twostageprob += (
            pulp.lpSum(s_hour[(w, j, i)]) == x_ji[(w, j, i)],
            f"sum(s_hour_{j}_{i}_scen{w})==x_{j}{i}",
        )
        for k in orders:
            if k != i and (j, k) in feasible_factories_orders:
                # Big M Method constraints
                # p indicates whether two orders are assigned to the same factory
                twostageprob += (
                    x_ji[(w, j, i)] + x_ji[(w, j, k)] <= 1 + M * p_jik[(w, j, i, k)],
                    f"Big_M_constraint_1_for_order_{j}_{i}_{k}_scen{w}",
                )
                twostageprob += (
                    2 - M * (1 - p_jik[(w, j, i, k)]) <= x_ji[(w, j, i)] + x_ji[(w, j, k)],
                    f"Big_M_constraint_2_for_order_{j}_{i}_{k}_scen{w}",
                )
                # q indicates whether order i is scheduled before order k
                # simplifiedprob += (
                #     s_i[i] - s_i[k] <= (M * (1 - q_jik[(j, i, k)])) + (M * (1 - p_jik[(j, i, k)])),
                #     f"Big_M_constraint_3_for_order_{j}_{i}_{k}",
                # )
                # simplifiedprob += (
                #     s_i[i] - s_i[k] >= -(q_jik[(j, i, k)] * M) - (M * (1 - p_jik[(j, i, k)])),
                #     f"Big_M_constraint_4_for_order_{j}_{i}_{k}",
                # )
                # ensures a non-overlapped-in-time execution sequence between two orders i and k
                twostageprob += (
                    s_i[w, i] + duration[i] - s_i[w, k]
                    <= (M * (1 - q_jik[(w, j, i, k)])) + (M * (1 - p_jik[(w, j, i, k)])),
                    f"Big_M_constraint_5_for_order_{j}_{i}_{k}_scen{w}",
                )
                twostageprob += (
                    s_i[w, k] + duration[k] - s_i[w, i]
                    <= (M * q_jik[(w, j, i, k)]) + (M * (1 - p_jik[(w, j, i, k)])),
                    f"Big_M_constraint_6_for_order_{j}_{i}_{k}_scen{w}",
                )

    # actual consumption for the scenario
    total_consumption_of_w = pulp.lpSum(
        [x_ji[(w, j, i)] * consumption[i] * duration[i] for j, i in feasible_factories_orders],
    )
    twostageprob += total_consumption_of_w <= energy_bought, f"Energy_Sufficiency_Scenario_{w}"

# define the objective function
# the goal is to minimize energy cost AND maximize profit

en_capacity_cost = energy_bought * fixed_energy_price

# total energy cost
expected_profit = []
for w in scenarios:
    scenario_hourly_cost = []
    for j, i in feasible_factories_orders:
        for h, e in enumerate(Eji[(w, j, i)]):
            scenario_hourly_cost.append(Eji[(w, j, i)][h] * s_hour[(w, j, i)][h])
    scenario_spot_cost = pulp.lpSum(scenario_hourly_cost)

    scenario_revenue = pulp.lpSum(
        [x_ji[(w, j, i)] * orders_profit[i] for j, i in feasible_factories_orders]
    )

    # total profit of orders placed in factories
    expected_profit.append(pscenarios[w] * (scenario_revenue - scenario_spot_cost))

# sum of (total energy cost - total profit)
twostageprob += pulp.lpSum(expected_profit) - en_capacity_cost, "Ypoloipo:"


# ----------------------------------------------------------------------------
# SOLVING THE PROBLEM & DISPLAY RESULTS
# ----------------------------------------------------------------------------

# print and solve problem
# print(twostageprob)
twostageprob.solve()
twostageprob.writeLP("stochastic_problem.lp")
print("Status:", pulp.LpStatus[twostageprob.status])

stage_1_cost_value = pulp.value(en_capacity_cost)

# output and display results
for v in twostageprob.variables():
    print(v.name, "=", v.varValue)

print(f"\nOptimal Energy to Buy (Stage 1): {pulp.value(energy_bought)} MWh")
print(f"Cost of Optimal Energy Bought: {stage_1_cost_value} €")
print(f"Expected Profit (Objective Value): {pulp.value(twostageprob.objective)} €")

for w in scenarios:
    print(f"\n--- SCENARIO {w} (Probability: {pscenarios[w]}) ---")
    active_orders = []
    cons = 0
    scenario_revenue = 0
    scenario_spot_cost = 0

    for j, i in feasible_factories_orders:
        if pulp.value(x_ji[(w, j, i)]) > 0.5:
            start = int(pulp.value(s_i[(w, i)]))
            spot = Eji[(w, j, i)][start]            
            print(f"- Order {i} at Factory {j}: Start Hour {int(start)} | Dur: {duration[i]}h | Cons: {consumption[i]} MWh) | Revenue: {orders_profit[i]} MW | Spot Cost: {spot:.1f}€ | Profit: {orders_profit[i] - spot:.1f}€")
            cons += consumption[i] * duration[i]
            scenario_revenue += orders_profit[i]
            scenario_spot_cost += spot

    scenario_spot_profit = scenario_revenue - scenario_spot_cost
    scenario_total_profit = scenario_spot_profit - stage_1_cost_value
    
    print(f"\n  SUMMARY OF THE SCENARIO {w}:")
    print(f"  Total Consumption: {cons} MWh (Limit: {pulp.value(energy_bought)})")
    print(f"  > Revenue:            {scenario_revenue} €")
    print(f"  > Spot Cost:          {scenario_spot_cost:.1f} €")
    print(f"  > Operational Profit: {scenario_spot_profit:.1f} €")
    print(f"  > Stage 1 Fixed Cost: {stage_1_cost_value} €")
    print(f"  > NET PROFIT:         {scenario_total_profit:.1f} €")

results_dict = {v.name: v.varValue for v in twostageprob.variables()}
pd.DataFrame(results_dict.items()).to_excel(
    f"results/problem_{scenario}.xlsx", engine="openpyxl"
)


# # ----------------------------------------------------------------------------
# # CREATION & DISPLAY OF THE DIAGRAM
# # ----------------------------------------------------------------------------


# # data preparation for standard variables

# plot_data = []

# for i in orders:
#     # finding the factory where the order i has been assigned
#     assigned_factory = None
#     for j in factories:
#         if (
#             (j, i) in x_ji
#             and x_ji[(j, i)].varValue is not None
#             and x_ji[(j, i)].varValue > 0.5
#         ):
#             assigned_factory = j
#             break

#     # retrieves time info
#     start_time = s_i[i].varValue
#     dur = duration[i]

#     # cost calculation
#     cost = 0
#     if start_time is not None:
#         start_idx = int(start_time)
#         if start_idx + dur <= 24:
#             cost = sum(forecast_en[start_idx : start_idx + dur])

#     # color determination
#     if i in orders_type_a:
#         color = "red"
#     else:
#         color = "blue"

#     # if order is assigned, then all info is saved to a dictionary and will be used for plotting
#     if assigned_factory is not None and start_time is not None:
#         plot_data.append(
#             {
#                 "order_id": i,
#                 "order_label": f"Order {i}",
#                 "start": start_time,
#                 "duration": dur,
#                 "cost": cost,
#                 "color": color,
#                 "factory_label": f"Factory {assigned_factory}",
#             }
#         )

# # 1st graph - Energy cost of each factory

# # figure creation with specific dimensions
# plt.figure(figsize=(12, 6))

# # exporting lists from plot_data
# p1_labels = [d["order_label"] for d in plot_data]
# p1_costs = [d["cost"] for d in plot_data]
# p1_colors = [d["color"] for d in plot_data]
# p1_factories = [d["factory_label"] for d in plot_data]

# bars = plt.bar(p1_labels, p1_costs, color=p1_colors)

# # add text inside bars
# for bar, factory_name in zip(bars, p1_factories):
#     height = bar.get_height()
#     plt.text(
#         bar.get_x() + bar.get_width() / 2,
#         height / 2,
#         factory_name,
#         ha="center",
#         va="center",
#         color="white",
#         fontweight="bold",
#         fontsize=10,
#     )

# # graph formatting
# plt.ylabel("Energy Cost (€/Mwhr)", fontsize=12)
# plt.title("Energy Cost per Order (Colored by Type)", fontsize=14)
# plt.grid(axis="y", linestyle="--", alpha=0.5)
# red_patch = mpatches.Patch(color="red", label="Type A Orders")
# blue_patch = mpatches.Patch(color="blue", label="Type B Orders")
# plt.legend(handles=[red_patch, blue_patch])
# plt.tight_layout()
# plt.show()


# # 2nd graph - Ganntt Chart

# # figure creation with specific dimensions
# plt.figure(figsize=(14, 8))

# # drawing grid lines
# plt.grid(axis="y", linestyle="--", color="black", alpha=0.5, zorder=0)
# plt.grid(axis="x", linestyle=":", alpha=0.3, zorder=0)

# for item in plot_data:
#     # creating bar
#     plt.barh(
#         y=item["order_label"],
#         width=item["duration"],
#         left=item["start"],
#         color=item["color"],
#         edgecolor="black",
#         alpha=0.9,
#         zorder=3,
#     )

#     # adjusting text size according to block size
#     if item["duration"] <= 1:
#         f_size = 7
#         f_weight = "normal"
#     else:
#         f_size = 9
#         f_weight = "bold"

#     # adding text
#     plt.text(
#         x=item["start"] + item["duration"] / 2,
#         y=item["order_label"],
#         s=item["factory_label"],
#         ha="center",
#         va="center",
#         color="white",
#         fontweight=f_weight,
#         fontsize=f_size,
#         zorder=4,
#     )

# # graph formatting
# plt.title("Order Scheduling per Hour (Gantt Chart)", fontsize=16)
# plt.xlabel("Hour of the Day", fontsize=12)
# plt.ylabel("Orders", fontsize=12)
# plt.xlim(0, 24)
# plt.xticks(range(0, 25))
# plt.gca().invert_yaxis()  # order 0 at top
# plt.legend(handles=[red_patch, blue_patch], loc="upper left", framealpha=1)
# plt.tight_layout()
# plt.show()


# # 3rd graph - Chart of the energy cost over time

# # creating 2 tables (one for each factory type)
# hourly_cost_type_a = np.zeros(24)
# hourly_cost_type_b = np.zeros(24)
# hourly_cost_type_c = np.zeros(24)

# # run through the data we already have in plot_data
# for item in plot_data:
#     # exporting the factory ID as integer
#     f_str = item["factory_label"]
#     f_id = int(f_str.split(" ")[1])
#     # exporting the starting time and duration as integer
#     start = int(item["start"])
#     dur = int(item["duration"])

#     # checking if factory is Type A or B
#     is_type_a = f_id in factories_type_a
#     is_type_b = f_id in factories_type_b

#     # adding the energy cost for each hour that the factory operates
#     for h in range(start, start + dur):
#         if h < 24:
#             cost_at_h = forecast_en[h]
#             if is_type_a:
#                 hourly_cost_type_a[h] += cost_at_h
#             elif is_type_b:
#                 hourly_cost_type_b[h] += cost_at_h
#             else:
#                 hourly_cost_type_c[h] += cost_at_h

# # calculation of mean value
# total_hourly_cost = hourly_cost_type_a + hourly_cost_type_b + hourly_cost_type_c
# average_cost = np.mean(total_hourly_cost)

# # figure creation with specific dimensions
# plt.figure(figsize=(14, 7))

# # drawing the red line for Type A factories
# plt.plot(
#     range(24), hourly_cost_type_a, color="red", linewidth=2.5, label="Type A Factories"
# )

# # drawing the blue line for Type B factories
# plt.plot(
#     range(24), hourly_cost_type_b, color="blue", linewidth=2.5, label="Type B Factories"
# )

# # drawing the green line for Type C factories
# plt.plot(
#     range(24), hourly_cost_type_c, color="green", linewidth=2.5, label="Type C Factories"
# )

# # drawing the dotted line of the average hourly cost
# plt.axhline(
#     y=average_cost,
#     color="black",
#     linestyle="--",
#     linewidth=1.5,
#     label=f"Average Hourly Cost ({average_cost:.1f} €/Mwhr)",
# )

# # graph formatting
# plt.title("Hourly Energy Cost Profile per Factory Type", fontsize=16)
# plt.xlabel("Hour of the day", fontsize=12)
# plt.ylabel("Energy Cost (€/Mwhr)", fontsize=12)
# # axis setting
# plt.xlim(0, 23)
# plt.xticks(range(0, 24))
# plt.grid(True, linestyle="-", alpha=0.3)  # background grid
# # filling the area below the lines
# plt.fill_between(range(24), hourly_cost_type_a, color="red", alpha=0.1)
# plt.fill_between(range(24), hourly_cost_type_b, color="blue", alpha=0.1)
# plt.fill_between(range(24), hourly_cost_type_c, color="green", alpha=0.1)
# plt.legend(loc="upper left", fontsize=11, frameon=True, shadow=True)
# plt.tight_layout()
# plt.show()


# # 4th graph - Combo chart of order sorting & "energy cost" curve

# # figure creation with specific dimensions
# fig, ax1 = plt.subplots(figsize=(14, 8))
# # creation of axis Y2
# ax2 = ax1.twinx() 

# # drawing the "energy cost" curve
# line_plot = ax2.plot(
#     range(24), 
#     forecast_en, 
#     color='orange', 
#     linewidth=4, 
#     alpha=0.8,
#     label='Energy Price (€/MWh)',
#     zorder=2    # behind the letters but in front of the grid
# )

# # adjustments for axis Y2
# ax2.set_ylabel("Energy Price (€/MWh)", fontsize=12, color='darkorange', fontweight='bold')
# ax2.tick_params(axis='y', labelcolor='darkorange')
# ax2.set_ylim(0, max(forecast_en) + 50)

# # export all factories for the Y1 axis
# all_factories = sorted(factories)
# factory_labels = [f"Factory {f}" for f in all_factories]

# for item in plot_data:
#     # finding the factory ID
#     f_id = int(item["factory_label"].split()[1])
#     # drawing bar
#     ax1.barh(
#         y=f_id, 
#         width=item["duration"], 
#         left=item["start"], 
#         color=item["color"], 
#         edgecolor='black',
#         alpha=0.85,
#         height=0.6,
#         zorder=1    # behind the curve
#     )
    
#     # adjusting text size
#     if item["duration"] <= 1:
#         f_size = 8
#         f_weight = 'bold'
#     else:
#         f_size = 10
#         f_weight = 'bold'

#     # adding text "Order X"
#     txt = ax1.text(
#         x=item["start"] + item["duration"] / 2,
#         y=f_id,
#         s=item["order_label"],
#         ha='center', 
#         va='center', 
#         color='white', 
#         fontweight=f_weight, 
#         fontsize=f_size, 
#         zorder=3
#     )
#     # black outline around the text
#     txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground='black')])

# # adjustments for axis Y1
# ax1.set_xlabel("Hour of Day", fontsize=12)
# ax1.set_ylabel("Production Line", fontsize=12)
# ax1.set_yticks(all_factories)
# ax1.set_yticklabels(factory_labels, fontsize=11, fontweight='bold')

# # adjustments for axis X and grid
# ax1.set_xlim(0, 24)
# ax1.set_xticks(range(0, 25))
# ax1.grid(axis='x', linestyle=':', alpha=0.5)
# ax1.invert_yaxis()  # factory 0 at top
# plt.title("Combined Schedule & Energy Cost Profile", fontsize=16, pad=20)

# # creation and placement of the custom legend of graph
# from matplotlib.lines import Line2D

# custom_lines = [
#     mpatches.Patch(color='red', label='Type A Orders'),
#     mpatches.Patch(color='blue', label='Type B Orders'),
#     Line2D([0], [0], color='orange', lw=3, label='Energy Price')
# ]
# ax1.legend(handles=custom_lines, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, shadow=True)
# plt.tight_layout()
# plt.show()
