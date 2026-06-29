'''
DIPLOMA THESIS — TWO-STAGE STOCHASTIC OPTIMIZATION PROBLEM
----------------------------------------------------------------------------
Stage 1: Accept or reject each order and assign it to a factory
         (decision made before the energy price scenario is revealed)
Stage 2: Schedule the accepted orders in time for each scenario
         (decision made after the energy price scenario is revealed)
Objective: Maximize expected profit across all scenarios
           while minimizing energy cost
'''



# import libraries
import pulp
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


# ----------------------------------------------------------------------------
# LOADING PROBLEM DATA FROM EXCEL
# ----------------------------------------------------------------------------

input_file = "stochastic_data.xlsx"

scenario = "scenario_0"

# orders setup from sheet
df_orders = pd.read_excel(input_file, sheet_name="Orders")
orders_type_a = df_orders[df_orders["type"] == "A"]["order_id"].tolist()
orders_type_b = df_orders[df_orders["type"] == "B"]["order_id"].tolist()
orders = sorted(df_orders["order_id"].tolist())     # all orders in ascending order

orders_revenue = dict(zip(df_orders["order_id"], df_orders["profit (€)"]))   # orders profit

duration_dict = df_orders.set_index("order_id")["duration (h)"].to_dict()
duration = [duration_dict[i] for i in orders]        # duration of execution of orders (in hours)

consumption_dict = df_orders.set_index("order_id")["consumption (MW/h)"].to_dict()
consumption = [consumption_dict[i] for i in orders]      # energy consumption of each order per hour (MW)

# factories setup from sheet
df_factories = pd.read_excel(input_file, sheet_name="Factories")
factories_type_a = df_factories[df_factories["type"] == "A"]["factory_id"].tolist()
factories_type_b = df_factories[df_factories["type"] == "B"]["factory_id"].tolist()
factories_type_c = df_factories[df_factories["type"] == "C"]["factory_id"].tolist()
factories = sorted(df_factories["factory_id"].tolist())     # all factories in ascending order

# scenarios setup
df_scenarios = pd.read_excel(input_file, sheet_name="Scenarios")
scenarios = df_scenarios["scenario_id"].tolist()
pscenarios = dict(zip(df_scenarios["scenario_id"], df_scenarios["probability"]))
num_scenarios = len(scenarios)
hour_cols = [f"hour_{h}" for h in range(24)]
scenario_prices = {
    row["scenario_id"]: [row[h] for h in hour_cols]
    for _, row in df_scenarios.iterrows()
}

# other constants
hours = list(range(0, 24))
# M variable for the Big M Method
M = 1000



# ----------------------------------------------------------------------------
# CREATING COMBINATIONS OF FACTORIES - ORDERS
# ----------------------------------------------------------------------------

# every combination of factories-orders
factories_orders = [(j, i) for i in orders for j in factories]

# dictionary for every feasible order-factory
orders_2_factories = {
    i: [j for j in factories if
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

def solve_stochastic(fixed_x_decisions=None, print_output=False):

    # creation of Eji list for every order and scenario (contains the energy cost of each order for every possible starting hour)
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


# ----------------------------------------------------------------------------
# STAGE 1 VARIABLES: order acceptance & factory assignment (scenario-independent)
# ----------------------------------------------------------------------------

    # x_ji = 1 if order i is accepted and assigned to factory j, 0 otherwise (Stage 1 decision)
    x_ji = pulp.LpVariable.dicts("x", feasible_factories_orders, cat="Binary")


# ----------------------------------------------------------------------------
# STAGE 2 VARIABLES: scheduling (scenario-dependent)
# ----------------------------------------------------------------------------

    # p_jik = 1 if orders i and k are assigned to the same factory j, 0 otherwise (Stage 2 decision)
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

    # q_jik = 1 if order i is scheduled before order k in factory j, 0 otherwise (Stage 2 decision)
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

    # s_i = starting time of order i in scenario w (Stage 2 decision)
    s_i = pulp.LpVariable.dicts(
        "startingtime", [(w, i) for w in scenarios for i in orders], lowBound=0, upBound=23, cat="Integer"
    )

    # creating a custom variable (s_hour) for the calculation of the starting time s_i
    # s_hour[h] = 1 if order i starts at hour h on factory j in scenario w, 0 otherwise (Stage 2 decision)
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

    twostageprob = pulp.LpProblem("Two-Stage Stochastic Problem", pulp.LpMaximize)


# ----------------------------------------------------------------------------
# SETTING THE CONSTRAINTS OF THE OPTIMIZATION PROBLEM
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# STAGE 1 CONSTRAINTS
# ----------------------------------------------------------------------------

    # ensures that each order is assigned to at most one factory
    order_x = {}
    for o, f in orders_2_factories.items(): 
        order_x[o] = pulp.lpSum([x_ji[(f_, o)] for f_ in f])
        twostageprob += (
            order_x[o] <= 1,
            f"sum(x_f{o})<=1",
        )

    # if fixed_x_decisions provided, fix Stage 1 variables (for EEV)
    if fixed_x_decisions is not None:
        for (j, i) in feasible_factories_orders:
            val = fixed_x_decisions.get((j, i), 0)
            if val is not None and val > 0.5:
                twostageprob += (x_ji[(j, i)] == 1, f"fix_x_{j}_{i}_accepted")
            else:
                twostageprob += (x_ji[(j, i)] == 0, f"fix_x_{j}_{i}_rejected")        


# ----------------------------------------------------------------------------
# STAGE 2 CONSTRAINTS (per scenario)
# ----------------------------------------------------------------------------

    # all the constraints in a loop for each scenario
    for w in scenarios:

        for i in orders:
            # factories that can execute order i
            valid_factories_for_i = orders_2_factories[i]
            # starting time of order i = sum of h * s_hour[h] over all feasible starts (only 1 is selected - the others are 0)
            hours_list = []
            for j in valid_factories_for_i:
                possible_starts = len(Eji[(w, j, i)])
                hours_list.extend([h * s_hour[(w, j, i)][h] for h in range(possible_starts)])
            # calculation of the starting time of each order i in scenario w
            twostageprob += (
                s_i[w, i] == pulp.lpSum(hours_list),
                f"calculate_start_time_for_order_{i}_scen{w}",
            )

        for j, i in feasible_factories_orders:
            # s_hour must sum to x_ji: if order i is accepted on factory j, exactly one start hour is chosen
            twostageprob += (
                pulp.lpSum(s_hour[(w, j, i)]) == x_ji[(j, i)],
                f"sum(s_hour_{j}_{i}_scen{w})==x_{j}{i}",
            )

            for k in orders:
                if k != i and (j, k) in feasible_factories_orders:
                    
                    # Big M Method constraints
                    # determines the shared-factory status between two orders i and k
                    twostageprob += (
                        x_ji[(j, i)] + x_ji[(j, k)] <= 1 + M * p_jik[(w, j, i, k)],
                        f"Big_M_constraint_1_for_order_{j}_{i}_{k}_scen{w}",
                    )
                    twostageprob += (
                        2 - M * (1 - p_jik[(w, j, i, k)]) <= x_ji[(j, i)] + x_ji[(j, k)],
                        f"Big_M_constraint_2_for_order_{j}_{i}_{k}_scen{w}",
                    )
                    # prevents time overlap during the execution of orders i and k
                    twostageprob += (
                        s_i[w, i] + duration[i] - s_i[w, k]
                        <= (M * (1 - q_jik[(w, j, i, k)])) + (M * (1 - p_jik[(w, j, i, k)])),
                        f"Big_M_constraint_3_for_order_{j}_{i}_{k}_scen{w}",
                    )
                    twostageprob += (
                        s_i[w, k] + duration[k] - s_i[w, i]
                        <= (M * q_jik[(w, j, i, k)]) + (M * (1 - p_jik[(w, j, i, k)])),
                        f"Big_M_constraint_4_for_order_{j}_{i}_{k}_scen{w}",
                    )


# ----------------------------------------------------------------------------
# OBJECTIVE FUNCTION
# ----------------------------------------------------------------------------

    # the goal is to maximize expected profit (revenue - energy spot cost) across all scenarios

    # total revenue of orders
    total_revenue = pulp.lpSum(
        [x_ji[(j, i)] * orders_revenue[i] for j, i in feasible_factories_orders]
    )

    # total energy cost
    expected_spot_cost = []
    for w in scenarios:
        # spot energy cost for scenario w
        scenario_hourly_cost = []
        for j, i in feasible_factories_orders:
            for h, e in enumerate(Eji[(w, j, i)]):
                scenario_hourly_cost.append(Eji[(w, j, i)][h] * s_hour[(w, j, i)][h])
        scenario_spot_cost = pulp.lpSum(scenario_hourly_cost)

        # weighted spot cost for scenario w
        expected_spot_cost.append(pscenarios[w] * scenario_spot_cost)

    # maximize total expected profit
    twostageprob += total_revenue - pulp.lpSum(expected_spot_cost), "Teliko katharo kerdos:"

    # control print output of the solver
    msg_flag = 1 if print_output else 0
    twostageprob.solve(pulp.PULP_CBC_CMD(msg=msg_flag))

    # save results
    expected_profit = pulp.value(twostageprob.objective)

    # extract Stage 1 decisions
    x_decisions = {(j, i): pulp.value(x_ji[(j, i)]) for (j, i) in feasible_factories_orders}

    return twostageprob, expected_profit, x_decisions, s_i, x_ji, Eji


# ----------------------------------------------------------------------------
# SOLVING THE PROBLEM & DISPLAY RESULTS
# ----------------------------------------------------------------------------

if __name__ == "__main__":

    # create output directory if it doesn't exist
    os.makedirs("stochastic_results", exist_ok=True)

    twostageprob, expected_profit, x_decisions, s_i, x_ji, Eji = solve_stochastic(print_output=True)

    # print and solve problem
    # print(twostageprob)
    twostageprob.solve()
    twostageprob.writeLP("stochastic_problem.lp")
    print("Status:", pulp.LpStatus[twostageprob.status])

    # output and display results
    for v in twostageprob.variables():
        print(v.name, "=", v.varValue)

    print(f"\nExpected Profit (Objective Value): {pulp.value(twostageprob.objective):.1f} €")

    # Stage 1 results: which orders were accepted and to which factories they were assigned  
    print("\n--- STAGE 1: Order Acceptance & Factory Assignment ---")
    for j, i in feasible_factories_orders:
        if x_ji[(j, i)].varValue is not None and x_ji[(j, i)].varValue > 0.5:
            print(f"  Order {i} → accepted & assigned to Factory {j}")

    # Stage 2 results: scheduling per scenario
    for w in scenarios:
        print(f"\n--- SCENARIO {w} (Probability: {pscenarios[w]}) ---")
        cons = 0
        scenario_revenue = 0
        scenario_spot_cost = 0

        for j, i in feasible_factories_orders:
            if x_ji[(j, i)].varValue is not None and x_ji[(j, i)].varValue > 0.5:
                start = int(pulp.value(s_i[(w, i)]))
                spot = Eji[(w, j, i)][start]            
                print(f"- Order {i} at Factory {j}: Start at {start:02d}:00 | Dur: {duration[i]}h | Cons/h: {consumption[i]} MW | Total Cons: {consumption[i] * duration[i]} MWh | Revenue: {orders_revenue[i]} € | Spot Cost: {spot:.1f}€ | Profit: {orders_revenue[i] - spot:.1f}€")
                cons += consumption[i] * duration[i]
                scenario_revenue += orders_revenue[i]
                scenario_spot_cost += spot

        scenario_profit = scenario_revenue - scenario_spot_cost
        print(f"\n  SUMMARY OF SCENARIO {w}:")
        print(f"  Total Consumption: {cons} MWh")
        print(f"  > Revenue:          {scenario_revenue} €")
        print(f"  > Spot Cost:        {scenario_spot_cost:.1f} €")
        print(f"  > FINAL PROFIT:     {scenario_profit:.1f} €")


    # save to excel file
    results_dict = {v.name: v.varValue for v in twostageprob.variables()}
    results_dict["Expected_Profit_Objective_€"] = pulp.value(twostageprob.objective)
    pd.DataFrame(results_dict.items(), columns=["Variable", "Value"]).to_excel(
        f"stochastic_results/problem_{scenario}.xlsx", engine="openpyxl", index=False
    )


# ----------------------------------------------------------------------------
# CREATION & DISPLAY OF THE DIAGRAMS
# ----------------------------------------------------------------------------

# save all graphs to a single PDF file
pdf_path = f"stochastic_results/problem_{scenario}_graphs.pdf"
with PdfPages(pdf_path) as pdf:

    for w in scenarios:
        current_prices = scenario_prices[w]

        # data preparation for standard variables
        plot_data = []

        for i in orders:
            # finding the factory where order i has been assigned
            assigned_factory = None
            for j in factories:
                if (
                    (j, i) in x_ji
                    and x_ji[(j, i)].varValue is not None
                    and x_ji[(j, i)].varValue > 0.5
                ):
                    assigned_factory = j
                    break

            # retrieves time info from Stage 2
            start_time = s_i[w, i].varValue
            dur = duration[i]

            # cost calculation
            cost = 0
            if start_time is not None:
                start_idx = int(start_time)
                if start_idx + dur <= 24:
                    cost = sum(current_prices[start_idx : start_idx + dur])

            # color determination
            if i in orders_type_a:
                color="red"
            else:
                color="blue"

            # save to plot_data only if order was accepted
            if assigned_factory is not None and start_time is not None:
                plot_data.append(
                    {
                        "order_id": i,
                        "order_label": f"Order {i}",
                        "start": start_time,
                        "duration": dur,
                        "cost": cost,
                        "color": color,
                        "factory_label": f"Factory {assigned_factory}",
                    }
                )

        # legend patches (shared across graphs)
        red_patch = mpatches.Patch(color="red", label="Type A Orders")
        blue_patch = mpatches.Patch(color="blue", label="Type B Orders")


        # 1st graph - Energy cost per order

        # figure creation with specific dimensions
        fig1, ax = plt.subplots(figsize=(12, 6))

        # exporting lists from plot_data
        p1_labels = [d["order_label"] for d in plot_data]
        p1_costs = [d["cost"] for d in plot_data]
        p1_colors = [d["color"] for d in plot_data]
        p1_factories = [d["factory_label"] for d in plot_data]

        bars = ax.bar(p1_labels, p1_costs, color=p1_colors,
                    edgecolor='black', linewidth=1.5, alpha=0.75)

        # add text inside bars
        for bar, factory_name, item in zip(bars, p1_factories, plot_data):
            height = bar.get_height()

            # factory label
            txt1 = ax.text(
                bar.get_x() + bar.get_width() / 2,
                height / 2 + 10,
                factory_name,
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=10
            )
            txt1.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground='black')])

            # total consumption label
            txt2 = ax.text(
                bar.get_x() + bar.get_width() / 2,
                height / 2 - (height/8),
                f"{consumption[item['order_id']] * duration[item['order_id']]} MWh",
                ha="center",
                va="center",
                color="white",
                fontweight="normal",
                fontsize=9
            )
            txt2.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground='black')])

        # graph formatting
        ax.set_ylabel("Energy Cost (€ per MWh)", fontsize=12, fontweight="bold")
        ax.set_title(f"Energy Cost per Order - Scenario {w} (p={pscenarios[w]}) - Colored by Type", fontsize=14, fontweight='bold')
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(handles=[red_patch, blue_patch], shadow=True,
                loc='upper center', bbox_to_anchor=(0.5, -0.12),
                ncol=2, framealpha=1)
        plt.tight_layout()
        pdf.savefig(fig1)
        plt.show()
        plt.close(fig1)


        # 2nd graph - Gantt Chart

        # figure creation with specific dimensions
        fig2, ax = plt.subplots(figsize=(14, 8))

        # drawing grid lines
        ax.grid(axis="y", linestyle="--", color="black", alpha=0.5, zorder=0)
        ax.grid(axis="x", linestyle=":", alpha=0.3, zorder=0)

        for item in plot_data:
            # creating bar
            ax.barh(
                y=item["order_label"],
                width=item["duration"],
                left=item["start"],
                color=item["color"],
                edgecolor="black",
                linewidth=1.5,
                alpha=0.85,
                zorder=3
            )

            # adjusting text size according to block size
            if item["duration"] <= 1:
                f_size = 8
                f_weight = "normal"
            else:
                f_size = 10
                f_weight = "bold"

            # adding text
            txt = ax.text(
                x=item["start"] + item["duration"] / 2,
                y=item["order_label"],
                s=item["factory_label"],
                ha="center",
                va="center",
                color="white",
                fontweight=f_weight,
                fontsize=f_size,
                zorder=4,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground='black')])

        # graph formatting
        ax.set_title(f"Order Scheduling (Gantt Chart) - Scenario {w} (p={pscenarios[w]})", fontsize=16, fontweight='bold')
        ax.set_xlabel("Hours of the Day", fontsize=12, fontweight="bold")
        ax.set_ylabel("Orders", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25))
        ax.invert_yaxis()  # order 0 at top
        ax.legend(handles=[red_patch, blue_patch], shadow=True,
                loc="upper center", bbox_to_anchor=(0.5, -0.12),
                ncol=2, framealpha=1)
        plt.tight_layout()
        pdf.savefig(fig2)
        plt.show()
        plt.close(fig2)


        # 3rd graph - Hourly energy cost profile

        # figure creation with specific dimensions
        fig3, ax = plt.subplots(figsize=(14, 7))

        # creating 2 tables (one for each factory type)
        hourly_cost_type_a = np.zeros(24)
        hourly_cost_type_b = np.zeros(24)
        hourly_cost_type_c = np.zeros(24)

        # run through the data we already have in plot_data
        for item in plot_data:
            # exporting the factory ID as integer
            f_id = int(item["factory_label"].split(" ")[1])
            # exporting the starting time and duration as integers
            start = int(item["start"])
            dur = int(item["duration"])

            # checking if factory is Type A or B
            is_type_a = f_id in factories_type_a
            is_type_b = f_id in factories_type_b

            # adding the energy cost for each hour that the factory operates
            for h in range(start, start + dur):
                if h < 24:
                    cost_at_h = current_prices[h]
                    if is_type_a:
                        hourly_cost_type_a[h] += cost_at_h
                    elif is_type_b:
                        hourly_cost_type_b[h] += cost_at_h
                    else:
                        hourly_cost_type_c[h] += cost_at_h

        # calculation of the two mean values
        total_hourly_cost = hourly_cost_type_a + hourly_cost_type_b + hourly_cost_type_c

        # 1st mean value: average over all 24 hours
        average_cost_24h = np.mean(total_hourly_cost)

        # 2nd mean value: average over active hours only
        active_hours = total_hourly_cost[total_hourly_cost > 0]
        average_cost_active = np.mean(active_hours) if len(active_hours) > 0 else 0

        # drawing the red/blue/green lines for Type A/B/C factories respectively
        ax.plot(
            range(24), hourly_cost_type_a, color="red", linewidth=2.5, label="Type A Factories"
        )
        ax.plot(
            range(24), hourly_cost_type_b, color="blue", linewidth=2.5, label="Type B Factories"
        )
        ax.plot(
            range(24), hourly_cost_type_c, color="green", linewidth=2.5, label="Type C Factories"
        )

        # drawing the dotted line of the average over all 24 hours
        ax.axhline(
            y=average_cost_24h,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Average Cost - All Hours ({average_cost_24h:.1f} € per MWh)",
        )

        # drawing the dotted line of the average over all 24 hours
        ax.axhline(
            y=average_cost_active,
            color="gray",
            linestyle="-.",
            linewidth=1.5,
            label=f"Average Cost - Active Hours ({average_cost_active:.1f} € per MWh)",
    )

        # graph formatting
        ax.set_title(f"Hourly Energy Cost Profile - Scenario {w} (p={pscenarios[w]})", fontsize=16, fontweight='bold')
        ax.set_xlabel("Hours of the day", fontsize=12, fontweight="bold")
        ax.set_ylabel("Energy Cost (€ per MWh)", fontsize=12, fontweight="bold")
        # axis setting
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24))
        ax.grid(True, linestyle="-", alpha=0.3)  # background grid
        # filling the area below the lines
        ax.fill_between(range(24), hourly_cost_type_a, color="red", alpha=0.1)
        ax.fill_between(range(24), hourly_cost_type_b, color="blue", alpha=0.1)
        ax.fill_between(range(24), hourly_cost_type_c, color="green", alpha=0.1)
        ax.legend(loc="upper center", fontsize=11, frameon=True, shadow=True,
                bbox_to_anchor=(0.5, -0.12), ncol=4)
        plt.tight_layout()
        pdf.savefig(fig3)
        plt.show()
        plt.close(fig3)


        # 4th graph - Combined schedule & energy price curve
        
        # figure creation with specific dimensions
        fig4, ax1 = plt.subplots(figsize=(14, 8))
        # creation of axis Y2
        ax2 = ax1.twinx() 

        # drawing the energy price curve
        ax2.plot(
            list(range(24)) + [23.9], 
            current_prices + [current_prices[-1]],    # repeat last value so the line completes at x=24
            color='orange', 
            linewidth=4, 
            alpha=0.8,
            label='Energy Price (€ per MWh)',
            zorder=2,    # behind the letters but in front of the grid
            drawstyle='steps-post'
        )

        # adjustments for axis Y2
        ax2.set_ylabel("Energy Price (€ per MWh)", fontsize=12, color='darkorange', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='darkorange')
        ax2.set_ylim(0, max(current_prices) + 50)

        # export all factories for the Y1 axis
        all_factories = sorted(factories)
        factory_labels = [f"Factory {f}" for f in all_factories]

        for item in plot_data:
            # finding the factory ID
            f_id = int(item["factory_label"].split()[1])
            # drawing bar
            ax1.barh(
                y=f_id, 
                width=item["duration"], 
                left=item["start"], 
                color=item["color"], 
                edgecolor='black',
                alpha=0.85,
                linewidth=1.5,
                height=0.6,
                zorder=1    # behind the curve
            )
            
            # adjusting text size
            if item["duration"] <= 1:
                f_size = 8
            else:
                f_size = 10

            # adding text "Order X"
            txt = ax1.text(
                x=item["start"] + item["duration"] / 2,
                y=f_id,
                s=item["order_label"],
                ha='center', 
                va='center', 
                color='white', 
                fontweight='bold', 
                fontsize=f_size, 
                zorder=3
            )
            # black outline around the text
            txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground='black')])

        # adjustments for axis Y1
        ax1.set_xlabel("Hours of Day", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Production Line", fontsize=12, fontweight="bold")
        ax1.set_yticks(all_factories)
        ax1.set_yticklabels(factory_labels, fontsize=11, fontweight='normal')

        # adjustments for axis X and grid
        ax1.set_xlim(0, 24)
        ax1.set_xticks(range(0, 25))
        ax1.grid(axis='x', linestyle=':', alpha=0.5)
        ax1.invert_yaxis()  # factory 0 at top
        ax1.set_title(f"Combined Schedule & Energy Cost Profile - Scenario {w} (p={pscenarios[w]})", fontsize=16, pad=20, fontweight='bold')

        # creation and placement of the custom legend of graph
        custom_lines = [
            mpatches.Patch(color='red', label='Type A Orders'),
            mpatches.Patch(color='blue', label='Type B Orders'),
            Line2D([0], [0], color='orange', lw=3, label='Energy Price')
        ]
        ax1.legend(handles=custom_lines, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, shadow=True)
        plt.tight_layout()
        pdf.savefig(fig4)
        plt.show()
        plt.close(fig4)

    print(f"\nResults saved to: stochastic_results/problem_{scenario}.xlsx")
    print(f"Graphs saved to:  stochastic_results/problem_{scenario}_graphs.pdf")