'''
DIPLOMA THESIS — DETERMINISTIC OPTIMIZATION PROBLEM
----------------------------------------------------------------------------
A single known energy price forecast is used (no uncertainty)
Objective: Maximize total profit from order execution
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

input_file = "deterministic_data.xlsx"

scenario = "scenario_0"

# orders setup from sheet
df_orders = pd.read_excel(input_file, sheet_name="Orders")
orders_type_a = df_orders[df_orders["type"] == "A"]["order_id"].tolist()
orders_type_b = df_orders[df_orders["type"] == "B"]["order_id"].tolist()
orders = sorted(df_orders["order_id"].tolist())     # all orders in ascending order

orders_revenue = dict(zip(df_orders["order_id"], df_orders["profit (€)"]))

duration_dict = df_orders.set_index("order_id")["duration (h)"].to_dict()
duration = [duration_dict[i] for i in orders]       # duration of execution of orders (in hours)

consumption_dict = df_orders.set_index("order_id")["consumption (MW/h)"].to_dict()
consumption = [consumption_dict[i] for i in orders]     # energy consumption of each order per hour (MW)

# factories setup from sheet
df_factories = pd.read_excel(input_file, sheet_name="Factories")
factories_type_a = df_factories[df_factories["type"] == "A"]["factory_id"].tolist()
factories_type_b = df_factories[df_factories["type"] == "B"]["factory_id"].tolist()
factories_type_c = df_factories[df_factories["type"] == "C"]["factory_id"].tolist()
factories = sorted(df_factories["factory_id"].tolist())     # all factories in ascending order

# energy price list setup from sheet
df_forecast = pd.read_excel(input_file, sheet_name="Forecast")
hour_cols = [f"hour_{h}" for h in range(24)]
# extract the first row (iloc[0]) and read all 24 columns
forecast_en = [df_forecast.iloc[0][h] for h in hour_cols]

# other constants
hours = list(range(0, 24))
# M variable for the Big M Method
M = 1000


# ----------------------------------------------------------------------------
# CREATING COMBINATIONS OF FACTORIES - ORDERS
# ----------------------------------------------------------------------------

# every combination of factories-orders and sorting the orders
factories_orders = [(j, i) for i in orders for j in factories]

# dictionary for every feasible order-factory
orders_2_factories = {
    i: [
        j
        for j in factories
        if (
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

def solve_deterministic(forecast_en, print_output=False):

    # creation of Eji list for every order
    Eji = {}
    for j, i in feasible_factories_orders:
        d = duration[i]
        cons = consumption[i]    
        Eji[(j, i)] = []
        for si in range(24):
            if si + d <= 24:
                cost = sum(forecast_en[si : si + d]) * cons
                Eji[(j, i)].append(cost)

    # defining indicative variables x, p & q
    x_ji = pulp.LpVariable.dicts("x", factories_orders, cat="Binary")
    p_jik = pulp.LpVariable.dicts(
        "p",
        [
            (j, i, k)
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
            (j, i, k)
            for i in orders
            for k in orders
            if i != k
            for j in set(orders_2_factories[i]).intersection(set(orders_2_factories[k]))
        ],
        cat="Binary",
    )

    # defining the starting time variable for the orders
    s_i = pulp.LpVariable.dicts(
        "startingtime", orders, lowBound=0, upBound=23, cat="Integer"
    )

    # creating a custom variable for the calculation of the starting time
    s_hour = {}
    for j, i in feasible_factories_orders:
        possible_starts = len(Eji[(j, i)])
        s_hour[(j, i)] = [
            pulp.LpVariable(f"s_hour_{j}_{i}_{h:02d}", cat="Binary")
            for h in range(possible_starts)
        ]


    # ----------------------------------------------------------------------------
    # DEFINING THE OPTIMIZATION PROBLEM
    # ----------------------------------------------------------------------------

    # define the linear problem
    deterministicproblem = pulp.LpProblem("Deterministic Problem", pulp.LpMaximize)

    # ----------------------------------------------------------------------------
    # SETTING THE CONSTRAINTS OF THE OPTIMIZATION PROBLEM & DEFINE THE OBJECTIVE FUNCTION
    # ----------------------------------------------------------------------------

    # an order can be assigned to a factory or not
    order_x = {}
    for o, f in orders_2_factories.items():
        # ensures that each order is assigned to a maximum of one factory
        order_x[o] = pulp.lpSum([x_ji[(f_, o)] for f_ in f])
        deterministicproblem += (
            order_x[o] <= 1,
            f"sum(x_f{o})<=1",
        )

    for i in orders:
        # factories that can execute order i
        valid_factories_for_i = orders_2_factories[i]
        # sum of s_i[i] for each factory (only 1 is selected - the others are 0)
        hours_list = []
        for j in valid_factories_for_i:
            possible_starts = len(Eji[(j, i)])
            hours_list.extend([h * s_hour[(j, i)][h] for h in range(possible_starts)])
        # calculation of the starting time of each order i
        deterministicproblem += (
            s_i[i] == pulp.lpSum(hours_list),
            f"calculate_start_time_for_order_{i}",
        )

    for j, i in feasible_factories_orders:
        # ensuring the proper function of the custom variable s_hour
        deterministicproblem += (
            pulp.lpSum(s_hour[(j, i)]) == x_ji[(j, i)],
            f"sum(s_hour_{j}_{i})==x_{j}{i}",
        )
        for k in orders:
            if k != i and (j, k) in feasible_factories_orders:
                # Big M Method constraints
                # p indicates whether two orders are assigned to the same factory
                deterministicproblem += (
                    x_ji[(j, i)] + x_ji[(j, k)] <= 1 + M * p_jik[(j, i, k)],
                    f"Big_M_constraint_1_for_orders_{j}_{i}_{k}",
                )
                deterministicproblem += (
                    2 - M * (1 - p_jik[(j, i, k)]) <= x_ji[(j, i)] + x_ji[(j, k)],
                    f"Big_M_constraint_2_for_orders_{j}_{i}_{k}",
                )
                # ensures a non-overlapped-in-time execution sequence between two orders i and k
                deterministicproblem += (
                    s_i[i] + duration[i] - s_i[k]
                    <= (M * (1 - q_jik[(j, i, k)])) + (M * (1 - p_jik[(j, i, k)])),
                    f"Big_M_constraint_3_for_orders_{j}_{i}_{k}",
                )
                deterministicproblem += (
                    s_i[k] + duration[k] - s_i[i]
                    <= (M * q_jik[(j, i, k)]) + (M * (1 - p_jik[(j, i, k)])),
                    f"Big_M_constraint_4_for_orders_{j}_{i}_{k}",
                )


# ----------------------------------------------------------------------------
# OBJECTIVE FUNCTION
# ----------------------------------------------------------------------------

    # the goal is to minimize energy cost AND maximize profit

    # total energy cost
    total_cost = []
    for j, i in feasible_factories_orders:
        for h, e in enumerate(Eji[(j, i)]):
            total_cost.append(-1 * e * s_hour[(j, i)][h])

    # total profit of orders placed in factories
    for o in orders:
        total_cost.append(orders_revenue[o] * order_x[o])

    # sum of (total energy cost - total profit)
    deterministicproblem += pulp.lpSum(total_cost), "Total energy cost"

    # control print output of the solver
    msg_flag = 1 if print_output else 0
    deterministicproblem.solve(pulp.PULP_CBC_CMD(msg=msg_flag))

    # save results
    profit = pulp.value(deterministicproblem.objective)
    
    # extract Stage 1 decisions
    x_decisions = {}
    for j, i in feasible_factories_orders:
        x_decisions[(j, i)] = pulp.value(x_ji[(j, i)])
        
    return deterministicproblem, profit, x_decisions, s_i, x_ji, Eji


# ----------------------------------------------------------------------------
# SOLVING THE PROBLEM & DISPLAY RESULTS
# ----------------------------------------------------------------------------

if __name__ == "__main__":

    # create output directory if it doesn't exist
    os.makedirs("deterministic_results", exist_ok=True)

    deterministicproblem, final_profit, x_decisions, s_i, x_ji, Eji = solve_deterministic(forecast_en, print_output=True)

    # print and solve problem
    # print(deterministicproblem)
    deterministicproblem.writeLP("deterministic_problem.lp")
    deterministicproblem.solve()
    print("Status:", pulp.LpStatus[deterministicproblem.status])

    # output and display results
    for v in deterministicproblem.variables():
        print(v.name, "=", v.varValue)


    print(f"Value of the objective function (Total Profit) = {pulp.value(deterministicproblem.objective)} €")
    print("\n--- RESULTS ---")
    total_revenue = 0
    total_spot_cost = 0

    for j, i in feasible_factories_orders:
        if x_ji[(j, i)].varValue is not None and x_ji[(j, i)].varValue > 0.5:
            start = int(s_i[i].varValue)
            spot = Eji[(j, i)][start]
            print(f"- Order {i} at Factory {j}: Start at {start:02d}:00 | Dur: {duration[i]}h | Cons/h: {consumption[i]} MW | Total Consumption: {consumption[i] * duration[i]} MWh | Revenue: {orders_revenue[i]} € | Spot Cost: {spot:.1f} € | Profit: {orders_revenue[i] - spot:.1f} €")
            total_revenue += orders_revenue[i]
            total_spot_cost += spot

    print(f"\n  SUMMARY:")
    print(f"  > Revenue:       {total_revenue} €")
    print(f"  > Spot Cost:     {total_spot_cost:.1f} €")
    print(f"  > FINAL PROFIT:  {total_revenue - total_spot_cost:.1f} €")
    
    
    # save to excel file
    results_dict = {v.name: v.varValue for v in deterministicproblem.variables()}
    results_dict["Total_Profit_Objective_€"] = pulp.value(deterministicproblem.objective)
    pd.DataFrame(results_dict.items(), columns=["Variable", "Value"]).to_excel(
        f"deterministic_results/problem_{scenario}.xlsx", engine="openpyxl", index=False
    )


# ----------------------------------------------------------------------------
# CREATION & DISPLAY OF THE DIAGRAMS
# ----------------------------------------------------------------------------


    # data preparation for standard variables

    plot_data = []

    for i in orders:
        # finding the factory where the order i has been assigned
        assigned_factory = None
        for j in factories:
            if (
                (j, i) in x_ji
                and x_ji[(j, i)].varValue is not None
                and x_ji[(j, i)].varValue > 0.5
            ):
                assigned_factory = j
                break

        # retrieves time info
        start_time = s_i[i].varValue
        dur = duration[i]

        # cost calculation
        cost = 0
        if start_time is not None:
            start_idx = int(start_time)
            if start_idx + dur <= 24:
                cost = sum(forecast_en[start_idx : start_idx + dur])

        # color determination
        if i in orders_type_a:
            color = "red"
        else:
            color = "blue"

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

    # save all graphs to a single PDF file
    pdf_path = f"deterministic_results/problem_{scenario}_graphs.pdf"
    with PdfPages(pdf_path) as pdf:

        # 1st graph - Energy cost of each factory

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
                height / 2,
                factory_name,
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=10,
            )
            txt1.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground='black')])


            # total consumption label
            txt2 = plt.text(
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
        ax.set_ylabel("Energy Cost (€/Mwhr)", fontsize=12, fontweight="bold")
        ax.set_title("Energy Cost of each factory - Colored by type of orders", fontsize=14, fontweight='bold')
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
                zorder=3,
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
        ax.set_title("Order Scheduling per Hour (Gantt Chart)", fontsize=16, fontweight='bold')
        ax.set_xlabel("Hours of the Day", fontsize=12, fontweight="bold")
        ax.set_ylabel("Orders", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25))
        ax.invert_yaxis()  # order 0 at top
        ax.legend(handles=[red_patch, blue_patch], shadow=True,
                loc='upper center', bbox_to_anchor=(0.5, -0.12),
                ncol=2, framealpha=1)
        plt.tight_layout()
        pdf.savefig(fig2)
        plt.show()
        plt.close(fig2)


        # 3rd graph - Chart of the energy cost over time

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
            # exporting the starting time and duration as integer
            start = int(item["start"])
            dur = int(item["duration"])

            # checking if factory is Type A or B
            is_type_a = f_id in factories_type_a
            is_type_b = f_id in factories_type_b

            # adding the energy cost for each hour that the factory operates
            for h in range(start, start + dur):
                if h < 24:
                    cost_at_h = forecast_en[h]
                    if is_type_a:
                        hourly_cost_type_a[h] += cost_at_h
                    elif is_type_b:
                        hourly_cost_type_b[h] += cost_at_h
                    else:
                        hourly_cost_type_c[h] += cost_at_h

        # calculation of mean value
        total_hourly_cost = hourly_cost_type_a + hourly_cost_type_b + hourly_cost_type_c
        average_cost = np.mean(total_hourly_cost)

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

        # drawing the dotted line of the average hourly cost
        ax.axhline(
            y=average_cost,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Average Hourly Cost ({average_cost:.1f} €/Mwhr)",
        )

        # graph formatting
        ax.set_title("Hourly Energy Cost Profile per Factory Type", fontsize=16, fontweight='bold')
        ax.set_xlabel("Hours of the day", fontsize=12, fontweight="bold")
        ax.set_ylabel("Energy Cost (€/Mwhr)", fontsize=12, fontweight="bold")
        # axis setting
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24))
        ax.grid(True, linestyle="-", alpha=0.3)  # background grid
        # filling the area below the lines
        ax.fill_between(range(24), hourly_cost_type_a, color="red", alpha=0.1)
        ax.fill_between(range(24), hourly_cost_type_b, color="blue", alpha=0.1)
        ax.fill_between(range(24), hourly_cost_type_c, color="green", alpha=0.1)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
                ncol=4, fontsize=11, frameon=True, shadow=True)
        plt.tight_layout()
        pdf.savefig(fig3)
        plt.show()
        plt.close(fig3)


        # 4th graph - Combo chart of order sorting & "energy cost" curve

        # figure creation with specific dimensions
        fig4, ax1 = plt.subplots(figsize=(14, 8))
        # creation of axis Y2
        ax2 = ax1.twinx() 

        # drawing the "energy cost" curve
        ax2.plot(
            list(range(24)) + [23.9], 
            forecast_en + [forecast_en[-1]],    # repeat last value so the line completes at x=24
            color='orange', 
            linewidth=4, 
            alpha=0.8,
            label='Energy Price (€/MWh)',
            zorder=2,    # behind the letters but in front of the grid
            drawstyle='steps-post'

        )

        # adjustments for axis Y2
        ax2.set_ylabel("Energy Price (€/MWh)", fontsize=12, color='darkorange', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='darkorange')
        ax2.set_ylim(0, max(forecast_en) + 50)

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
                height=0.6,
                linewidth=1.5,
                zorder=1    # behind the curve
            )
            
            # adjusting text size
            if item["duration"] <= 1:
                f_size = 8
                f_weight = 'bold'
            else:
                f_size = 10
                f_weight = 'bold'

            # adding text "Order X"
            txt = ax1.text(
                x=item["start"] + item["duration"] / 2,
                y=f_id,
                s=item["order_label"],
                ha='center', 
                va='center', 
                color='white', 
                fontweight=f_weight, 
                fontsize=f_size, 
                zorder=3
            )
            # black outline around the text
            txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground='black')])

        # adjustments for axis Y1
        ax1.set_xlabel("Hours of Day", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Production Line", fontsize=12, fontweight='bold')
        ax1.set_yticks(all_factories)
        ax1.set_yticklabels(factory_labels, fontsize=11, fontweight='normal')

        # adjustments for axis X and grid
        ax1.set_xlim(0, 24)
        ax1.set_xticks(range(0, 25))
        ax1.grid(axis='x', linestyle=':', alpha=0.5)
        ax1.invert_yaxis()  # factory 0 at top
        plt.title("Combined Schedule & Energy Cost Profile", fontsize=16, pad=20, fontweight='bold')

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


    print(f"\nResults saved to: deterministic_results/problem_{scenario}.xlsx")
    print(f"Graphs saved to:  deterministic_results/problem_{scenario}_graphs.pdf")


# expose shared data for external use
def get_problem_data():
    # returns all problem data needed by external scripts
    return {
        "orders": orders,
        "orders_type_a": orders_type_a,
        "orders_type_b": orders_type_b,
        "orders_revenue": orders_revenue,
        "duration": duration,
        "consumption": consumption,
        "factories": factories,
        "factories_type_a": factories_type_a,
        "factories_type_b": factories_type_b,
        "factories_type_c": factories_type_c,
        "orders_2_factories": orders_2_factories,
        "feasible_factories_orders": feasible_factories_orders,
    }