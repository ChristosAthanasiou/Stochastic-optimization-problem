'''
DIPLOMA THESIS — EXPECTED RESULT OF THE EV SOLUTION (EEV) CALCULATION
----------------------------------------------------------------------------
The EV Stage 1 decisions (order acceptance) are fixed,
and the stochastic problem is solved for Stage 2 only.
EEV = Expected Profit when using EV decisions under all scenarios
'''

from ev_calculation import EV_xji_decisions
from stochastic_problem_code import solve_stochastic

print("\n############################################################################")
print("\n============================================================================")
print("            EEV (Expected Result of EV Solution) CALCULATION")
print("============================================================================")

print("\nFixed Stage 1 decisions (from EV):")
for (j, i), val in EV_xji_decisions.items():
    if val is not None and val > 0.5:
        print(f"  Order {i} → assigned to Factory {j}")

_, EEV, _, _, _, _ = solve_stochastic(fixed_x_decisions=EV_xji_decisions, print_output=False)

print("\n============================================================================")
print(f"         EEV (Expected Result of EV Solution) = {EEV:.1f} €")
print("============================================================================")