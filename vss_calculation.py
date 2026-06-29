'''
DIPLOMA THESIS — VALUE OF THE STOCHASTIC SOLUTION (VSS) CALCULATION
----------------------------------------------------------------------------
VSS = SP - EEV
VSS measures the benefit of using the stochastic solution (SP) instead
of applying the deterministic EV solution to all scenarios (EEV).
A higher VSS means the stochastic approach adds more value.
'''

from ev_calculation import EV_xji_decisions
from stochastic_problem_code import solve_stochastic


# solve SP (Stochastic Problem - no fixed decisions)
_, SP, _, _, _, _ = solve_stochastic(fixed_x_decisions=None, print_output=False)


# solve EEV (fixed Stage 1 decisions from EV)
_, EEV, _, _, _, _ = solve_stochastic(fixed_x_decisions=EV_xji_decisions, print_output=False)


# calculation of the VSS metric
VSS = SP - EEV


print("============================================================================")
print("                VALUE OF THE STOCHASTIC SOLUTION (VSS)")
print("============================================================================")
print(f"  SP  (Stochastic Problem)          = {SP:.1f} €")
print(f"  EEV (Expected Result of EV Sol.)  = {EEV:.1f} €")
print(f"  VSS (= SP - EEV)                  = {VSS:.1f} €")
print("============================================================================")