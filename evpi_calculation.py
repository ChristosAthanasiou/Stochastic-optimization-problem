'''
DIPLOMA THESIS — EXPECTED VALUE OF PERFECT INFORMATION (EVPI) CALCULATION
----------------------------------------------------------------------------
EVPI = WS - SP
EVPI measures the maximum amount the decision-maker would be willing to pay
for perfect information about which scenario will occur.
A higher EVPI means uncertainty is more costly.
'''

from ws_calculation import WS
from stochastic_problem_code import solve_stochastic


# solve SP (Stochastic Problem - no fixed decisions)
_, SP, _, _, _, _ = solve_stochastic(fixed_x_decisions=None, print_output=False)


# calculation of the EVPI metric
EVPI = WS - SP


print("============================================================================")
print("             EXPECTED VALUE OF PERFECT INFORMATION (EVPI)")
print("============================================================================")
print(f"  WS   (Wait-and-See)               = {WS:.1f} €")
print(f"  SP   (Stochastic Problem)          = {SP:.1f} €")
print(f"  EVPI (= WS - SP)                  = {EVPI:.1f} €")
print("============================================================================")