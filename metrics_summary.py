'''
DIPLOMA THESIS — STOCHASTIC PROGRAMMING PERFORMANCE METRICS SUMMARY
----------------------------------------------------------------------------
Summarizes all key metrics of the stochastic programming analysis:
  SP   (Stochastic Problem / Recourse Problem)
  WS   (Wait-and-See)
  EV   (Expected Value)
  EEV  (Expected Result of the EV Solution)
  VSS  (Value of the Stochastic Solution)  = SP - EEV
  EVPI (Expected Value of Perfect Information) = WS - SP
----------------------------------------------------------------------------
Key relationships:
  EEV ≤ SP ≤ WS
  VSS ≥ 0
  EVPI ≥ 0
'''



from ws_calculation import WS
from ev_calculation import EV, EV_xji_decisions
from stochastic_problem_code import solve_stochastic

# solve SP (stochastic programming problem)
_, SP, _, _, _, _ = solve_stochastic(fixed_x_decisions=None, print_output=False)

# solve EEV (fixed Stage 1 decisions from EV)
_, EEV, _, _, _, _ = solve_stochastic(fixed_x_decisions=EV_xji_decisions, print_output=False)

# calculate VSS and EVPI
VSS  = SP - EEV
EVPI = WS - SP

# summary output
print("============================================================================")
print("               STOCHASTIC PROGRAMMING PERFORMANCE METRICS")
print("============================================================================")
print(f"  SP   (Stochastic Problem)          = {SP:.1f} €")
print(f"  WS   (Wait-and-See)                = {WS:.1f} €")
print(f"  EV   (Expected Value)              = {EV:.1f} €")
print(f"  EEV  (Expected Result of EV Sol.)  = {EEV:.1f} €")
print("----------------------------------------------------------------------------")
print(f"  VSS  (= SP - EEV)                  = {VSS:.1f} €")
print(f"  EVPI (= WS - SP)                   = {EVPI:.1f} €")
print("============================================================================")
print(f"\n  Verification 1: EEV ≤ SP ≤ WS")
print(f"  {EEV:.1f} ≤ {SP:.1f} ≤ {WS:.1f} → {'Correct' if EEV <= SP <= WS else 'False'}")
print(f"\n  Verification 2: VSS ≥ 0 & EVPI ≥ 0")
print(f"  {VSS:.1f} ≥ 0 & {EVPI:.1f} ≥ 0 → {'Correct' if VSS >= 0 and EVPI >= 0 else 'False'}")
print("============================================================================")