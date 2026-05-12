import numpy as np
from scipy.integrate import solve_ivp
import json

# Initial values:
rhoRe1100 = 1 
rhoRe1210 = 0
rhoIm1210 = 0
rhoRe2200 = 0
rho0 = np.array([rhoRe1100, rhoRe1210, rhoIm1210, rhoRe2200])

Delta = 0
Omega0 = 5

def F(t, rho):
    rhoRe1100 = rho[0]
    rhoRe1210 = rho[1]
    rhoIm1210 = rho[2]
    rhoRe2200 = rho[3]
    Exp = np.exp(-(1 / 4) * (-5 + t) ** 2)
    return np.array([
        Exp * Omega0 * rhoIm1210,
        Delta * rhoIm1210,
        -Delta * rhoRe1210 + 0.5 * Exp * Omega0 * (-rhoRe1100 + rhoRe2200),
        -Exp * Omega0 * rhoIm1210
    ])

t_span = (0, 10)  # sec
t_eval = np.linspace(t_span[0], t_span[1], 500)  # points of output

sol = solve_ivp(F, t_span, rho0, t_eval=t_eval, rtol=1e-4)

# Map of resulting excitation as a function of Δd and Ω0
def excite_ion(newDelta, newOmega0):
    global Delta
    Delta = newDelta
    global Omega0
    Omega0 = newOmega0
    sol = solve_ivp(F, t_span, rho0, rtol=1e-4)
    rhoRe1100, rhoRe1210, rhoIm1210, rhoRe2200 = sol.y
    return rhoRe2200[-1] 


Delta_range = np.linspace(0, +3, 30)
Omega0_range = np.linspace(0, +8, 80)
Delta_range = np.linspace(0, +3, 60)
Omega0_range = np.linspace(0, +8, 160)
Delta_range = np.linspace(0, +3, 60)
Omega0_range = np.linspace(0, +8, 160)

Deltas, Omega0s = np.meshgrid(Delta_range, Omega0_range)

Excitation = np.vectorize(excite_ion)(Deltas, Omega0s)


with open('pulsed_excitation_map.json', "w") as f:
    json.dump({
        'Delta_detuning*tau_pulse max': Delta_range[-1],
        'OmegaRabi*tau_pulse max': Omega0_range[-1],
        'Excitation probability': Excitation.tolist()
    }, f, indent=1)