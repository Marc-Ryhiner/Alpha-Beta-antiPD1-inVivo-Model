import numpy as np
from scipy.integrate import solve_ivp, cumtrapz
import matplotlib.pyplot as plt

# Base parameters (no fitting)
base_params = {
    'k_on': 131.4, 'k_off': 1.611, 'k_int': 0.7602, 'k_rel': 2.138,
    'lambda_Lu': 4.323e-3, 'S_b': 0.374, 'S_c': 0.713,
    'Ncell': 1e5, 'R': 0.000814
}

# Constants for dose calculation
S_i = 1.83e-7  # Gy/Bq·h (interstitial)
alpha = 0.177  # Gy^-1

# ODE system for Phase 1 (0–3h)
def ode_system(t, y, ci0, A0, alpha, S_i, p):
    ci, cb, cc, A, D, S = y
    k_on, k_off, k_int, k_rel = p['k_on'], p['k_off'], p['k_int'], p['k_rel']
    λ = p['lambda_Lu']
    dci = k_off * cb - k_on * (p['R'] - cb) * ci
    dcb = k_on * (p['R'] - cb) * ci + k_rel * cc - (k_off + k_int) * cb
    dcc = k_int * cb - k_rel * cc
    dA = -λ * A
    dD = (A / ci0) * (S_i * ci0 + (p['S_b'] * cb + p['S_c'] * cc) / p['Ncell'])
    dS = -alpha * dD * np.exp(-alpha * D)
    return [dci, dcb, dcc, dA, dD, dS]

# ODE system for Phase 2 (3–171h)
def ode_system_2(t, y, ci0, A0, alpha, S_i, p):
    ci, cb, cc, A, D, S = y
    k_on, k_off, k_int, k_rel = p['k_on'], p['k_off'], p['k_int'], p['k_rel']
    λ = p['lambda_Lu']
    exp_decay = np.exp(0.0277 * t)
    dci = k_off * cb - k_on * (p['R'] * np.exp(t * 0.0277) - cb) * ci
    dcb = k_on * (p['R'] * np.exp(t * 0.0277) - cb) * ci + k_rel * cc - k_off * cb - k_int * cb
    dcc = k_int * cb - k_rel * cc
    dA = -λ * A
    dD = (A / ci0) * (p['S_b'] * (cb + ci) + p['S_c'] * cc) / (p['Ncell'] * exp_decay)
    dS = -alpha * dD * np.exp(-alpha * D)
    return [dci, dcb, dcc, dA, dD, dS]

# Simulation function
def simulate_dose_and_dose_rate(ci0, A0, alpha, S_i, p):
    y0 = [ci0, 0, 0, A0, 0, 1]
    t_eval1 = np.linspace(0, 3, 100)
    sol1 = solve_ivp(
        ode_system, [0, 3], y0,
        args=(ci0, A0, alpha, S_i, p),
        t_eval=t_eval1,
        method='BDF', rtol=1e-6, atol=1e-8
    )

    # Phase 2
    ci1, cb1, cc1, A1, D1, S1 = sol1.y[:, -1]
    ci2 = cb1 + cc1
    A2 = A1 * cc1 / (ci1 + cb1 + cc1) + A1 * cb1 / (ci1 + cb1 + cc1)
    y0_phase2 = [0, cb1, cc1, A2, D1, S1]
    t_eval2 = np.linspace(3, 171, 500)
    sol2 = solve_ivp(
        ode_system_2, [0, 168], y0_phase2,
        args=(ci2, A2, alpha, 0, p),
        t_eval=np.linspace(0, 168, 500),
        method='BDF', rtol=1e-6, atol=1e-8
    )

    # Merge all
    t_total = np.concatenate((sol1.t, sol2.t + 3))
    D_total = np.concatenate((sol1.y[4], sol2.y[4]))

    # Dose rate via finite difference
    dose_rate = np.gradient(D_total, t_total)

    return t_total, D_total, dose_rate

# Run the simulation
ci0 = 0.05
A0 = 2e6  # Bq
time, dose, dose_rate = simulate_dose_and_dose_rate(ci0, A0, alpha, S_i, base_params)

# Plot: Dose rate and dose
fig, ax1 = plt.subplots(figsize=(8, 5))

color1 = 'tab:blue'
plt.axvline(3, color='gray', linestyle='--', label='Cell Plating')
ax1.set_xlabel('Time (h)')
ax1.set_ylabel('Dose Rate (Gy per h)', color=color1)
ax1.plot(time, dose_rate, color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(0.1, 200)
ax1.set_ylim(6e-4, 0.6)
ax1.yaxis.grid(True, which='both', ls='--', linewidth=0.5, color='#aab7cc')
ax1.xaxis.grid(True, which='both', ls='--', linewidth=0.5)
plt.legend(loc='center left')

# Second axis for cumulative dose
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Accumulated Dose (Gy)', color=color2)
ax2.plot(time, dose, color=color2, linestyle='--', label='Accumulated Dose')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_yscale('log')
ax2.set_ylim(0.04, 6)
ax2.yaxis.grid(True, which='both', ls='--', linewidth=0.5, color='#d2a8a8')

plt.title('Dose Rate and Accumulated Dose during Lu-177 Treatment $A_0=2 MBq$')
fig.tight_layout()
plt.show()
