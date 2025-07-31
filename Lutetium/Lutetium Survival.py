import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Base parameters
base_params = {
    'k_on': 131.4, 'k_off': 1.611, 'k_int': 0.7602, 'k_rel': 2.138,
    'lambda_Lu': 4.323e-3, 'S_b': 0.374, 'S_c': 0.713,
    'Ncell': 1e5, 'R': 0.000814
}

# ODE system for Phase 1
def ode_system(t, y, ci0, A0, alpha, S_i, p):
    ci, cb, cc, A, D, S = y
    k_on, k_off, k_int, k_rel = p['k_on'], p['k_off'], p['k_int'], p['k_rel']
    λ = p['lambda_Lu']
    dci = k_off * cb - k_on * (p['R'] - cb) * ci
    dcb = k_on * (p['R'] - cb) * ci + k_rel * cc - (k_off + k_int) * cb
    dcc = k_int * cb - k_rel * cc
    dA = -λ * A
    dD = (A / ci0) * (S_i * ci0 + (p['S_b'] * cb + p['S_c'] * cc) / p['Ncell'])
    dS = -alpha * dD * S
    return [dci, dcb, dcc, dA, dD, dS]

# ODE system for Phase 2
def ode_system_2(t, y, ci0, A0, alpha, S_i, p):
    ci, cb, cc, A, D, S = y
    k_on, k_off, k_int, k_rel = p['k_on'], p['k_off'], p['k_int'], p['k_rel']
    λ = p['lambda_Lu']
    exp_decay = np.exp(0.0277 * t)
    dci = k_off * cb - k_on * (p['R'] * np.exp(t * 0.0277) - cb) * ci
    dcb = k_on * (p['R'] * np.exp(t * 0.0277) - cb) * ci + k_rel * cc - (k_off + k_int) * cb
    dcc = k_int * cb - k_rel * cc
    dA = -λ * A
    dD = (A / ci0) * (p['S_b'] * (cb + ci) + p['S_c'] * cc) / (p['Ncell'] * exp_decay)
    dS = -alpha * dD * S
    return [dci, dcb, dcc, dA, dD, dS]

# Full simulation
def simulate_survival_over_time(ci0, A0, alpha, S_i, p):
    y0 = [ci0, 0, 0, A0, 0, 1]
    t_eval1 = np.linspace(0, 3, 100)
    sol1 = solve_ivp(
        ode_system, [0, 3], y0,
        args=(ci0, A0, alpha, S_i, p),
        t_eval=t_eval1, method='BDF', rtol=1e-6, atol=1e-8
    )

    ci1, cb1, cc1, A1, D1, S1 = sol1.y[:, -1]
    ci2 = cb1 + cc1
    A2 = A1 * (cc1 + cb1) / (ci1 + cb1 + cc1)
    y0_phase2 = [0, cb1, cc1, A2, D1, S1]
    t_eval2 = np.linspace(3, 171, 500)
    sol2 = solve_ivp(
        ode_system_2, [0, 168], y0_phase2,
        args=(ci2, A2, alpha, 0, p),
        t_eval=np.linspace(0, 168, 500),
        method='BDF', rtol=1e-6, atol=1e-8
    )

    t_total = np.concatenate((sol1.t, sol2.t + 3))
    A_total = np.concatenate((sol1.y[3], sol2.y[3]))
    D_total = np.concatenate((sol1.y[4], sol2.y[4]))
    S_total = np.concatenate((sol1.y[5], sol2.y[5]))

    # Derivatives
    dS_dt = -np.gradient(S_total, t_total)  # survival rate
    dD_dt = np.gradient(D_total, t_total)   # dose rate
    clinical_effect_rate = alpha * dD_dt
    clinical_effect = alpha * D_total

    return t_total, dS_dt, clinical_effect_rate, S_total, clinical_effect

# Parameters
ci0 = 0.05
A0 = 2e6  # 2 MBq
alpha = 0.177
S_i = 1.83e-7

# Run simulation
time, survival_rate, clinical_effect_rate, survival, clinical_effect = simulate_survival_over_time(ci0, A0, alpha, S_i, base_params)

# Plotting
fig, ax1 = plt.subplots(figsize=(8, 6))

# Left axis: survival rate and clinical effect rate
color1 = 'tab:orange'
ax1.plot(time, survival_rate, label='Survival rate', color=color1)
ax1.plot(time, clinical_effect_rate, label=r'Clinical effect rate', color=color1, linestyle='-.')
ax1.set_xlabel('Time (h)', color='black')
ax1.set_ylabel('Rate (per h)', color=color1)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.axvline(3, color='gray', linestyle='--', label='Cell plating')
ax1.set_xlim([0.1, 200])
ax1.set_ylim([3e-5, 2e-1])
ax1.yaxis.grid(True, which='both', linestyle='--', linewidth=0.5, color='#d4b8a0')
ax1.xaxis.grid(True, which='both', linestyle='--', linewidth=0.5)

# Right axis: survival and clinical effect
ax2 = ax1.twinx()
color1 = 'tab:green'
ax2.plot(time, survival, label='Survival', color=color1)
ax2.plot(time, clinical_effect, label='Clinical effect', color=color1, linestyle='-.')
ax2.set_ylabel('Total', color=color1)
ax2.set_yscale('log')
ax2.set_ylim([0.007, 1.1])
ax2.tick_params(axis='y', labelcolor=color1)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5, color='#a3c9a3')

# Custom legend handles
custom_lines = [
    Line2D([0], [0], color='black', linestyle='-', label='Survival'),
    Line2D([0], [0], color='black', linestyle='-.', label='Clinical Effect'),
    Line2D([0], [0], color='gray', linestyle='--', label='Cell Plating'),
]

# Add to figure
ax1.legend(handles=custom_lines, loc='center left')

plt.title('Survival and Clinical Effect during Lu-177 Treatment')
plt.tight_layout()
plt.show()
