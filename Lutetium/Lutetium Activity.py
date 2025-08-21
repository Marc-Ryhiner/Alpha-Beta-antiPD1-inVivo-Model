import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Base parameters (no fitting)
base_params = {
    'k_on': 131.4, 'k_off': 1.611, 'k_int': 0.7602, 'k_rel': 2.138,
    'lambda_Lu': 4.323e-3, 'S_b': 0.374, 'S_c': 0.713,
    'Ncell': 1e5, 'R': 0.000814
}


# ODE system for Phase 1 (0–3h)
def ode_system(t, y, ci0, A0, alpha, S_i, p):
    ci, cb, cc, A, D, S = y
    k_on, k_off, k_int, k_rel = p['k_on'], p['k_off'], p['k_int'], p['k_rel']
    λ = p['lambda_Lu']
    dci = k_off * cb - k_on * (p['R'] - cb) * ci
    dcb = k_on * (p['R'] - cb) * ci + k_rel * cc - (k_off + k_int) * cb
    dcc = k_int * cb - k_rel * cc
    dA = -λ * A
    dD = 0  # not used for plotting activity
    dS = 0
    return [dci, dcb, dcc, dA, dD, dS]


# ODE system for Phase 2 (3–171h)
def ode_system_2(t, y, ci0, A0, alpha, S_i, p):
    ci, cb, cc, A, D, S = y
    k_on, k_off, k_int, k_rel = p['k_on'], p['k_off'], p['k_int'], p['k_rel']
    λ = p['lambda_Lu']
    dci = k_off * cb - k_on * (p['R'] * np.exp(t * 0.0277) - cb) * ci
    dcb = k_on * (p['R'] * np.exp(t * 0.0277) - cb) * ci + k_rel * cc - k_off * cb - k_int * cb
    dcc = k_int * cb - k_rel * cc
    dA = -λ * A
    dD = 0  # not used for plotting activity
    dS = 0
    return [dci, dcb, dcc, dA, dD, dS]


# Simulate activity over time for A0 = 2 MBq
def simulate_activity_over_time(ci0, A0, alpha, S_i, p):
    y0 = [ci0, 0, 0, A0, 0, 1]

    # Phase 1 (0–3 h)
    t_eval1 = np.linspace(0, 3, 100)
    sol1 = solve_ivp(
        ode_system, [0, 3], y0,
        args=(ci0, A0, alpha, S_i, p),
        t_eval=t_eval1,
        method='BDF', rtol=1e-6, atol=1e-8
    )

    # Phase 2 (3–171 h)
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

    # Merge time and activity
    t_total = np.concatenate((sol1.t, sol2.t + 3))
    A_total = np.concatenate((sol1.y[3], sol2.y[3]))

    return t_total, A_total


# Run simulation and plot
ci0 = 0.05  # for example, from dataset (or arbitrary)
A0 = 2e6  # 2 MBq in Bq
alpha = 0.15  # dummy value (not relevant here)
S_i = 1e-7  # dummy value (not relevant here)

time, activity = simulate_activity_over_time(ci0, A0, alpha, S_i, base_params)

plt.figure(figsize=(8, 5))
plt.plot(time, activity / 1e6, label='Activity', color='darkblue')
plt.axvline(3, color='gray', linestyle='--', label='Cell Plating')
plt.xscale('log')
plt.yscale('log')
plt.xlim([0.1, 200])
plt.ylim(0.01, 3)
plt.xlabel('Time (h)')
plt.ylabel('Activity (MBq)')
plt.title('Lutetium Activity during Lu-177 Treatment for $A_0=2 MBq$')
plt.grid(True, which='both', ls='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()