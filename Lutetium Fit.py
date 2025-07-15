import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Fixed parameters (excluding those to be fitted)
base_params = {
    'k_on': 131.4, 'k_off': 1.611, 'k_int': 0.7602, 'k_rel': 2.138,
    'lambda_Lu': 4.323e-3, 'S_b': 0.374, 'S_c': 0.713,
    'Ncell': 1e5
}

# Experimental dataset
data = np.array([
    [0.0025, 1e5, 0.802, 0.044],
    [0.005, 2e5, 0.754, 0.034],
    [0.0075, 3e5, 0.681, 0.035],
    [0.01, 4e5, 0.582, 0.032],
    [0.0125, 5e5, 0.510, 0.043],
    [0.025, 1e6, 0.494, 0.024],
    [0.05, 2e6, 0.387, 0.026],
    [0.125, 5e6, 0.227, 0.015],
])

# ODE system
def ode_system(t, y, ci0, A0, alpha, R, S_i, p):
    ci, cb, cc, A, D, S = y
    k_on, k_off, k_int, k_rel = p['k_on'], p['k_off'], p['k_int'], p['k_rel']
    λ = p['lambda_Lu']
    Sb, Sc, N = p['S_b'], p['S_c'], p['Ncell']
    dci = k_off * cb - k_on * (R - cb) * ci
    dcb = k_on * (R - cb) * ci + k_rel * cc - (k_off + k_int) * cb
    dcc = k_int * cb - k_rel * cc
    dA = -λ * A
    dD = (A / ci0) * (S_i * ci0 + (Sb * cb + Sc * cc) / N)
    dS = -alpha * dD * np.exp(-alpha * D)
    return [dci, dcb, dcc, dA, dD, dS]

# Simulation function
def simulate_S3(ci0, A0, alpha, R, S_i):
    y0 = [ci0, 0, 0, A0, 0, 1]
    sol = solve_ivp(
        ode_system, [0, 3], y0,
        args=(ci0, A0, alpha, R, S_i, base_params),
        t_eval=[3],
        method='LSODA',
        rtol=1e-6, atol=1e-8
    )
    return float(sol.y[-1, 0])

# Objective function
def objective(theta):
    alpha, R, S_i = theta
    chi2 = 0
    for ci0, A0, S_obs, sigma in data:
        S_model = simulate_S3(ci0, A0, alpha, R, S_i)
        chi2 += ((S_model - S_obs) / sigma) ** 2
    return chi2

# Initial guesses and bounds: [alpha, R, S_i]
x0 = [0.15, 0.0003, 8e-8]  # initial guesses
bounds = [(0, 1),          # alpha (Gy⁻¹)
          (1e-5, 1e-2),    # R (nmol/mL)
          (1e-9, 1e-6)]    # S_i (Gy/Bq·h)

# Optimization
res = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B')
alpha_fit, R_fit, Si_fit = res.x
print("α = {alpha_fit:.4f} Gy⁻¹")
print("R = {R_fit:.6f} nmol")
print("S_i = {Si_fit:.2e} Gy/Bq·h")

# Generate model prediction
A0_range = np.logspace(3, np.log10(6e6), 200)
ci0_range = np.logspace(np.log10(2.5e-5), np.log10(0.15), 200)
S_continuous = np.array([
    simulate_S3(ci0, A0, alpha_fit, R_fit, Si_fit)
    for ci0, A0 in zip(ci0_range, A0_range)
])

# Plotting
plt.figure(figsize=(8, 5))
plt.errorbar(np.concatenate(([0], data[:, 1] / 1e6)), np.concatenate(([1], data[:, 2])), yerr=np.concatenate(([0], data[:, 3])),
             fmt='^', capsize=5, label='Survival data', color='#8B5E3C')
plt.plot(A0_range / 1e6, S_continuous, label='Survival fit', color='#C4A484')
plt.xlabel('Initial Activity (MBq)')
plt.ylabel('Surviving Fraction after 3h of Treatment')
plt.title('Medium Effect, Receptor Concentration, and Linear Effect Fit to Cell Survival')
plt.xlim([0, 6])
plt.ylim([0.2, 1])
plt.grid(True, ls="--", linewidth=0.5)
plt.legend(ncol=2)
plt.tight_layout()
plt.show()