import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

# ------------------------------
# Fixed parameters
# ------------------------------
params_fixed = {
    "n": 1.101e-7,   # day^-1 cells^-1
    "s": 1.3e4,      # cells day^-1
    "p": 0.1245,     # day^-1
    "g": 2.019e7,    # cells
    "m": 3.422e-10,  # day^-1 cells^-1
    "d": 0.0412      # day^-1
}

# Initial conditions
V0 = 100000
E0 = 38
y0 = np.array([V0, E0], dtype=float)
cells_to_mL = 0.001 / 100000.0


# ------------------------------
# ODE system (s is fitted)
# ------------------------------
def tumor_immune_system(t, y, lambda_dup, c_cr, s_2, dataset,
                        d=params_fixed["d"], n=params_fixed["n"], s_1=params_fixed["s"],
                        g=params_fixed["g"], m=params_fixed["m"], p=params_fixed["p"]):
    V, E = y

    # --- Piecewise param handling ---
    if dataset.startswith("UCLA"):
        if t <= 13:
            s_eff = s_1
            m_eff = m
        elif t <= 33:
            s_eff = s_2
            m_eff = 0
        else:
            s_eff = s_1
            m_eff = m

    elif dataset.startswith("CHUV"):
        if t <= 11:
            s_eff = s_1
            m_eff = m
        else:
            s_eff = s_2
            m_eff = 0


    # --- ODEs ---
    dVdt = lambda_dup * V * (1 - c_cr * V) - n * E * V
    dEdt = s_eff + (p * E * V) / (g + V) - m_eff * E * V - d * E
    return [dVdt, dEdt]


# ------------------------------
# Simulation helpers
# ------------------------------
def simulate(days, lambda_dup, c_cr, s, dataset):
    sol = solve_ivp(
        tumor_immune_system,
        [0, max(days)],
        y0,
        args=(lambda_dup, c_cr, s, dataset),
        t_eval=days,
        rtol=1e-6,
        atol=1e-9
    )
    return sol.y[0] * cells_to_mL


# ------------------------------
# Global fit function (λ_dup, c_cr, s are fitted; p,g fixed)
# ------------------------------
def fit_each_replicate_global(days, replicates, dataset_name, lambda_prev, ccr_prev):
    # 99% prediction intervals for λ_dup and c_cr
    lambda_lower, lambda_upper = np.percentile(lambda_prev, [0.5, 99.5])
    ccr_lower, ccr_upper = np.percentile(ccr_prev, [0.5, 99.5])

    fitted_params = []

    # --- Choose color scheme depending on dataset ---
    if dataset_name.startswith("UCLA 1"):
        cmap = plt.cm.get_cmap("Blues")
        colors = [cmap(i) for i in np.linspace(0.9, 0.3, len(replicates))]
    elif dataset_name.startswith("UCLA 2") or dataset_name.startswith("CHUV"):
        cmap = [
            "#e0e7f1", "#c6d3e6", "#aec1da", "#96afcf",
            "#7e9dc1", "#688bb6", "#5375a6", "#426494",
            "#355580", "#2a476d", "#203857", "#162940"
        ]
        colors = [cmap[int(i)] for i in np.linspace(0, len(cmap)-1, len(replicates))]

    for idx, rep in enumerate(replicates):
        rep = np.array(rep, dtype=float)
        mask = ~np.isnan(rep)
        if mask.sum() < 2:
            print(f"Replicate {idx + 1} has too few points; skipping.")
            continue

        # --- Objective function for fitting λ_dup, c_cr, s ---
        def obj(params):
            lambda_dup, c_cr, s = params
            pred = simulate(
                days[mask], lambda_dup, c_cr, s, dataset_name
            )
            return np.sum((pred - rep[mask]) ** 2)

        # --- Bounds for DE ---
        bounds = [
            (lambda_lower, lambda_upper),  # λ_dup
            (ccr_lower, ccr_upper),        # c_cr
            (0, 3e5)    # s (adjust range if needed)
        ]
        if dataset_name == "UCLA 1" and idx == 3:
            bounds = [
                (0.1, 0.2),  # λ_dup
                (ccr_lower, ccr_upper),  # c_cr
                (0, 3e5)  # s (adjust range if needed)
            ]
        if dataset_name == "UCLA 2" and idx == 0:
            bounds = [
                (0.2, 0.3),  # λ_dup
                (ccr_lower, ccr_upper),  # c_cr
                (0, 3e5)  # s (adjust range if needed)
            ]
        if dataset_name == "UCLA 2" and (idx == 1):
            bounds = [
                (lambda_lower, lambda_upper),  # λ_dup
                (ccr_lower, ccr_upper),  # c_cr
                (0, 3e5)  # s (adjust range if needed)
            ]
        if dataset_name == "UCLA 2" and (idx == 4 or idx == 6):
            bounds = [
                (0.2, 0.3),  # λ_dup
                (ccr_lower, ccr_upper),  # c_cr
                (0, 3e6)  # s (adjust range if needed)
            ]
        if dataset_name == "UCLA 2" and (idx == 5):
            bounds = [
                (0.2, 0.3),  # λ_dup
                (ccr_lower, ccr_upper),  # c_cr
                (0, 3e5)  # s (adjust range if needed)
            ]
        if dataset_name == "UCLA 2" and (idx == 9):
            bounds = [
                (0.2, 0.3),  # λ_dup
                (ccr_lower, ccr_upper),  # c_cr
                (0, 3e5)  # s (adjust range if needed)
            ]
        if dataset_name == "UCLA 2" and (idx == 10):
            bounds = [
                (0.1, 0.3),  # λ_dup
                (1e-13, 1e-12),  # c_cr
                (0, 3e6)  # s (adjust range if needed)
            ]
        if dataset_name == "CHUV":
            bounds = [
                (lambda_lower, lambda_upper), #λ_dup
                (ccr_lower, ccr_upper),       # c_cr
                (0, 3e6)                      # s (adjust range if needed)
            ]


        result = differential_evolution(obj, bounds, polish=True, seed=42)
        lambda_fit, ccr_fit, s_fit = result.x
        fitted_params.append((lambda_fit, ccr_fit, s_fit))

        print(f"Replicate {idx + 1}: λ_dup={lambda_fit:.4f}, c_cr={ccr_fit:.2e}, "
              f"s={s_fit:.4f}, "
              f"loss={result.fun:.4e}")

        # --- Plot ---
        t_sim = np.linspace(min(days[mask]), max(days[mask]), 100)
        V_fit = simulate(
            t_sim, lambda_fit, ccr_fit, s_fit, dataset_name
        )

        plt.scatter(days, rep, s=40, label=f"Replicate {idx + 1}", color=colors[idx])
        plt.plot(t_sim, V_fit, lw=2, color=colors[idx])

    plt.xlabel("Days post inoculation")
    plt.ylabel("Tumor size (mL)")
    plt.title(f"Dataset {dataset_name}")
    plt.legend()
    plt.show()

    return fitted_params


# ------------------------------
# Example usage
# ------------------------------

# Provide previously fitted λ_dup and c_cr for each dataset (arrays)
lambdadup_UCLA1 = [0.4332, 0.3305, 0.3024, 0.2577, 0.2930, 0.2726, 0.4144, 0.3486, 0.4196]
ccr_UCLA1 = [2.03e-09, 5.70e-10, 1.23e-09, 1.13e-09, 1.20e-08, 1.35e-09, 2.19e-09, 1.15e-09, 2.75e-09]
# === DATASET 1 ===
days1 = np.array([0, 11, 17, 25, 28, 32, 35, 39, 42, 46, 49, 53, 56, 59, 62, 66, 73, 80, 87, 94, 101, 108, 115, 122, 129, 136, 143, 150, 157, 164, 171, 185, 192])
group1 = [
    [0.001, 0.17, 0.57, 1.40, 2.17, 2.64, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days1)-15),
    [0.001, 0.13, 0.64, 1.53, 2.67, 3.45, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days1)-15),
    [0.001, 0.12, 0.26, 0.47, 0.81, 1.31, 2.38, 3.89, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days1)-15),
    [0.001, 0.06, 0.11, 0.07, 0.07, 0.07, 0.15, 0.23, 0.32, 0.44, 0.62, 0.99, 1.28, 1.80, 2.57] + [np.nan]*(len(days1)-15),
    [0.001, 0.07, 0.35, 0.93, 1.37, 2.57, 3.83, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days1)-15),
    [0.001, 0.06, 0.22, 0.45, 0.71, 1.31, 1.97, 2.85, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days1)-15),
    [0.001, 0.08, 0.14, 0.41, 0.52, 0.84, 1.21, 2.03, 2.91, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days1)-15),
    [0.001, 0.14, 0.31, 0.69, 1.27, 2.41, 3.72, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days1)-15),
]
fitted_params_UCLA1 = fit_each_replicate_global(days1, group1, "UCLA 1",
                                                lambdadup_UCLA1, ccr_UCLA1)

lambdadup_UCLA2 = [0.4354, 0.6408, 0.5347, 0.4376, 0.5038, 0.4688, 0.4781, 0.4383, 0.4238, 0.4714, 0.4398, 0.4914]
ccr_UCLA2 = [2.50e-09, 4.73e-09, 5.31e-09, 1.72e-09, 3.59e-09, 4.10e-09, 3.41e-09, 2.09e-09, 3.54e-09, 3.37e-09, 3.00e-09, 5.73e-09]
# === DATASET 2 ===
days2 = np.array([0, 5, 8, 12, 15, 19, 22, 27, 33, 40, 48, 55, 63, 69, 76])
group2 = [
    [0.001, 0.076, 0.094, 0.19, 0.12, 0.22, 0.29, 0.69, 2.06, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.001, 0.14, 0.27, 0.52, 0.70, 0.94, 1.43, 3.41, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.001, 0.13, 0.23, 0.45, 0.46, 0.89, 2.19, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.001, 0.18, 0.20, 0.43, 0.60, 1.26, 2.24, 2.16, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.001, 0.065, 0.058, 0.096, 0.12, 0.055, 0.031, 0.039, 0.051, 0.0071, 0.0079, 0.0054, 0.0040, 0.0048, 0],
    [0.001, 0.21, 0.25, 0.48, 0.41, 0.46, 0.58, 1.67, 3.87, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.001, 0.063, 0.16, 0.11, 0.075, 0.024, 0.021, 0.018, 0, 0, 0, 0, 0, 0, 0],
    [0.001, 0.10, 0.19, 0.23, 0.61, 1.32, 2.35, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.001, 0.14, 0.22, 0.37, 0.69, 0.93, 1.58, 2.32, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.001, 0.095, 0.17, 0.27, 0.27, 0.49, 0.69, 1.49, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.001, 0.091, 0.15, 0.19, 0.16, 0.082, 0.065, 0.11, 0.065, 0.075, 0.18, 0.29, 0.54, 1.77, np.nan],
    [0.001, 0.11, 0.10, 0.26, 0.34, 1.00, 1.36, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
]
fitted_params_UCLA2 = fit_each_replicate_global(days2, group2, "UCLA 2",
                                                lambdadup_UCLA2, ccr_UCLA2)

lambdadup_CHUV = [0.6232, 0.7905, 0.6517, 0.5419, 0.5491]
ccr_CHUV = [1.02e-08, 1.96e-08, 2.01e-09, 6.07e-10, 3.36e-09]
# === DATASET 3 ===
days3 = np.array([0, 8, 10, 14, 16])
group3 = [
    [0.001, 0.30, 0.49, 0.68, 1.26],
    [0.001, 0.16, 0.26, 0.63, 0.40],
    [0.001, 0.075, 0.15, 0.47, 0.25],
    [0.001, 0.13, 0.38, 0.38, 0.22],
    [0.001, 0.13, 0.22, 0.22, 0.19]
]
fitted_params_CHUV = fit_each_replicate_global(days3, group3, "CHUV",
                                                lambdadup_CHUV, ccr_CHUV)