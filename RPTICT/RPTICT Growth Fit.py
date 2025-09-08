import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, differential_evolution
import matplotlib.cm as cm

# ------------------------------
# Fixed parameters
# ------------------------------
params_fixed = {
    "n": 1.101e-7,     # day^-1 cells^-1
    "s": 1.3e4,        # cells day^-1
    "p": 0.1245,       # day^-1
    "g": 2.019e7,      # cells
    "m": 3.422e-10,    # day^-1 cells^-1
    "d": 0.0412        # day^-1
}

# Initial conditions in CELLS
V0 = 100000        # tumor cells
E0 = 38            # effector T cells
y0 = np.array([V0, E0], dtype=float)

# Conversion: 100000 cells = 0.001 mL
cells_to_mL = 0.001 / 100000.0


# ------------------------------
# ODE system
# ------------------------------
def tumor_immune_system(t, y, lambda_dup, c_cr, p=params_fixed["p"], g=params_fixed["g"],
                        m=params_fixed["m"], d=params_fixed["d"], n=params_fixed["n"], s=params_fixed["s"]):
    V, E = y
    dVdt = lambda_dup * V * (1 - c_cr * V) - n * E * V
    dEdt = s + (p * E * V) / (g + V) - m * E * V - d * E
    return [dVdt, dEdt]


# ------------------------------
# Helper: simulate model
# ------------------------------
def simulate(days, lambda_dup, c_cr):
    sol = solve_ivp(tumor_immune_system, [0, max(days)], y0, args=(lambda_dup, c_cr),
                    t_eval=days, dense_output=False, rtol=1e-6, atol=1e-9)
    V = sol.y[0] * cells_to_mL  # convert to mL
    return V


# --- Base warm-gray palette ---
warm_grays = [
    "#f2f0ef", "#e6e3e1", "#d9d3ce", "#ccc4c0",
    "#bfb8b3", "#b3aaa5", "#a69e99", "#99908d",
    "#8c8580", "#7f7974", "#736c68", "#5c4f4b"
]


def sample_from_palette(palette, n_reps):
    """Return n_reps colors evenly spaced across a given palette."""
    if n_reps <= len(palette):
        # Just slice out evenly spaced indices
        idxs = np.linspace(0, len(palette)-1, n_reps).astype(int)
        return [palette[i] for i in idxs]
    else:
        # Interpolate if more replicates than palette entries
        idxs = np.linspace(0, len(palette)-1, n_reps)
        return [palette[int(round(i))] for i in idxs]


def get_colors(n_reps, dataset_name):
    if dataset_name == "UCLA 1":
        # Neutral grayscale
        return [str(0.15 + 0.75 * i/(n_reps-1)) if n_reps > 1 else "0.3" for i in range(n_reps)]
    elif dataset_name in ["UCLA 2", "CHUV"]:
        # Warm grays sampled across the array
        return sample_from_palette(warm_grays, n_reps)
    else:
        return ["0.3"] * n_reps


# ------------------------------
# Helper: simulate model at given timepoints (for fitting)
# ------------------------------
def simulate_at(days, lambda_dup, c_cr):
    sol = solve_ivp(tumor_immune_system, [min(days), max(days)], y0,
                    args=(lambda_dup, c_cr),
                    t_eval=days, rtol=1e-6, atol=1e-9)
    V = sol.y[0] * cells_to_mL
    return V


# ------------------------------
# Helper: simulate model over a continuous range (for plotting)
# ------------------------------
def simulate_range(t_min, t_max, lambda_dup, c_cr, n_points=100):
    t_span = np.linspace(t_min, t_max, n_points)
    sol = solve_ivp(tumor_immune_system, [t_min, t_max], y0,
                    args=(lambda_dup, c_cr),
                    t_eval=t_span, rtol=1e-6, atol=1e-9)
    V = sol.y[0] * cells_to_mL
    return t_span, V


# ------------------------------
# Loss function for fitting
# ------------------------------
def residuals(params, days, data):
    lambda_dup, c_cr = params
    pred = simulate(days, lambda_dup, c_cr)
    mask = ~np.isnan(data)
    return pred[mask] - data[mask]


# ------------------------------
# Global fitting function for one dataset
# ------------------------------
def fit_each_replicate_global(days, replicates, dataset_name):
    bounds = [(0, 2), (1e-12, 1e-6)]  # (lambda_dup, c_cr)

    fitted_params = []
    n_reps = len(replicates)

    # generate grayscale colormap from black → light gray
    colors = get_colors(n_reps, dataset_name)

    plt.figure(figsize=(7, 5))
    for idx, rep in enumerate(replicates):
        rep = np.array(rep, dtype=float)
        mask = ~np.isnan(rep)
        if mask.sum() < 2:
            print(f"Replicate {idx+1} in {dataset_name} has too few data points; skipping.")
            continue

        # --- Global optimization with differential evolution ---
        def obj(params):
            pred = simulate_at(days[mask], *params)
            return np.sum((pred - rep[mask])**2)

        result = differential_evolution(obj, bounds, polish=True, seed=42)
        lambda_fit, ccr_fit = result.x
        fitted_params.append((lambda_fit, ccr_fit))
        print(f"{dataset_name} rep {idx+1}: λ_dup={lambda_fit:.4f}, c_cr={ccr_fit:.2e} (loss={result.fun:.4e})")

        # --- Plot replicate with smooth curve in shades of gray ---
        t_min, t_max = np.min(days[mask]), np.max(days[mask])
        t_sim, V_fit = simulate_range(t_min, t_max, lambda_fit, ccr_fit, n_points=100)

        gray = colors[idx]
        plt.scatter(days, rep, color=gray, s=40, label=f"Replicate {idx+1}")
        plt.plot(t_sim, V_fit, '-', lw=2, color=gray)

    plt.xlabel("Days post inoculation")
    plt.ylabel("Tumor size (mL)")
    plt.title(f"Dataset {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return fitted_params


# ------------------------------
# Example usage with your datasets
# ------------------------------

# === DATASET 1 ===
days1 = np.array([0, 11, 17, 25, 28, 32, 35, 39, 42, 46, 49, 53, 56, 59, 62, 66, 73, 80, 87, 94, 101, 108, 115, 122, 129, 136, 143, 150, 157, 164, 171, 185, 192])
group1 = [
    [0.001, 0.23, 1.01, 4.02, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days1)-8),
    [0.001, 0.16, 0.51, 2.16, 4.41, np.nan, np.nan, np.nan] + [np.nan]*(len(days1)-8),
    [0.001, 0.08, 0.27, 1.07, 2.03, 3.71, np.nan, np.nan] + [np.nan]*(len(days1)-8),
    [0.001, 0.06, 0.14, 0.47, 0.99, 1.65, 2.15, 3.84] + [np.nan]*(len(days1)-8),
    [0.001, 0.02, 0.10, 0.44, 0.55, np.nan, np.nan, np.nan] + [np.nan]*(len(days1)-8),
    [0.001, 0.13, 0.31, 0.68, 1.04, 2.00, 3.05, np.nan] + [np.nan]*(len(days1)-8),
    [0.001, 0.20, 0.77, 3.49, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days1)-8),
    [0.001, 0.16, 0.53, 2.60, 4.42, np.nan, np.nan, np.nan] + [np.nan]*(len(days1)-8),
    [0.001, 0.23, 0.83, 2.84, 3.24, np.nan, np.nan, np.nan] + [np.nan]*(len(days1)-8)
]
fit_each_replicate_global(days1, group1, "UCLA 1")

# === DATASET 2 ===
days2 = np.array([0, 5, 8, 12, 15, 19, 22, 27, 33, 40, 48, 55, 63, 69, 76])
group2 = [
    [0.001, 0.38, 0.25, 0.34, 0.49, 1.67, 2.75] + [np.nan]*(len(days2)-7),
    [0.001, 0.077, 0.29, 0.98, 1.81, 1.65, 2.32] + [np.nan]*(len(days2)-7),
    [0.001, 0.13, 0.23, 0.26, 1.17, 1.61, np.nan] + [np.nan]*(len(days2)-7),
    [0.001, 0.13, 0.28, 0.50, 0.81, 1.87, 3.66] + [np.nan]*(len(days2)-7),
    [0.001, 0.19, 0.23, 0.45, 0.96, 2.15, np.nan] + [np.nan]*(len(days2)-7),
    [0.001, 0.078, 0.15, 0.39, 0.67, 1.60, 2.09] + [np.nan]*(len(days2)-7),
    [0.001, 0.094, 0.25, 0.28, 0.78, 1.98, np.nan] + [np.nan]*(len(days2)-7),
    [0.001, 0.21, 0.22, 0.28, 0.67, 1.81, 3.20] + [np.nan]*(len(days2)-7),
    [0.001, 0.13, 0.18, 0.35, 0.46, 1.21, 2.00] + [np.nan]*(len(days2)-7),
    [0.001, 0.14, 0.30, 0.42, 0.67, 1.93, np.nan] + [np.nan] * (len(days2)-7),
    [0.001, 0.088, 0.074, 0.28, 0.73, 1.47, 2.49] + [np.nan] * (len(days2)-7),
    [0.001, 0.067, 0.16, 0.33, 0.84, 1.15, 1.73] + [np.nan]*(len(days2)-7)
]
fit_each_replicate_global(days2, group2, "UCLA 2")

# === DATASET 3 ===
days3 = np.array([0, 8, 10, 14, 16])
group3 = [
    [0.001, 0.028, 0.35, 0.82, 0.88],
    [0.001, 0.26, 0.42, np.nan, np.nan],
    [0.001, 0.17, 0.56, np.nan, np.nan],
    [0.001, 0.19, 0.40, 1.52, 3.74],
    [0.001, 0.0063, 0.26, 1.14, 1.88]
]
fit_each_replicate_global(days3, group3, "CHUV")
