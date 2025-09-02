import matplotlib.pyplot as plt
import numpy as np

# Define days (same for all groups)
days = [5, 8, 12, 15, 19, 22, 27, 33, 40, 48, 55, 63, 69, 76]


# Replicates
groupA = [
    [0.38, 0.25, 0.34, 0.49, 1.67, 2.75] + [np.nan]*(len(days)-6),
    [0.077, 0.29, 0.98, 1.81, 1.65, 2.32] + [np.nan]*(len(days)-6),
    [0.13, 0.23, 0.26, 1.17, 1.61, np.nan] + [np.nan]*(len(days)-6),
    [0.13, 0.28, 0.50, 0.81, 1.87, 3.66] + [np.nan]*(len(days)-6),
    [0.19, 0.23, 0.45, 0.96, 2.15, np.nan] + [np.nan]*(len(days)-6),
    [0.078, 0.15, 0.39, 0.67, 1.60, 2.09] + [np.nan]*(len(days)-6),
    [0.094, 0.25, 0.28, 0.78, 1.98, np.nan] + [np.nan]*(len(days)-6),
    [0.21, 0.22, 0.28, 0.67, 1.81, 3.20] + [np.nan]*(len(days)-6),
    [0.13, 0.18, 0.35, 0.46, 1.21, 2.00] + [np.nan]*(len(days)-6),
    [0.14, 0.30, 0.42, 0.67, 1.93, np.nan] + [np.nan] * (len(days)-6),
    [0.088, 0.074, 0.28, 0.73, 1.47, 2.49] + [np.nan] * (len(days)-6),
    [0.067, 0.16, 0.33, 0.84, 1.15, 1.73] + [np.nan]*(len(days)-6)
]

groupB = [
    [0.076, 0.094, 0.19, 0.12, 0.22, 0.29, 0.69, 2.06, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.14, 0.27, 0.52, 0.70, 0.94, 1.43, 3.41, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.13, 0.23, 0.45, 0.46, 0.89, 2.19, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.18, 0.20, 0.43, 0.60, 1.26, 2.24, 2.16, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.065, 0.058, 0.096, 0.12, 0.055, 0.031, 0.039, 0.051, 0.0071, 0.0079, 0.0054, 0.0040, 0.0048, 0],
    [0.21, 0.25, 0.48, 0.41, 0.46, 0.58, 1.67, 3.87, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.063, 0.16, 0.11, 0.075, 0.024, 0.021, 0.018, 0, 0, 0, 0, 0, 0, 0],
    [0.10, 0.19, 0.23, 0.61, 1.32, 2.35, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.14, 0.22, 0.37, 0.69, 0.93, 1.58, 2.32, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.095, 0.17, 0.27, 0.27, 0.49, 0.69, 1.49, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.091, 0.15, 0.19, 0.16, 0.082, 0.065, 0.11, 0.065, 0.075, 0.18, 0.29, 0.54, 1.77, np.nan],
    [0.11, 0.10, 0.26, 0.34, 1.00, 1.36, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
]

groupC = [
    [0.076, 0.12, 0.081, 0.076, 0.084, 0.14, 0.31, 1.32] + [np.nan]*(len(days)-8),
    [0.15, 0.34, 0.58, 0.89, 1.08, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-8),
    [0.049, 0.13, 0.26, 0.41, 0.75, 1.24, np.nan, np.nan] + [np.nan]*(len(days)-8),
    [0.10, 0.17, 0.33, 0.60, 1.35, 2.06, np.nan, np.nan] + [np.nan]*(len(days)-8),
    [0.11, 0.15, 0.26, 0.35, 0.94, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-8),
    [0.19, 0.25, 0.51, 1.07, 1.55, 2.02, np.nan, np.nan] + [np.nan]*(len(days)-8),
    [0.10, 0.10, 0.17, 0.39, 0.48, 0.76, 1.14, 2.31] + [np.nan]*(len(days)-8),
    [0.13, 0.25, 0.20, 0.30, 0.60, 0.99, np.nan, np.nan] + [np.nan]*(len(days)-8),
    [0.10, 0.20, 0.32, 0.74, 0.97, 1.43, np.nan, np.nan] + [np.nan]*(len(days)-8),
    [0.086, 0.14, 0.90, 0.28, 0.49, 0.71, 1.14, np.nan] + [np.nan]*(len(days)-8),
    [0.22, 0.14, 0.22, 0.61, 0.60, 1.08, 1.62, np.nan] + [np.nan]*(len(days)-8),
    [0.10, 0.16, 0.34, 0.35, 0.47, 0.88, 1.84, np.nan] + [np.nan]*(len(days)-8)
]

groupD = [
    [0.062, 0.20, 0.27, 0.37, 0.31, 0.45, 0.53, 1.17, 2.77, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-13),
    [0.15, 0.25, 0.45, 0.75, 1.50, 1.35, 2.08, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-13),
    [0.093, 0.065, 0.089, 0.092, 0.15, 0.11, 0.16, 0.27, 0.32, 0.45, 0.94, 2.01, 5.60] + [np.nan]*(len(days)-13),
    [0.17, 0.30, 0.32, 0.73, 0.76, 1.24, 1.20, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-13),
    [0.065, 0.14, 0.30, 0.28, 0.27, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-13),
    [0.19, 0.24, 0.48, 0.51, 0.30, 0.32, 0.28, 0.42, 1.27, 2.72, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-13),
    [0.11, 0.26, 0.62, 0.71, 0.78, 1.19, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-13),
    [0.11, 0.26, 0.62, 0.53, 0.38, 0.44, 0.64, 0.78, 1.55, 3.58, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-13),
    [0.075, 0.19, 0.38, 0.50, 0.21, 0.15, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-13),
    [0.11, 0.15, 0.30, 0.46, 0.45, 0.72, 1.06, 2.48, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-13),
    [0.11, 0.20, 0.33, 0.61, 0.64, 0.83, 1.50, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-13),
    [0.10, 0.26, 0.42, 0.43, 0.40, 0.49, 1.07, 2.41, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-13)
]

groups = {
    "Control": groupA,
    "αPD-1": groupB,
    "Lu-177": groupC,
    "Lu-177/αPD-1": groupD
}

# Colors for each group
group_colors = ["#8c8580", "#426494", "#c8661f", "#4e7f4a"]  # one color per group

# Compute averages ignoring NaNs
averages = {}
for group_name, replicates in groups.items():
    averages[group_name] = np.nanmean(replicates, axis=0)

markers = ['o', 's', '^', 'd']

# Plot all averaged curves on the same plot
plt.figure(figsize=(10,6))
for (group_name, avg), color, marker in zip(averages.items(), group_colors, markers):
    plt.plot(days, avg, marker=marker, label=group_name, color=color, linewidth=2)

plt.xlabel("Days post inoculation")
plt.ylabel("Tumor size (mL)")
plt.title("Tumor Growth for Lutetium Combination Treatment")
plt.legend()
plt.grid(True, which='both', ls='--', linewidth=0.5)
plt.tight_layout()
plt.show()