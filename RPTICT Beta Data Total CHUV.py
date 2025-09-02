import matplotlib.pyplot as plt
import numpy as np

# Define days (same for all groups)
days = [0, 8, 10, 14, 16]

# Replicates
groupA = [
    [0.001, 0.028, 0.35, 0.82, 0.88],
    [0.001, 0.26, 0.42, np.nan, np.nan],
    [0.001, 0.17, 0.56, np.nan, np.nan],
    [0.001, 0.19, 0.40, 1.52, 3.74],
    [0.001, 0.0063, 0.26, 1.14, 1.88]
]

groupB = [
    [0.001, 0.30, 0.49, 0.68, 1.26],
    [0.001, 0.16, 0.26, 0.63, 0.40],
    [0.001, 0.075, 0.15, 0.47, 0.25],
    [0.001, 0.13, 0.38, 0.38, 0.22],
    [0.001, 0.13, 0.22, 0.22, 0.19]
]

groupC = [
    [0.001, 0.064, 0.21, np.nan, np.nan],
    [0.001, 0.24, 0.23, np.nan, np.nan],
    [0.001, 0.14, 0.19, 0.33, 0.26],
    [0.001, 0.31, 0.50, 0.30, 0.17],
    [0.001, 0.12, 0.28, 0.17, 0.094]
]

groupD = [
    [0.001, 0.075, 0.57, 0.33, 0.23],
    [0.001, 0.16, 0.20, 0.23, 0.050],
    [0.001, 0.075, 0.28, 0.36, 0.44],
    [0.001, 0.31, 0.79, 0.51, 0.45],
    [0.001, 0.24, 0.23, 0.23, 0.16]
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