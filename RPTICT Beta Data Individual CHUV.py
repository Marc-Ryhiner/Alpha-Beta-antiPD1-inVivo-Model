
import matplotlib.pyplot as plt
import pandas as pd
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

grays = ["#d9d3ce", "#bfb8b3", "#a69e99", "#8c8580", "#736c68"]
blues = ["#c6d3e6", "#96afcf", "#688bb6", "#426494", "#2a476d"]
oranges = ["#f8d7b5", "#f2b97d", "#e68a3c", "#c8661f", "#9c4713"]
greens = ["#c9ddc6", "#9fc69a", "#74a96f", "#4e7f4a", "#335c32"]

colors = [grays, blues, oranges, greens]

# Plot each group individually with all replicates
for (group_name, replicates), color in zip(groups.items(), colors):
    plt.figure(figsize=(8,5))
    for i, (rep, c) in enumerate(zip(replicates, color), start=1):
        plt.plot(days[1:], rep[1:], marker="o", label=f"Replicate {i}", color=c)
    plt.xlabel("Days post inoculation")
    plt.ylabel("Tumor size (mL)")
    plt.title(f"{group_name}")
    plt.legend()
    plt.grid(True, which='both', ls='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
