
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define days (same for all groups)
days = [11, 17, 25, 28, 32, 35, 39, 42, 46, 49, 53, 56, 59, 62, 66, 73, 80, 87, 94, 101, 108, 115, 122, 129, 136, 143, 150, 157, 164, 171, 185, 192]


# Replicates
groupA = [
    [0.23, 1.01, 4.02, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-7),
    [0.16, 0.51, 2.16, 4.41, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-7),
    [0.08, 0.27, 1.07, 2.03, 3.71, np.nan, np.nan] + [np.nan]*(len(days)-7),
    [0.06, 0.14, 0.47, 0.99, 1.65, 2.15, 3.84] + [np.nan]*(len(days)-7),
    [0.02, 0.10, 0.44, 0.55, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-7),
    [0.13, 0.31, 0.68, 1.04, 2.00, 3.05, np.nan] + [np.nan]*(len(days)-7),
    [0.20, 0.77, 3.49, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-7),
    [0.16, 0.53, 2.60, 4.42, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-7),
    [0.23, 0.83, 2.84, 3.24, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-7)
]

groupB = [
    [0.17, 0.57, 1.40, 2.17, 2.64, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-14),
    [0.13, 0.64, 1.53, 2.67, 3.45, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-14),
    [0.12, 0.26, 0.47, 0.81, 1.31, 2.38, 3.89, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-14),
    [0.06, 0.11, 0.07, 0.07, 0.07, 0.15, 0.23, 0.32, 0.44, 0.62, 0.99, 1.28, 1.80, 2.57] + [np.nan]*(len(days)-14),
    [0.07, 0.35, 0.93, 1.37, 2.57, 3.83, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-14),
    [0.06, 0.22, 0.45, 0.71, 1.31, 1.97, 2.85, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-14),
    [0.08, 0.14, 0.41, 0.52, 0.84, 1.21, 2.03, 2.91, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-14),
    [0.14, 0.31, 0.69, 1.27, 2.41, 3.72, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-14),
]

groupC = [
    [0.17, 0.53, 2.03, 2.98, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-15),
    [0.054, 0.13, 0.48, 0.74, 1.14, 1.58, 1.62, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-15),
    [0.080, 0.32, 1.37, 2.12, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-15),
    [0.045, 0.093, 0.53, 0.75, 1.10, 1.56, 2.61, 3.51, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-15),
    [0.046, 0.064, 0.037, 0.035, 0.034, 0.054, 0.072, 0.080, 0.099, 0.086, 0.11, 0.14, 0.19, 0.22, 0.21] + [np.nan]*(len(days)-15),
    [0.079, 0.28, 0.98, 1.91, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-15),
    [0.077, 0.38, 0.76, 1.06, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-15),
    [0.13, 0.47, 1.85, 2.44, 4.78, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [np.nan]*(len(days)-15),
]

groupD = [
    [0.040, 0.070, 0.090, 0.13, 0.14, 0.19, 0.30, 0.40, 0.58, 0.83, 1.46, 2.26, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan],
    [0.044, 0.11, 0.22, 0.27, 0.33, 0.56, 1.20, 1.50, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan],
    [0.024, 0.017, 0.0096, 0.0062, 0.0044, 0.0060, 0.0080, 0.0092, 0.0045, 0.0034, 0.0045, 0.0075, 0.0024, 0.0061,
     0.0055, 0.0035, 0.0040, 0.0030, 0.0032, 0.0037, 0.0045, 0.0038, 0.0046, 0.0044, 0.0027, 0.0033, 0.0041, 0.0027,
     0.0021, 0.0025, 0, 0],
    [0.10, 0.19, 0.092, 0.057, 0.026, 0.025, 0.013, 0.011, 0.012, 0.0095, 0.0091, 0.0094, 0.013, 0.013, 0.011, 0.015,
     0.017, 0.019, 0.023, 0.029, 0.047, 0.055, 0.052, 0.068, 0.077, 0.098, 0.18, 0.23, 0.37, 0.88, np.nan, np.nan],
    [0.11, 0.30, 0.31, 0.40, 0.89, 1.62, 3.27, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan],
    [0.051, 0.14, 0.21, 0.35, 0.78, 1.23, 1.95, 2.79, 3.87, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan],
    [0.11, 0.23, 0.28, 0.38, 0.76, 1.33, 2.35, 3.29, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan],
    [0.048, 0.041, 0.038, 0.051, 0.081, 0.14, 0.28, 0.43, 0.48, 0.98, 1.59, 2.12, 2.22, 2.86, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan]
]

groups = {
    "Control": groupA,
    "αPD-1": groupB,
    "Ac-225": groupC,
    "Ac-225/αPD-1": groupD
}

grays = [str(i/10) for i in range(1, 10)]
blues = [(0, 0, i/9) for i in range(2, 10)]
oranges = [(i/9, 0.4*i/9, 0) for i in range(2, 10)]
greens = [(0, i/9, 0) for i in range(2, 10)]

colors = [grays, blues, oranges, greens]

# Plot each group individually with all replicates
for (group_name, replicates), color in zip(groups.items(), colors):
    plt.figure(figsize=(8,5))
    for i, (rep, c) in enumerate(zip(replicates, color), start=1):
        plt.plot(days, rep, marker="o", label=f"Replicate {i}", color=c)
    plt.xlabel("Days post inoculation")
    plt.ylabel("Tumor size (mL)")
    plt.title(f"{group_name}")
    plt.legend()
    plt.grid(True, which='both', ls='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
