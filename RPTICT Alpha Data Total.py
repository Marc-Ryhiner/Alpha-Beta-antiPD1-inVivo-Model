
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Ensure all columns are the same length by padding missing entries with NaN
days = [11, 17, 25, 28, 32, 35, 39, 42, 46, 49, 53, 56, 59, 62, 66, 73, 80, 87, 94, 101, 108, 115, 122, 129, 136, 143, 150, 157, 164, 171, 185, 192]

control = [0.141, 0.496, 1.975, 2.383, 2.452, 2.597, 3.842] + [np.nan]*(len(days)-7)
aPD1    = [0.104, 0.325, 0.743, 1.199, 1.824, 2.209, 2.251, 1.613, 0.438, 0.621, 0.987, 1.278, 1.797, 2.572] + [np.nan]*(len(days)-14)
RLT     = [0.084, 0.284, 1.004, 1.504, 1.765, 1.066, 1.435, 1.795, 0.099, 0.086, 0.112, 0.145, 0.186, 0.218, 0.211] + [np.nan]*(len(days)-15)
RLT_aPD1= [0.065, 0.137, 0.156, 0.205, 0.375, 0.637, 1.170, 1.205, 1.030, 0.456, 0.766, 1.100, 0.745, 0.960, 0.008, 0.009, 0.010, 0.011, 0.013, 0.016, 0.026, 0.029, 0.028, 0.036, 0.040, 0.051, 0.091, 0.117, 0.188, 0.442, 0.000, 0.000]

df = pd.DataFrame({
    "days": days,
    "control": control,
    "aPD1": aPD1,
    "RLT": RLT,
    "RLT_aPD1": RLT_aPD1
})

# Plotting
plt.figure(figsize=(8,5))
plt.plot(df["days"], df["control"], marker="o", label="Control", color='gray')
plt.plot(df["days"], df["aPD1"], marker="s", label="αPD-1", color='blue')
plt.plot(df["days"], df["RLT"], marker="^", label="Ac-225", color='orange')
plt.plot(df["days"], df["RLT_aPD1"], marker="d", label="Ac-225/αPD-1", color='green')

plt.xlabel("Days post inoculation")
plt.ylabel("Mean tumor size (mL)")
plt.title("Tumor Growth for Actinium Combination Treatment")
plt.legend()
plt.grid(True, which='both', ls='--', linewidth=0.5)
plt.tight_layout()
plt.show()
