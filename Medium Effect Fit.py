
import numpy as np
import matplotlib.pyplot as plt

# Given constants
lambda_Lu = 4.323e-3  # h⁻¹
A0 = 0.4              # MBq
alpha = 0.16          # Gy⁻¹
t_cal = 3.0           # calibration time in hours
S_cal = 0.94          # observed survival at t_cal
S_err = 0.033         # observed survival error at t_cal

# Analytical formula for Si
Si = - (lambda_Lu) / (alpha * A0) * np.log(S_cal) / (1 - np.exp(-lambda_Lu * t_cal))
print(f"Calibrated S_i = {Si:.6f} MBq⁻¹·Gy·h⁻¹")

# Time axis and predicted S(t)
t = np.linspace(0, 3.2, 300)
D = (Si * A0 / lambda_Lu) * (1 - np.exp(-lambda_Lu * t))
S = np.exp(-alpha * D)

# Plot
plt.figure(figsize=(8, 5))
plt.errorbar([t_cal], [S_cal], [S_err], fmt='^', label='Survival data', capsize=5, color='#8B5E3C')
plt.plot(t, S, label='Survival fit', color='#C4A484')
plt.title('Medium Effect Fit to Cell Survival')
plt.xlabel('Time (h)')
plt.ylabel('Surviving Fraction')
plt.xlim([0, 3.2])
plt.ylim([0.9, 1])
plt.legend(ncol=2)
plt.grid(True)
plt.show()
