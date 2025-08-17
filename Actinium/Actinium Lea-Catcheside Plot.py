import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

df = pd.read_csv(f"Lea-Catcheside2.csv")

x_axis = df['time']
y_axis_2 = df['2']


lea_interpolator = interp1d(x_axis, y_axis_2, kind="cubic", bounds_error=False, fill_value="extrapolate")


def derivative_interp(interp_func, t, h=1e-3):
    return (interp_func(t + h) - interp_func(t - h)) / (2 * h)


y_axis_1 = []

for t in x_axis:
    y_axis_1.append(derivative_interp(lea_interpolator, t))


# Plot
fig, ax1 = plt.subplots(figsize=(8, 5))

color1 = 'tab:purple'
ax1.set_xlabel('Time after experiment start (h)')
ax1.set_ylabel('Lea-Catcheside factor rate (per h)', color=color1)
ax1.plot(x_axis, np.abs(y_axis_1), color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(0.1, 200)
ax1.set_ylim(2e-5, 0.2)
ax1.yaxis.grid(True, which='major', ls='--', linewidth=0.5, color='#b5a8d2')
ax1.xaxis.grid(True, which='both', ls='--', linewidth=0.5)
ax1.axvline(3, color='gray', linestyle='--', label='Cell Plating')
ax1.legend()

ax2 = ax1.twinx()
color2 = '#ffff00'
ax2.set_ylabel('Lea-Catcheside factor', color=color2)
ax2.plot(x_axis, y_axis_2, color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_yscale('log')
ax2.set_ylim(0.08, 1)
ax2.yaxis.grid(True, which='major', ls='--', linewidth=0.5, color='#d2c85a')


plt.title('Lea-Catcheside Factor Rate and Total during Ac-225 Treatment $A_0=2 kBq$')
fig.tight_layout()
#fig.legend()
plt.show()