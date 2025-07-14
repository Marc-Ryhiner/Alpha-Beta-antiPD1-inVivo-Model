import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# Experimental time points (hours)
t_data = np.array([0, 1, 2, 3])
# Converted % → fractions → nmol
cb_data = np.array([0.0, 18.5, 21.3, 21.7]) / 100 * 0.0015
cc_data = np.array([0.0, 4.86, 7.08, 7.42]) / 100 * 0.0015

# Measurement uncertainties (fraction)
cb_err = np.array([1e-10, 3.45, 3.98, 4.81]) / 100 * 0.0015
cc_err = np.array([1e-10, 0.48, 0.383, 0.975]) / 100 * 0.0015

R = 0.005  # nmol/mL

def ode_system(t, y, kon, koff, kint, krel):
    cm, cb, cc = y
    dcm = koff * cb - kon * (R - cb) * cm
    dcb = kon * (R - cb) * cm + krel * cc - (koff + kint) * cb
    dcc = kint * cb - krel * cc
    return [dcm, dcb, dcc]

def model(t, kon, koff, kint, krel):
    y0 = [0.0015, 0.0, 0.0]
    sol = solve_ivp(
        ode_system, [t.min(), t.max()], y0,
        args=(kon, koff, kint, krel),
        t_eval=t, method='BDF'
    )
    cb, cc = sol.y[1], sol.y[2]
    return np.concatenate([cb, cc])

# Combine data and uncertainties for fitting
y_data = np.concatenate([cb_data, cc_data])
y_err = np.concatenate([cb_err, cc_err])

# Bounds and initial guesses
#bounds_lower = [
#    7.7e-3,    # k_on min (mL/nmol/h)
#    7.7e-6,    # k_off min (/h)
#    1.67e-7,   # k_int min (/h)
#    2.67e-8    # k_rel min (/h)
#]
bounds_lower = [
    0,    # k_on min (mL/nmol/h)
    0,    # k_off min (/h)
    0,   # k_int min (/h)
    0    # k_rel min (/h)
]
#bounds_upper = [
#    77,       # k_on max (mL/nmol/h)
#    0.077,    # k_off max (/h)
#    1.67e-3,  # k_int max (/h)
#    2.67e-4   # k_rel min (/h)
#]
bounds_upper = [
    np.inf,   # k_on max (mL/nmol/h)
    np.inf,   # k_off max (/h)
    np.inf,   # k_int max (/h)
    np.inf    # k_rel min (/h)
]
#p0 = [0.77, 7.7e-4, 1.67e-5, 2.67e-6]
p0 = [0.1, 1.3, 1, 3]

popt, pcov = curve_fit(
    model,
    t_data,
    y_data,
    sigma=y_err,
    absolute_sigma=True,
    p0=p0,
    bounds=(bounds_lower, bounds_upper),
    method='trf'
)
kon_fit, koff_fit, kint_fit, krel_fit = popt
perr = np.sqrt(np.diag(pcov))
#popt = [0.2, 2.9, 1, 3]

print("Fitted parameters:")
print(f"kon  = {kon_fit:.3e} ± {perr[0]:.1e}  mL·nmol⁻¹·h⁻¹")
print(f"koff = {koff_fit:.3e} ± {perr[1]:.1e}  h⁻¹")
print(f"kint = {kint_fit:.3e} ± {perr[2]:.1e}  h⁻¹")
print(f"krel = {krel_fit:.3e} ± {perr[3]:.1e}  h⁻¹")

# Generate smooth fits over the time range
t_fit = np.linspace(0, 3.2, 100)
sol = solve_ivp(
    ode_system, [0, 3.2], [0.0015, 0, 0],
    args=tuple(popt), t_eval=t_fit
)
cb_fit, cc_fit = sol.y[1], sol.y[2]

# Plotting
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(t_data, cb_data * 1000, yerr=cb_err * 1000, fmt='o', label='Bound data', capsize=5, color='#0072B2')
ax.errorbar(t_data, cc_data * 1000, yerr=cc_err * 1000, fmt='s', label='Cytoplasm data', capsize=5, color='#D55E00')
ax.plot(t_fit, cb_fit * 1000, '-', label='Bound fit', color='#009E73')
ax.plot(t_fit, cc_fit * 1000, '--', label='Cytoplasm fit', color='#F0E442')
ax.grid(True, linewidth=0.5)
ax.set_xlabel('Time (h)')
ax.set_ylabel('Concentration (pM)')
ax.legend(ncol=2)
ax.set_xlim([0, 3.2])
ax.set_ylim([0, 0.4])
ax.set_title('Cellular Kinetic Parameter Fit to Compartment Concentrations')
plt.show()