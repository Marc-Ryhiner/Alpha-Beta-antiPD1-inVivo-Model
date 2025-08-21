import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# --- Load Lea-Catcheside factors ---
lea_interpolators = {}


def load_lea_csv(activity):
    df = pd.read_csv(f"Lea-Catcheside{activity}.csv")
    times = df["time"].values
    values = df[activity].values
    f = PchipInterpolator(times, np.clip(values, 0, 1), extrapolate=True)
    return lambda t: np.clip(f(t), 0.0, 1.0)


lea_interpolators["0.0"] = load_lea_csv("0")
lea_interpolators["0.037"] = load_lea_csv("0.037")
lea_interpolators["0.1"] = load_lea_csv("0.1")
lea_interpolators["0.185"] = load_lea_csv("0.185")
lea_interpolators["0.25"] = load_lea_csv("0.25")
lea_interpolators["0.37"] = load_lea_csv("0.37")
lea_interpolators["0.5"] = load_lea_csv("0.5")
lea_interpolators["0.75"] = load_lea_csv("0.75")
lea_interpolators["1.25"] = load_lea_csv("1.25")
lea_interpolators["1.85"] = load_lea_csv("1.85")


# Robust getter for Lea–Catcheside: choose nearest available activity curve (A0 is in Bq, keys are in kBq)
_lea_keys = np.array([float(k) for k in lea_interpolators.keys()])
def get_lea_interp(A0):
    target = A0 / 1000.0  # convert to kBq to match keys
    idx = int(np.argmin(np.abs(_lea_keys - target)))
    key = str(_lea_keys[idx])
    # ensure exact string key that exists in dict (handles '0.0' vs '0.0...')
    for k in lea_interpolators.keys():
        if float(k) == float(key):
            return lea_interpolators[k]
    # fallback (shouldn't hit)
    return lea_interpolators["0.0"]


mu225ac = 2.91e-3  # h⁻¹
mu221fr = 8.64
mu217at = 7.60e4
mu213bi = 0.911
mu213po = 6.73e5
mu209tl = 3.57
mu209pb = 0.215

pr = {
    'Fr221_g': 0.114,
    'Bi213_a': 0.0214,
    'Bi213_b': 0.9786,
    'Bi213_g': 0.259,
}

# Fixed parameters (excluding those to be fitted)
base_params = {
    'k_on': 131.4, 'k_off': 1.611, 'k_int': 0.7602, 'k_rel': 2.138,
    'Ncell': 1e5, 'R': 0.000814
}

S_values = {'S_i_225Ac': 2.14e-6, 'S_i_221Fr': 2.36e-6, 'S_i_221Frg': 8.01e-8, 'S_i_217At': 2.62e-6, 'S_i_213Bia': 2.17e-6,
            'S_i_213Bib': 1.60e-7, 'S_i_213Big': 1.63e-8, 'S_i_213Po': 3.09e-6, 'S_i_209Tl': 2.38e-7, 'S_i_209Pb': 7.28e-8,
            'S_m_225Ac': 65.5, 'S_m_221Fr': 60.1, 'S_m_221Frg': 0.0588, 'S_m_217At': 55.1, 'S_m_213Bia': 64.4,
            'S_m_213Bib': 0.152, 'S_m_213Big': 0.151, 'S_m_213Po': 119, 'S_m_209Tl': 0.138, 'S_m_209Pb': 0.210,
            'S_c_225Ac': 118, 'S_c_221Fr': 109, 'S_c_221Frg': 0.371, 'S_c_217At': 101, 'S_c_213Bia': 117,
            'S_c_213Bib': 0.282, 'S_c_213Big': 0.281, 'S_c_213Po': 88.2, 'S_c_209Tl': 0.256, 'S_c_209Pb': 0.210}

# Known parameters to be filled in
px_f = 0.55
px_s = 0.45
lambd_f = 0.990
lambd_s = 0.0578
alpha_low = 0.177

# Experimental dataset
data = np.array([
    [0, 0, 1, 0],
    [0.000164, 37, 0.93, 0],
    [0.000444, 100, 0.808, 0],
    [0.000822, 185, 0.615, 0.043],
    [0.00111, 250, 0.626, 0],
    [0.00164, 370, 0.494, 0.048],
    [0.00222, 500, 0.43, 0.063],
    [0.00333, 750, 0.135, 0.038],
    [0.00556, 1250, 0.00246, 0.00177],
    [0.00822, 1850, 0.000994, 0.000486]
])

def ode_system(t, y, ci0, A0, alpha, beta, px_f, px_s, lambd_f, lambd_s, alpha_low, p):
    eps = 1e-12

    i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb, Ac, Fr, Frg, At, Bia, Bib, Big, Po, Tl, Pb, D_high, D_low = y

    kon, koff, kint, krel = p['k_on'], p['k_off'], p['k_int'], p['k_rel']
    N, R = p['Ncell'], p['R']

    di225ac = koff*m225ac - (kon*(R - m225ac - m_u) + mu225ac)*i225ac
    dm225ac = kon*(R - m225ac - m_u)*i225ac + krel*c225ac - (koff + kint + mu225ac)*m225ac
    dc225ac = kint*m225ac - (krel + mu225ac)*c225ac

    diu = koff*m_u + mu225ac*i225ac - kon*(R - m225ac - m_u)*iu
    dm_u = kon*(R - m225ac - m_u)*iu + krel*cu + mu225ac*m225ac - (koff + kint)*m_u
    dcu = kint*m_u + mu225ac*c225ac - krel*cu

    di221fr = mu225ac*(i225ac+m225ac)-mu221fr*i221fr
    dc221fr = mu225ac*c225ac-mu221fr*c221fr

    di217at = mu221fr*i221fr-mu217at*i217at
    dc217at = mu221fr*c221fr-mu217at*c217at

    di213bi = mu217at*i217at-mu213bi*i213bi
    dc213bi = mu217at*c217at-mu213bi*c213bi

    di209tl = pr['Bi213_a']*mu213bi*i213bi-mu209tl*i209tl
    dc209tl = pr['Bi213_a']*mu213bi*c213bi-mu209tl*c209tl

    di213po = pr['Bi213_b']*mu213bi*i213bi-mu213po*i213po
    dc213po = pr['Bi213_b']*mu213bi*c213bi-mu213po*c213po

    di209pb = mu209tl*i209tl+mu213po*i213po-mu209pb*i209pb
    dc209pb = mu209tl*c209tl+mu213po*c213po-mu209pb*c209pb

    dAc = -mu225ac*Ac
    dFr = mu221fr*(Ac - Fr)
    dFrg = pr['Fr221_g'] * dFr
    dAt = mu217at*(Fr - At)
    dBia = mu213bi*(pr['Bi213_a']*At - Bia)
    dBib = mu213bi*(pr['Bi213_b']*At - Bib)
    dBig = pr['Bi213_g']*mu213bi*(At - Big)
    dPo = mu213po*(Bib - Po)
    dTl = mu209tl*(Bia - Tl)
    dPb = mu209pb*(Po + Tl - Pb)

    dD_high = Ac * (S_values['S_i_225Ac'] + (S_values['S_m_225Ac'] * m225ac + S_values['S_c_225Ac'] * c225ac) / (N * (i225ac + m225ac + c225ac) + eps)) + \
              Fr * (S_values['S_i_221Fr'] + S_values['S_c_221Fr'] * c221fr / (N * (i221fr + c221fr) + eps)) + \
              At * (S_values['S_i_217At'] + S_values['S_c_217At'] * c217at / (N * (i217at + c217at) + eps)) + \
              Bia * (S_values['S_i_213Bia'] + S_values['S_c_213Bia'] * c213bi / (N * (i213bi + c213bi) + eps)) + \
              Po * (S_values['S_i_213Po'] + S_values['S_c_213Po'] * c213po / (N * (i213po + c213po) + eps))

    dD_low = Frg * (S_values['S_i_221Frg'] + S_values['S_c_221Frg'] * c221fr / (N * (i221fr + c221fr) + eps)) + \
             Bib * (S_values['S_i_213Bib'] + S_values['S_c_213Bib'] * c213bi / (N * (i213bi + c213bi) + eps)) + \
             Big * (S_values['S_i_213Big'] + S_values['S_c_213Big'] * c213bi / (N * (i213bi + c213bi) + eps)) + \
             Tl * (S_values['S_i_209Tl'] + S_values['S_c_209Tl'] * c209tl / (N * (i209tl + c209tl) + eps)) + \
             Pb * (S_values['S_i_209Pb'] + S_values['S_c_209Pb'] * c209pb / (N * (i209pb + c209pb) + eps))

#    # Use nearest Lea–Catcheside curve for the provided A0
#    lea = get_lea_interp(A0)
#
#    def derivative_interp(interp_func, t, D, h=1e-3):
#        return (interp_func(t + h) - interp_func(t - h)) * D ** 2 / (2 * h)
#
#    G_num = lea(t) * (D_high ** 2)
#    dG_num = derivative_interp(lea, t, D_high)
#
#    sum_term = alpha * dD_high + beta * dG_num + alpha_low * dD_low
#    exp_term = - (alpha * D_high + beta * G_num + alpha_low * D_low)
#    dS = -sum_term * np.exp(exp_term)
#
#    if A0 == 2000:
#        print(alpha * dD_high + beta * dG_num + alpha_low * dD_low)

    return [di225ac, dm225ac, dc225ac, diu, dm_u, dcu, di221fr, dc221fr, di217at, dc217at, di213bi, dc213bi, di209tl, dc209tl, di213po, dc213po, di209pb, dc209pb,
            dAc, dFr, dFrg, dAt, dBia, dBib, dBig, dPo, dTl, dPb,
            dD_high, dD_low]


def ode_system_2(t, y, ci0, A0, alpha, beta, px_f, px_s, lambd_f, lambd_s, alpha_low, p):
    eps = 1e-12

    i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb, Ac, Fr, Frg, At, Bia, Bib, Big, Po, Tl, Pb, D_high, D_low = y

    kon, koff, kint, krel = p['k_on'], p['k_off'], p['k_int'], p['k_rel']
    N, R = p['Ncell'], p['R']

    di225ac = koff * m225ac - (kon * (R * np.exp(t * 0.0277) - m225ac - m_u) + mu225ac) * i225ac
    dm225ac = kon * (R * np.exp(t * 0.0277) - m225ac - m_u) * i225ac + krel * c225ac - (koff + kint + mu225ac) * m225ac
    dc225ac = kint * m225ac - (krel + mu225ac) * c225ac

    diu = koff * m_u + mu225ac * i225ac - kon * (R * np.exp(t * 0.0277) - m225ac - m_u) * iu
    dm_u = kon * (R * np.exp(t * 0.0277) - m225ac - m_u) * iu + krel * cu + mu225ac * m225ac - (koff + kint) * m_u
    dcu = kint * m_u + mu225ac * c225ac - krel * cu

    di221fr = mu225ac * (i225ac + m225ac) - mu221fr * i221fr
    dc221fr = mu225ac * c225ac - mu221fr * c221fr

    di217at = mu221fr * i221fr - mu217at * i217at
    dc217at = mu221fr * c221fr - mu217at * c217at

    di213bi = mu217at * i217at - mu213bi * i213bi
    dc213bi = mu217at * c217at - mu213bi * c213bi

    di209tl = pr['Bi213_a'] * mu213bi * i213bi - mu209tl * i209tl
    dc209tl = pr['Bi213_a'] * mu213bi * c213bi - mu209tl * c209tl

    di213po = pr['Bi213_b'] * mu213bi * i213bi - mu213po * i213po
    dc213po = pr['Bi213_b'] * mu213bi * c213bi - mu213po * c213po

    di209pb = mu209tl * i209tl + mu213po * i213po - mu209pb * i209pb
    dc209pb = mu209tl * c209tl + mu213po * c213po - mu209pb * c209pb

    dAc = -mu225ac*Ac
    dFr = mu221fr*(Ac - Fr)
    dFrg = pr['Fr221_g'] * dFr
    dAt = mu217at*(Fr - At)
    dBia = mu213bi*(pr['Bi213_a']*At - Bia)
    dBib = mu213bi*(pr['Bi213_b']*At - Bib)
    dBig = pr['Bi213_g']*mu213bi*(At - Big)
    dPo = mu213po*(Bib - Po)
    dTl = mu209tl*(Bia - Tl)
    dPb = mu209pb*(Po + Tl - Pb)

    dD_high = Ac * ((S_values['S_m_225Ac'] * (i225ac + m225ac) + S_values['S_c_225Ac'] * c225ac) / (N * np.exp(t * 0.0277) * (i225ac + m225ac + c225ac) + eps)) + \
              Fr * ((S_values['S_m_221Fr'] * i221fr + S_values['S_c_221Fr'] * c221fr) / (N * np.exp(t * 0.0277) * (i221fr + c221fr) + eps)) + \
              At * ((S_values['S_m_217At'] * i217at + S_values['S_c_217At'] * c217at) / (N * np.exp(t * 0.0277) * (i217at + c217at) + eps)) + \
              Bia * ((S_values['S_m_213Bia'] * i213bi + S_values['S_c_213Bia'] * c213bi) / (N * np.exp(t * 0.0277) * (i213bi + c213bi) + eps)) + \
              Po * ((S_values['S_m_213Po'] * i213po + S_values['S_c_213Po'] * c213po) / (N * np.exp(t * 0.0277) * (i213po + c213po) + eps))

    dD_low = Frg * ((S_values['S_m_221Frg'] * i221fr + S_values['S_c_221Frg'] * c221fr) / (N * np.exp(t * 0.0277) * (i221fr + c221fr) + eps)) + \
             Bib * ((S_values['S_m_213Bib'] * i213bi + S_values['S_c_213Bib'] * c213bi) / (N * np.exp(t * 0.0277) * (i213bi + c213bi) + eps)) + \
             Big * ((S_values['S_m_213Big'] * i213bi + S_values['S_c_213Big'] * c213bi) / (N * np.exp(t * 0.0277) * (i213bi + c213bi) + eps)) + \
             Tl * ((S_values['S_m_209Tl'] * i209tl + S_values['S_c_209Tl'] * c209tl) / (N * np.exp(t * 0.0277) * (i209tl + c209tl) + eps)) + \
             Pb * ((S_values['S_m_209Pb'] * i209pb + S_values['S_c_209Pb'] * c209pb) / (N * np.exp(t * 0.0277) * (i209pb + c209pb) + eps))


#    # Use nearest Lea–Catcheside curve for the provided A0
#    lea = get_lea_interp(A0)
#
#    def derivative_interp(interp_func, t, D, h=1e-3):
#        return (interp_func(t + h) - interp_func(t - h)) * D ** 2 / (2 * h)
#
#
#    G_num = lea(t + 3) * (D_high ** 2)
#    dG_num = derivative_interp(lea, t + 3, D_high)
#
#    sum_term = alpha * dD_high + beta * dG_num + alpha_low * dD_low
#    exp_term = - (alpha * D_high + beta * G_num + alpha_low * D_low)
#    dS = -sum_term * np.exp(exp_term)
#
#    if A0 == 2000:
#        print(alpha * dD_high + beta * dG_num + alpha_low * dD_low)

    return [di225ac, dm225ac, dc225ac, diu, dm_u, dcu, di221fr, dc221fr, di217at, dc217at, di213bi, dc213bi, di209tl, dc209tl, di213po, dc213po, di209pb, dc209pb,
            dAc, dFr, dFrg, dAt, dBia, dBib, dBig, dPo, dTl, dPb,
            dD_high, dD_low]


# Simulation wrapper
def simulate_survival(ci0, A0, alpha, beta):
    y0 = [ci0 * 0.9985 * 0.2652946819207342, 0, 0,
          ci0 * 0.001454 * 0.2652946819207342, 0, 0,
          ci0 * 3.320e-4 * 0.2652946819207342, 0,
          ci0 * 3.774e-8 * 0.2652946819207342, 0,
          ci0 * 9.343e-4 * 0.2652946819207342, 0,
          ci0 * 2.530e-6 * 0.2652946819207342, 0,
          ci0 * 1.238e-9 * 0.2652946819207342, 0,
          ci0 * 1.794e-4 * 0.2652946819207342, 0,
          A0 * 0.9985 * 0.2652946819207342,
          A0 * 0.9857 * 0.2652946819207342,
          A0 * 0.1124 * 0.2652946819207342,
          A0 * 0.9857 * 0.2652946819207342,
          A0 * 6.259e-3 * 0.2652946819207342,
          A0 * 0.2862 * 0.2652946819207342,
          A0 * 0.08664 * 0.2652946819207342,
          A0 * 0.2862 * 0.2652946819207342,
          A0 * 3.104e-3 * 0.2652946819207342,
          A0 * 0.01325 * 0.2652946819207342,
          0, 0]


    # Solve phase 1
    sol1 = solve_ivp(
        ode_system, [0, 3], y0,
        args=(ci0, A0, alpha, beta, px_f, px_s, lambd_f, lambd_s, alpha_low, base_params),
        dense_output=True, method='BDF', rtol=1e-6, atol=1e-8
    )

    i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb, Ac, Fr, Frg, At, Bia, Bib, Big, Po, Tl, Pb, D_high, D_low = sol1.y[:, -1]

    # Phase 2
    i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb = [sol1.y[0, -1], sol1.y[1, -1], sol1.y[2, -1], sol1.y[3, -1], sol1.y[4, -1], sol1.y[5, -1], sol1.y[6, -1], sol1.y[7, -1], sol1.y[8, -1], sol1.y[9, -1], sol1.y[10, -1], sol1.y[11, -1], sol1.y[12, -1], sol1.y[13, -1], sol1.y[14, -1], sol1.y[15, -1], sol1.y[16, -1], sol1.y[17, -1]]
    eps = 1e-12
    Ac2 = Ac * m225ac / (i225ac + m225ac + c225ac + eps) + Ac * c225ac / (i225ac + m225ac + c225ac + eps)
    Fr2 = Fr * c221fr / (i221fr + c221fr + eps)
    Frg2 = Frg * c221fr / (i221fr + c221fr + eps)
    At2 = At * c217at / (i217at + c217at + eps)
    Bia2 = Bia * c213bi / (i213bi + c213bi + eps)
    Bib2 = Bib * c213bi / (i213bi + c213bi + eps)
    Big2 = Big * c213bi / (i213bi + c213bi + eps)
    Po2 = Po * c213po / (i213po + c213po + eps)
    Tl2 = Tl * c209tl / (i209tl + c209tl + eps)
    Pb2 = Pb * c209pb / (i209pb + c209pb + eps)
    i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb = [0, sol1.y[1, -1], sol1.y[2, -1], 0, sol1.y[4, -1], sol1.y[5, -1], 0, sol1.y[7, -1], 0, sol1.y[9, -1], 0, sol1.y[11, -1], 0, sol1.y[13, -1], 0, sol1.y[15, -1], 0, sol1.y[17, -1]]
    y0_phase2 = [i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb, Ac2, Fr2, Frg2, At2, Bia2, Bib2, Big2, Po2, Tl2, Pb2, D_high, D_low]

    sol2 = solve_ivp(
        ode_system_2, [0, 168], y0_phase2,
        args=(0, A0, alpha, beta, px_f, px_s, lambd_f, lambd_s, alpha_low, base_params),
        t_eval=[168], method='BDF', rtol=1e-6, atol=1e-8
    )


    # Use nearest Lea–Catcheside curve for the provided A0
    lea = get_lea_interp(A0)


    G = lea(171)
    dose_high = sol2.y[-2, -1]
    dose_low = sol2.y[-1, -1]

    survival = np.exp(- alpha * dose_high - beta * dose_high ** 2 * G - alpha_low * dose_low)

    if A0 == 2000:
        print('Clinical Effect linear: ' + str(dose_high * alpha))
        print('Clinical Effect quadratic: ' + str(dose_high * beta * G))
        print('Clinical Effect low: ' + str(dose_low * alpha_low))

    return survival
    #return float(sol2.y[-1, -1])


# Objective function
def objective(theta):
    eps = 1e-12
    alpha, beta = theta
    chi2 = 0
    for ci0, A0, S_obs, sigma in data:
        S_model = simulate_survival(ci0, A0, alpha, beta)
        log_model = np.log(S_model + eps)
        log_obs = np.log(S_obs + eps)
        log_sigma = np.log(sigma + eps)
        chi2 += ((log_model - log_obs) / log_sigma) ** 2
    return chi2

# Initial guesses and bounds for alpha and beta
x0 = [1, 0]
bounds = [
    (0, np.inf), (0, 0)
]

# Optimization
res = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B', options={'disp': True})
print("Fitted parameters:")
print(f"alpha = {res.x[0]:.4f}, beta = {res.x[1]:.4f}")


# -------------------------------
# Plot: experimental vs model curve (0..2 kBq)
# -------------------------------

# Proportional relationship ci0 = k * A0 (estimate k from the dataset, ignoring zeros)
mask = data[:,1] > 0
k_prop = np.mean(data[mask,0] / data[mask,1])  # ~4.44e-06 from your table
print(f"Estimated proportionality ci0 = k*A0, k = {k_prop:.6e}")

# Build a dense activity grid from 0 to 2000 (i.e., 0..2 kBq)
A_grid = np.linspace(0, 2000, 101)
ci_grid = k_prop * A_grid

alpha_fit, beta_fit = res.x
S_model_grid = []
for ci0, A0 in zip(ci_grid, A_grid):
    S_model_grid.append(simulate_survival(ci0, A0, alpha_fit, beta_fit))
S_model_grid = np.array(S_model_grid)


# Plot
plt.figure(figsize=(8,5))
# Experimental data with ±sigma error bars
plt.errorbar(data[:,1], data[:,2], yerr=data[:,3], label='Survival data',
             fmt='o', capsize=5, color='#A0522D')

# Model curve
plt.plot(A_grid, S_model_grid, '-', label='Survival fit', color='#D2B48C')

plt.xlim(0, 2000)
plt.ylim(0.0001, 1)
plt.yscale('log')
plt.xlabel('Initial activity (kBq)')
plt.ylabel('Surviving fraction')
plt.title('Linear Effect Fit to Cell Survival during Ac-225 Treatment')
plt.grid(True, which='both', alpha=0.3)
plt.legend(ncol=2)
plt.tight_layout()
plt.show()

