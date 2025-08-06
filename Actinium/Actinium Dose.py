import numpy as np
from scipy.integrate import solve_ivp, cumtrapz
import matplotlib.pyplot as plt


S_values = {'S_i_225Ac': 2.14e-6, 'S_i_221Fr': 2.36e-6, 'S_i_221Frg': 8.01e-8, 'S_i_217At': 2.62e-6, 'S_i_213Bia': 2.17e-6,
            'S_i_213Bib': 1.60e-7, 'S_i_213Big': 1.63e-8, 'S_i_213Po': 3.09e-6, 'S_i_209Tl': 2.38e-7, 'S_i_209Pb': 7.28e-8,
            'S_m_225Ac': 65.5, 'S_m_221Fr': 60.1, 'S_m_221Frg': 0.0588, 'S_m_217At': 55.1, 'S_m_213Bia': 64.4,
            'S_m_213Bib': 0.152, 'S_m_213Big': 0.151, 'S_m_213Po': 119, 'S_m_209Tl': 0.138, 'S_m_209Pb': 0.210,
            'S_c_225Ac': 118, 'S_c_221Fr': 109, 'S_c_221Frg': 0.371, 'S_c_217At': 101, 'S_c_213Bia': 117,
            'S_c_213Bib': 0.282, 'S_c_213Big': 0.281, 'S_c_213Po': 88.2, 'S_c_209Tl': 0.256, 'S_c_209Pb': 0.210}

# --- Constants ---
λ = {
    'Ac225': 2.91e-3,  # h⁻¹
    'Fr221': 8.64,
    'At217': 7.60e4,
    'Bi213': 0.911,
    'Po213': 6.73e5,
    'Tl209': 3.57,
    'Pb209': 0.215,
}
p = {
    'Fr221_g': 0.114,
    'Bi213_a': 0.0214,
    'Bi213_b': 0.9786,
    'Bi213_g': 0.259,
}
# Kinetics
kon = 131.4
koff = 1.611
kint = 0.7602
krel = 2.138
mu225ac = λ['Ac225']
mu221fr = λ['Fr221']
mu217at = λ['At217']
mu213bi = λ['Bi213']
mu209tl = λ['Tl209']
mu213po = λ['Po213']
mu209pb = λ['Pb209']

# Receptor capacity
Rtot = 0.000814
N_cell = 100000

alpha = 1  # Gy^-1 (placeholder)


def deriv_0(t, y):
    # compartments for 225Ac: md, bd, cd, mu, bu, cu, ms, cs
    i225ac, i221fr, i217at, i213bi, i209tl, i213po, i209pb, = y

    di225ac = -mu225ac*i225ac
    di221fr = mu225ac*i225ac-mu221fr*i221fr
    di217at = mu221fr*i221fr-mu217at*i217at
    di213bi = mu217at*i217at-mu213bi*i213bi
    di209tl = p['Bi213_a']*mu213bi*i213bi-mu209tl*i209tl
    di213po = p['Bi213_b']*mu213bi*i213bi-mu213po*i213po
    di209pb = mu209tl*i209tl+mu213po*i213po-mu209pb*i209pb

    return [di225ac, di221fr, di217at, di213bi, di209tl, di213po, di209pb]


# ODE system
def deriv(t, y):
    i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb, = y

    di225ac = koff*m225ac - (kon*(Rtot - m225ac - m_u) + mu225ac)*i225ac
    dm225ac = kon*(Rtot - m225ac - m_u)*i225ac + krel*c225ac - (koff + kint + mu225ac)*m225ac
    dc225ac = kint*m225ac - (krel + mu225ac)*c225ac

    diu = koff*m_u + mu225ac*i225ac - kon*(Rtot - m225ac - m_u)*iu
    dm_u = kon*(Rtot - m225ac - m_u)*iu + krel*cu + mu225ac*m225ac - (koff + kint)*m_u
    dcu = kint*m_u + mu225ac*c225ac - krel*cu

    di221fr = mu225ac*(i225ac+m225ac)-mu221fr*i221fr
    dc221fr = mu225ac*c225ac-mu221fr*c221fr

    di217at = mu221fr*i221fr-mu217at*i217at
    dc217at = mu221fr*c221fr-mu217at*c217at

    di213bi = mu217at*i217at-mu213bi*i213bi
    dc213bi = mu217at*c217at-mu213bi*c213bi

    di209tl = p['Bi213_a']*mu213bi*i213bi-mu209tl*i209tl
    dc209tl = p['Bi213_a']*mu213bi*c213bi-mu209tl*c209tl

    di213po = p['Bi213_b']*mu213bi*i213bi-mu213po*i213po
    dc213po = p['Bi213_b']*mu213bi*c213bi-mu213po*c213po

    di209pb = mu209tl*i209tl+mu213po*i213po-mu209pb*i209pb
    dc209pb = mu209tl*c209tl+mu213po*c213po-mu209pb*c209pb

    return [di225ac, dm225ac, dc225ac, diu, dm_u, dcu, di221fr, dc221fr, di217at, dc217at, di213bi, dc213bi, di209tl, dc209tl, di213po, dc213po, di209pb, dc209pb]


# ODE system
def deriv_2(t, y):
    i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb, = y

    di225ac = koff*m225ac - (kon*(Rtot * np.exp(t * 0.0277) - m225ac - m_u) + mu225ac)*i225ac
    dm225ac = kon*(Rtot * np.exp(t * 0.0277) - m225ac - m_u)*i225ac + krel*c225ac - (koff + kint + mu225ac)*m225ac
    dc225ac = kint*m225ac - (krel + mu225ac)*c225ac

    diu = koff*m_u + mu225ac*i225ac - kon*(Rtot * np.exp(t * 0.0277) - m225ac - m_u)*iu
    dm_u = kon*(Rtot * np.exp(t * 0.0277) - m225ac - m_u)*iu + krel*cu + mu225ac*m225ac - (koff + kint)*m_u
    dcu = kint*m_u + mu225ac*c225ac - krel*cu

    di221fr = mu225ac*(i225ac+m225ac)-mu221fr*i221fr
    dc221fr = mu225ac*c225ac-mu221fr*c221fr

    di217at = mu221fr*i221fr-mu217at*i217at
    dc217at = mu221fr*c221fr-mu217at*c217at

    di213bi = mu217at*i217at-mu213bi*i213bi
    dc213bi = mu217at*c217at-mu213bi*c213bi

    di209tl = p['Bi213_a']*mu213bi*i213bi-mu209tl*i209tl
    dc209tl = p['Bi213_a']*mu213bi*c213bi-mu209tl*c209tl

    di213po = p['Bi213_b']*mu213bi*i213bi-mu213po*i213po
    dc213po = p['Bi213_b']*mu213bi*c213bi-mu213po*c213po

    di209pb = mu209tl*i209tl+mu213po*i213po-mu209pb*i209pb
    dc209pb = mu209tl*c209tl+mu213po*c213po-mu209pb*c209pb

    return [di225ac, dm225ac, dc225ac, diu, dm_u, dcu, di221fr, dc221fr, di217at, dc217at, di213bi, dc213bi, di209tl, dc209tl, di213po, dc213po, di209pb, dc209pb]


# Activity decay chain (same as original)
def activity_chain(t, A):
    Ac, Fr, Frg, At, Bia, Bib, Big, Po, Tl, Pb = A
    dAc = -λ['Ac225']*Ac
    dFr = λ['Fr221']*(Ac - Fr)
    dFrg = p['Fr221_g'] * dFr
    dAt = λ['At217']*(Fr - At)
    dBia = λ['Bi213']*(p['Bi213_a']*At - Bia)
    dBib = λ['Bi213']*(p['Bi213_b']*At - Bib)
    dBig = p['Bi213_g']*λ['Bi213']*(At - Big)
    dPo = λ['Po213']*(Bib - Po)
    dTl = λ['Tl209']*(Bia - Tl)
    dPb = λ['Pb209']*(Po + Tl - Pb)
    return [dAc, dFr, dFrg, dAt, dBia, dBib, dBig, dPo, dTl, dPb]


# Combined ODE system
def combined_0(t, Z):
    comps = Z[:7]
    acts = Z[7:]
    return deriv_0(t, comps) + activity_chain(t, acts)


# Combined ODE system
def combined(t, Z):
    comps = Z[:18]
    acts = Z[18:]
    return deriv(t, comps) + activity_chain(t, acts)


# Combined ODE system
def combined_2(t, Z):
    comps = Z[:18]
    acts = Z[18:]
    return deriv_2(t, comps) + activity_chain(t, acts)


tspan = (0, 0.5)
t_eval = np.linspace(*tspan, 167)
y0 = [0.00889, 0, 0, 0, 0, 0, 0]
A0 = [2000, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Z0 = y0 + A0
sol_0 = solve_ivp(combined_0, tspan, Z0, t_eval=t_eval, method='BDF')

# Unpack
pre225ac, pre221fr, pre217at, pre213bi, pre209tl, pre213po, pre209pb = sol_0.y[:7]
Ac, Fr, Frg, At, Bia, Bib, Big, Po, Tl, Pb = sol_0.y[7:]


LET_hi_pre = Ac + Fr + At + Bia + Po
LET_lo_pre = Frg + Bib + Big + Tl + Pb


# Initial conditions: chelated Ac in md, zero elsewhere
y0 = [sol_0.y[0, -1], 0, 0, 0.00889 - sol_0.y[0, -1], 0, 0, sol_0.y[1, -1], 0, sol_0.y[2, -1], 0, sol_0.y[3, -1], 0, sol_0.y[4, -1], 0, sol_0.y[5, -1], 0, sol_0.y[6, -1], 0]  # concentrations
A0 = [sol_0.y[7, -1], sol_0.y[8, -1], sol_0.y[9, -1], sol_0.y[10, -1], sol_0.y[11, -1], sol_0.y[12, -1], sol_0.y[13, -1], sol_0.y[14, -1], sol_0.y[15, -1], sol_0.y[16, -1]]   # activity states
Z0 = y0 + A0

tspan = (0, 3.0)  # hours
t_eval = np.linspace(*tspan, 1000)

sol_1 = solve_ivp(combined, tspan, Z0, t_eval=t_eval, method='BDF')

# Unpack
i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb = sol_1.y[:18]
Ac, Fr, Frg, At, Bia, Bib, Big, Po, Tl, Pb = sol_1.y[18:]


Ac_dose_rate = Ac * (S_values['S_i_225Ac'] + (S_values['S_m_225Ac'] * m225ac + S_values['S_c_225Ac'] * c225ac) / ((i225ac + m225ac + c225ac) * N_cell))
Fr_dose_rate = Fr * (S_values['S_i_221Fr'] + (S_values['S_c_221Fr'] * c221fr) / ((i221fr + c221fr) * N_cell))
Frg_dose_rate = Frg * (S_values['S_i_221Frg'] + (S_values['S_c_221Frg'] * c221fr) / ((i221fr + c221fr) * N_cell))
At_dose_rate = At * (S_values['S_i_217At'] + (S_values['S_c_217At'] * c217at) / ((i217at + c217at) * N_cell))
Bia_dose_rate = Bia * (S_values['S_i_213Bia'] + (S_values['S_c_213Bia'] * c213bi) / ((i213bi + c213bi) * N_cell))
Bib_dose_rate = Bib * (S_values['S_i_213Bib'] + (S_values['S_c_213Bib'] * c213bi) / ((i213bi + c213bi) * N_cell))
Big_dose_rate = Big * (S_values['S_i_213Big'] + (S_values['S_c_213Big'] * c213bi) / ((i213bi + c213bi) * N_cell))
Po_dose_rate = Po * (S_values['S_i_213Po'] + (S_values['S_c_213Po'] * c213po) / ((i213po + c213po) * N_cell))
Tl_dose_rate = Tl * (S_values['S_i_209Tl'] + (S_values['S_c_209Tl'] * c209tl) / ((i209tl + c209tl) * N_cell))
Pb_dose_rate = Pb * (S_values['S_i_209Pb'] + (S_values['S_c_209Pb'] * c209pb) / ((i209pb + c209pb) * N_cell))

HighLET_dose_rate_1 = Ac_dose_rate + Fr_dose_rate + At_dose_rate + Bia_dose_rate + Po_dose_rate
LowLET_dose_rate_1 = Frg_dose_rate + Bib_dose_rate + Big_dose_rate + Tl_dose_rate + Pb_dose_rate

Ac225iprop = i225ac[-1] / (i225ac[-1] + m225ac[-1] + c225ac[-1])
Fr221iprop = i221fr[-1] / (i221fr[-1] + c221fr[-1])
At217iprop = i217at[-1] / (i217at[-1] + c217at[-1])
Bia213iprop = i213bi[-1] / (i213bi[-1] + c213bi[-1])
Po213iprop = i213po[-1] / (i213po[-1] + c213po[-1])
Frg221iprop = i221fr[-1] / (i221fr[-1] + c221fr[-1])
Bib213iprop = i213bi[-1] / (i213bi[-1] + c213bi[-1])
Big213iprop = i213bi[-1] / (i213bi[-1] + c213bi[-1])
Tl209iprop = i209tl[-1] / (i209tl[-1] + c209tl[-1])
Pb209iprop = i209pb[-1] / (i209pb[-1] + c209pb[-1])

# Initial conditions: chelated Ac in md, zero elsewhere
y0 = [0, sol_1.y[1, -1], sol_1.y[2, -1], 0, sol_1.y[4, -1], sol_1.y[5, -1], 0, sol_1.y[7, -1], 0, sol_1.y[9, -1], 0, sol_1.y[11, -1], 0, sol_1.y[13, -1], 0, sol_1.y[15, -1], 0, sol_1.y[17, -1]]  # concentrations
A0 = [sol_1.y[18, -1] * (1 - Ac225iprop),
      sol_1.y[19, -1] * (1 - Fr221iprop),
      sol_1.y[20, -1] * (1 - Frg221iprop),
      sol_1.y[21, -1] * (1 - At217iprop),
      sol_1.y[22, -1] * (1 - Bia213iprop),
      sol_1.y[23, -1] * (1 - Bib213iprop),
      sol_1.y[24, -1] * (1 - Big213iprop),
      sol_1.y[25, -1] * (1 - Po213iprop),
      sol_1.y[26, -1] * (1 - Tl209iprop),
      sol_1.y[27, -1] * (1 - Pb209iprop)]   # activity states
Z0 = y0 + A0

tspan = (0, 168)  # hours
t_eval = np.linspace(*tspan, 1000)

sol_2 = solve_ivp(combined_2, tspan, Z0, t_eval=t_eval, method='BDF')

# Unpack
i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb = sol_2.y[:18]
Ac, Fr, Frg, At, Bia, Bib, Big, Po, Tl, Pb = sol_2.y[18:]

print(Ac_dose_rate[-1])

Ac_dose_rate = Ac * ((S_values['S_m_225Ac'] * (i225ac + m225ac) + S_values['S_c_225Ac'] * c225ac) / ((i225ac + m225ac + c225ac) * N_cell * np.exp(0.0277 * sol_2.t)))
print(Ac_dose_rate[0])
Fr_dose_rate = Fr * ((S_values['S_m_221Fr'] * i221fr + S_values['S_c_221Fr'] * c221fr) / ((i221fr + c221fr) * N_cell * np.exp(0.0277 * sol_2.t)))
Frg_dose_rate = Frg * ((S_values['S_m_221Frg'] * i221fr + S_values['S_c_221Frg'] * c221fr) / ((i221fr + c221fr) * N_cell * np.exp(0.0277 * sol_2.t)))
At_dose_rate = At * ((S_values['S_m_217At'] * i217at + S_values['S_c_217At'] * c217at) / ((i217at + c217at) * N_cell * np.exp(0.0277 * sol_2.t)))
Bia_dose_rate = Bia * ((S_values['S_m_213Bia'] * i213bi + S_values['S_c_213Bia'] * c213bi) / ((i213bi + c213bi) * N_cell * np.exp(0.0277 * sol_2.t)))
Bib_dose_rate = Bib * ((S_values['S_m_213Bib'] * i213bi + S_values['S_c_213Bib'] * c213bi) / ((i213bi + c213bi) * N_cell * np.exp(0.0277 * sol_2.t)))
Big_dose_rate = Big * ((S_values['S_m_213Big'] * i213bi + S_values['S_c_213Big'] * c213bi) / ((i213bi + c213bi) * N_cell * np.exp(0.0277 * sol_2.t)))
Po_dose_rate = Po * ((S_values['S_m_213Po'] * i213po + S_values['S_c_213Po'] * c213po) / ((i213po + c213po) * N_cell * np.exp(0.0277 * sol_2.t)))
Tl_dose_rate = Tl * ((S_values['S_m_209Tl'] * i209tl + S_values['S_c_209Tl'] * c209tl) / ((i209tl + c209tl) * N_cell * np.exp(0.0277 * sol_2.t)))
Pb_dose_rate = Pb * ((S_values['S_m_209Pb'] * i209pb + S_values['S_c_209Pb'] * c209pb) / ((i209pb + c209pb) * N_cell * np.exp(0.0277 * sol_2.t)))

HighLET_dose_rate_2 = Ac_dose_rate + Fr_dose_rate + At_dose_rate + Bia_dose_rate + Po_dose_rate
LowLET_dose_rate_2 = Frg_dose_rate + Bib_dose_rate + Big_dose_rate + Tl_dose_rate + Pb_dose_rate

HighLET_dose_1 = cumtrapz(HighLET_dose_rate_1, sol_1.t)
HighLET_dose_2 = cumtrapz(HighLET_dose_rate_2, sol_2.t)
LowLET_dose_1 = cumtrapz(LowLET_dose_rate_1, sol_1.t)
LowLET_dose_2 = cumtrapz(LowLET_dose_rate_2, sol_2.t)

# Plot: Dose rate and dose
fig, ax1 = plt.subplots(figsize=(8, 5))

color1 = 'tab:blue'
ax1.set_xlabel('Time (h)')
ax1.set_ylabel('Dose Rate (Gy per h)', color=color1)
ax1.plot(sol_1.t, HighLET_dose_rate_1, color='steelblue')
line1, = ax1.plot(sol_2.t + 3, HighLET_dose_rate_2, color='steelblue', label=' ')
ax1.plot(sol_1.t, LowLET_dose_rate_1, color='dodgerblue')
line2, = ax1.plot(sol_2.t + 3, LowLET_dose_rate_2, color='dodgerblue', label=' ')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(0.1, 200)
ax1.set_ylim(3e-6, 0.3)
ax1.yaxis.grid(True, which='both', ls='--', linewidth=0.5, color='#aab7cc')
ax1.xaxis.grid(True, which='both', ls='--', linewidth=0.5)

# Second axis for cumulative dose
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Accumulated Dose (Gy)', color=color2)
ax2.plot(sol_1.t[1:], HighLET_dose_1, color='crimson')
line3, = ax2.plot(np.concatenate((np.expand_dims(sol_1.t[-1], axis=0), sol_2.t[1:] + 3)), np.concatenate((np.expand_dims(HighLET_dose_1[-1], axis=0), HighLET_dose_2 + HighLET_dose_1[-1])), color='crimson', label=' ')
ax2.plot(sol_1.t[1:], LowLET_dose_1, color='tomato')
line4, = ax2.plot(np.concatenate((np.expand_dims(sol_1.t[-1], axis=0), sol_2.t[1:] + 3)), np.concatenate((np.expand_dims(LowLET_dose_1[-1], axis=0), LowLET_dose_2 + LowLET_dose_1[-1])), color='tomato', label=' ')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_yscale('log')
ax2.set_ylim(1e-5, 20)
ax2.yaxis.grid(True, which='both', ls='--', linewidth=0.5, color='#d2a8a8')

line5 = plt.axvline(3, color='gray', linestyle='--', label='Cell Plating')

lines = [line1, line3, line2, line4, line5]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left', title='High / Low LET', ncol=3, alignment='left', columnspacing=0)
plt.title('Dose Rate and Accumulated Dose during Ac-225 Treatment $A_0=2 kBq$')
fig.tight_layout()
#fig.legend()
plt.show()