
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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
Rtot = 0.00516

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

    diu = koff*m_u + iu*i225ac - kon*(Rtot - m225ac - m_u)*iu
    dmu = kon*(Rtot - m225ac - m_u)*iu + krel*cu + mu225ac*m225ac - (koff + kint)*mu225ac
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

    return [di225ac, dm225ac, dc225ac, diu, dmu, dcu, di221fr, dc221fr, di217at, dc217at, di213bi, dc213bi, di209tl, dc209tl, di213po, dc213po, di209pb, dc209pb]


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


tspan = (0, 0.5)
t_eval = np.linspace(*tspan, 167)
y0 = [0.00889, 0, 0, 0, 0, 0, 0]
A0 = [2000, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Z0 = y0 + A0
sol_0 = solve_ivp(combined_0, tspan, Z0, t_eval=t_eval, method='LSODA')

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

sol_1 = solve_ivp(combined, tspan, Z0, t_eval=t_eval, method='LSODA')

# Unpack
i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb = sol_1.y[:18]
Ac, Fr, Frg, At, Bia, Bib, Big, Po, Tl, Pb = sol_1.y[18:]

# Compute α (high LET) sources: Bi213 α + Po213 + At217 + Fr221 decay
#alpha = λ['Bi213']*p['Bi213_a']*At + λ['Po213']*Po + λ['At217']*At + λ['Fr221']*Fr
alpha = Ac + Fr + At + Bia + Po

# Compute β+γ (low LET)
#beta_gamma = λ['Bi213']*p['Bi213_b']*At + p['Fr221_g']*λ['Fr221']*(Ac - Fr) + \
#             p['Bi213_g']*λ['Bi213']*(At - Big) + λ['Tl209']*Tl + λ['Pb209']*(Po + Tl)
beta_gamma = Frg + Bib + Big + Tl + Pb

## Multiply compartment fractions (interstitial = bu, bound = bd, cytoplasm = mu_+cd)
#interstitial = id + i_s
#bound = bd
#cyto = cd + cs

LET_hi_int = Ac * i225ac / (i225ac + m225ac + c225ac) +\
             Fr * i221fr / (i221fr + c221fr) +\
             At * i217at / (i217at + c217at) +\
             Bia * i213bi / (i213bi + c213bi) +\
             Po * i213po / (i213po + c213po)
LET_hi_bound = Ac * m225ac / (i225ac + m225ac + c225ac)
LET_hi_cyto = Ac * c225ac / (i225ac + m225ac + c225ac) +\
             Fr * c221fr / (i221fr + c221fr) +\
             At * c217at / (i217at + c217at) +\
             Bia * c213bi / (i213bi + c213bi) +\
             Po * c213po / (i213po + c213po)

LET_lo_int = Frg * i221fr / (i221fr + c221fr) +\
             Bib * i213bi / (i213bi + c213bi) +\
             Big * i213bi / (i213bi + c213bi) +\
             Tl * i209tl / (i209tl + c209tl) +\
             Pb * i209pb / (i209pb + c209pb)
LET_lo_cyto = Frg * c221fr / (i221fr + c221fr) +\
             Bib * c213bi / (i213bi + c213bi) +\
             Big * c213bi / (i213bi + c213bi) +\
             Tl * c209tl / (i209tl + c209tl) +\
             Pb * c209pb / (i209pb + c209pb)

# Plot
plt.figure(figsize=(8,5))
plt.plot(sol_0.t, LET_hi_pre, label='High LET pre Experiment', color='#FF4500')
plt.plot(sol_0.t, LET_lo_pre, label='Low LET pre Experiment', color='#1E90FF', ls='--')
plt.plot(sol_1.t + 0.5, LET_hi_int, label='High LET Interstitial', color='#B22222')
plt.plot(sol_1.t + 0.5, LET_hi_bound, label='High LET Bound', color='#FF8C00')
plt.plot(sol_1.t + 0.5, LET_hi_cyto, label='High LET Cytoplasm', color='#FFD4A3')
plt.plot(sol_1.t + 0.5, LET_lo_int, label='Low LET Interstitial', color='#004080', ls='--')
plt.plot(sol_1.t + 0.5, LET_lo_cyto, label='Low LET Cytoplasm', color='#7FDBFF', ls='--')

plt.axvline(0.5, color='black', linestyle='--', lw=1)
plt.text(0.51, plt.ylim()[1]*0.8, 'Experiment start', rotation=90, va='center', fontsize=6)

plt.xlim([0, 3.5])
plt.ylim([0, 7200])
plt.xlabel('Time (h)')
plt.ylabel('Activity (Bq)')
plt.title('Compartment- and LET-specific Activities during Ac-225 Treatment')
plt.legend()
plt.grid(True, ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()
