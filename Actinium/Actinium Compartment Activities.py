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
Rtot = 0.000814


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
y0 = [0.00889 * np.exp(mu225ac * 0.5) * 0.2652946819207342, 0, 0, 0, 0, 0, 0]
A0 = [2000 * np.exp(mu225ac * 0.5) * 0.2652946819207342, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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

#print(Ac[-1] + Fr[-1] + At[-1] + Bia[-1] + Po[-1] + Frg[-1] + Bib[-1] + Big[-1] + Tl[-1] + Pb[-1])

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

Ac225iprop = i225ac[-1] / (i225ac[-1] + m225ac[-1] + c225ac[-1])
Ac225mprop = m225ac[-1] / (i225ac[-1] + m225ac[-1] + c225ac[-1])
Ac225cprop = c225ac[-1] / (i225ac[-1] + m225ac[-1] + c225ac[-1])
Uiprop = iu[-1] / (iu[-1] + m_u[-1] + cu[-1])
Umprop = m_u[-1] / (iu[-1] + m_u[-1] + cu[-1])
Ucprop = cu[-1] / (iu[-1] + m_u[-1] + cu[-1])
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

alpha = Ac + Fr + At + Bia + Po
beta_gamma = Frg + Bib + Big + Tl + Pb


LET_hi_int_2 = Ac * i225ac / (i225ac + m225ac + c225ac) +\
             Fr * i221fr / (i221fr + c221fr) +\
             At * i217at / (i217at + c217at) +\
             Bia * i213bi / (i213bi + c213bi) +\
             Po * i213po / (i213po + c213po)
LET_hi_bound_2 = Ac * m225ac / (i225ac + m225ac + c225ac)
LET_hi_cyto_2 = Ac * c225ac / (i225ac + m225ac + c225ac) +\
             Fr * c221fr / (i221fr + c221fr) +\
             At * c217at / (i217at + c217at) +\
             Bia * c213bi / (i213bi + c213bi) +\
             Po * c213po / (i213po + c213po)

LET_lo_int_2 = Frg * i221fr / (i221fr + c221fr) +\
             Bib * i213bi / (i213bi + c213bi) +\
             Big * i213bi / (i213bi + c213bi) +\
             Tl * i209tl / (i209tl + c209tl) +\
             Pb * i209pb / (i209pb + c209pb)
LET_lo_cyto_2 = Frg * c221fr / (i221fr + c221fr) +\
             Bib * c213bi / (i213bi + c213bi) +\
             Big * c213bi / (i213bi + c213bi) +\
             Tl * c209tl / (i209tl + c209tl) +\
             Pb * c209pb / (i209pb + c209pb)

# Plot
plt.figure(figsize=(8,5))
plt.plot(sol_0.t, LET_hi_pre, label='High LET pre Experiment', color='#FF4500')
plt.plot(sol_0.t, LET_lo_pre, label='Low LET pre Experiment', color='#1E90FF', ls='--')
plt.plot(sol_0.t, LET_lo_pre + LET_hi_pre, label='Total pre Experiment', color='grey')
plt.plot(sol_1.t + 0.5, LET_hi_int, label='High LET Interstitial', color='#B22222')
plt.plot(sol_1.t + 0.5, LET_hi_bound, label='High LET Bound', color='#FF8C00')
plt.plot(sol_1.t + 0.5, LET_hi_cyto, label='High LET Cytoplasm', color='#FFD4A3')
plt.plot(sol_1.t + 0.5, LET_lo_int, label='Low LET Interstitial', color='#004080', ls='--')
plt.plot(sol_1.t + 0.5, LET_lo_cyto, label='Low LET Cytoplasm', color='#7FDBFF', ls='--')
plt.plot(sol_1.t + 0.5, LET_hi_int + LET_hi_bound + LET_hi_cyto + LET_lo_int + LET_hi_cyto, label='Total', color='black')
plt.plot(sol_2.t + 3.5, LET_hi_int_2, color='#B22222')
plt.plot(sol_2.t + 3.5, LET_hi_bound_2, color='#FF8C00')
plt.plot(sol_2.t + 3.5, LET_hi_cyto_2, color='#FFD4A3')
plt.plot(sol_2.t + 3.5, LET_lo_int_2, color='#004080', ls='--')
plt.plot(sol_2.t + 3.5, LET_lo_cyto_2, color='#7FDBFF', ls='--')
plt.plot(sol_2.t + 3.5, LET_hi_int_2 + LET_hi_bound_2 + LET_hi_cyto_2 + LET_lo_int_2 + LET_hi_cyto_2, color='black')
plt.axvline(0.5, color='gray', linestyle=':', label='Experiment Start')
plt.axvline(3.5, color='gray', linestyle='--', label='Cell Plating')

#plt.plot(sol_0.t, LET_hi_pre + LET_lo_pre)
#plt.plot(sol_1.t + 0.5, LET_hi_int + LET_hi_bound + LET_hi_cyto + LET_lo_int + LET_lo_cyto)
#plt.plot(sol_2.t + 3.5, LET_hi_int_2 + LET_hi_bound_2 + LET_hi_cyto_2 + LET_lo_int_2 + LET_lo_cyto_2)

#plt.axvline(0.5, color='black', linestyle='--', lw=1)
#plt.text(0.51, plt.ylim()[1]*0.8, 'Experiment start', rotation=90, va='center', fontsize=6)

plt.xlim([0.1, 171.5])
plt.ylim([0.1, 4e3])
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Time (h)')
plt.ylabel('Activity (Bq)')
plt.title('Compartment- and LET Activities during Ac-225 Treatment for $A_{t=0.5}=2kBq$')
plt.legend()
plt.grid(True, ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()
