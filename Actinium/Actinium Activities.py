#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.lines import Line2D
#from scipy.integrate import solve_ivp
#
## Decay constants (λ in h⁻¹)
#λ = {
#    'Ac225': 2.91e-3,
#    'Fr221': 8.64,
#    'At217': 7.60e4,
#    'Bi213': 0.911,
#    'Po213': 6.73e5,
#    'Tl209': 3.57,
#    'Pb209': 0.215,
#}
## Branching fractions
#p = {
#    'Fr221_γ': 0.114,
#    'Bi213_α': 0.0214,
#    'Bi213_β': 0.9786,
#    'Bi213_γ': 0.259,
#}
#
#def chain_deriv(t, A):
#    Ac225, Fr221, Fr221g, At217, Bi213a, Bi213b, Bi213g, Po213, Tl209, Pb209 = A
#
#    dAc225 = -λ['Ac225'] * Ac225
#    dFr221 =  λ['Fr221'] * (Ac225 - Fr221)
#    dFr221g = p['Fr221_γ'] * dFr221
#    dAt217 =  λ['At217'] * (Fr221 - At217)
#    dBi213a = λ['Bi213'] * (p['Bi213_α'] * At217 - Bi213a)
#    dBi213b = λ['Bi213'] * (p['Bi213_β'] * At217 - Bi213b)
#    dBi213g = p['Bi213_γ'] * λ['Bi213'] * (At217 - Bi213g)
#    dPo213 = λ['Po213'] * (Bi213b - Po213)
#    dTl209 = λ['Tl209'] * (Bi213a - Tl209)
#    dPb209 = λ['Pb209'] * (Po213 + Tl209 - Pb209)
#
#    return [dAc225, dFr221, dFr221g, dAt217,
#            dBi213a, dBi213b, dBi213g,
#            dPo213, dTl209, dPb209]
#
#def chain_deriv_2(t, A):
#    Ac225, Fr221, Fr221g, At217, Bi213a, Bi213b, Bi213g, Po213, Tl209, Pb209 = A
#
#    dAc225 = -λ['Ac225'] * Ac225
#    dFr221 =  λ['Fr221'] * (Ac225 - Fr221)
#    dFr221g = p['Fr221_γ'] * dFr221
#    dAt217 =  λ['At217'] * (Fr221 - At217)
#    dBi213a = λ['Bi213'] * (p['Bi213_α'] * At217 - Bi213a)
#    dBi213b = λ['Bi213'] * (p['Bi213_β'] * At217 - Bi213b)
#    dBi213g = p['Bi213_γ'] * λ['Bi213'] * (At217 - Bi213g)
#    dPo213 = λ['Po213'] * (Bi213b - Po213)
#    dTl209 = λ['Tl209'] * (Bi213a - Tl209)
#    dPb209 = λ['Pb209'] * (Po213 + Tl209 - Pb209)
#
#    return [dAc225, dFr221, dFr221g, dAt217,
#            dBi213a, dBi213b, dBi213g,
#            dPo213, dTl209, dPb209]
#
# Initial condition: 2000 Bq of Ac-225 and zero for others
#A0 = [2000] + [0]*9
#
## Time span from 0 to 1.5 h
#t_eval = np.linspace(0, 3.5, 1002)
#
## Solve ODE system
#sol = solve_ivp(chain_deriv, [0, t_eval[-1]], A0, t_eval=t_eval)  # SciPy’s solve_ivp :contentReference[oaicite:1]{index=1}

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.lines import Line2D

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
y0 = [0.00889, 0, 0, 0, 0, 0, 0]
A0 = [2000, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Z0 = y0 + A0
sol_0 = solve_ivp(combined_0, tspan, Z0, t_eval=t_eval, method='BDF')

# Unpack
pre225ac, pre221fr, pre217at, pre213bi, pre209tl, pre213po, pre209pb = sol_0.y[:7]
Ac_0, Fr_0, Frg_0, At_0, Bia_0, Bib_0, Big_0, Po_0, Tl_0, Pb_0 = sol_0.y[7:]


# Initial conditions: chelated Ac in md, zero elsewhere
y0 = [sol_0.y[0, -1], 0, 0, 0.00889 - sol_0.y[0, -1], 0, 0, sol_0.y[1, -1], 0, sol_0.y[2, -1], 0, sol_0.y[3, -1], 0, sol_0.y[4, -1], 0, sol_0.y[5, -1], 0, sol_0.y[6, -1], 0]  # concentrations
A0 = [sol_0.y[7, -1], sol_0.y[8, -1], sol_0.y[9, -1], sol_0.y[10, -1], sol_0.y[11, -1], sol_0.y[12, -1], sol_0.y[13, -1], sol_0.y[14, -1], sol_0.y[15, -1], sol_0.y[16, -1]]   # activity states
Z0 = y0 + A0

tspan = (0, 3.0)  # hours
t_eval = np.linspace(*tspan, 1000)

sol_1 = solve_ivp(combined, tspan, Z0, t_eval=t_eval, method='BDF')
sol_1.t += 0.5

# Unpack
i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb = sol_1.y[:18]
Ac_1, Fr_1, Frg_1, At_1, Bia_1, Bib_1, Big_1, Po_1, Tl_1, Pb_1 = sol_1.y[18:]

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
sol_2.t += 3.5

# Unpack
i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb = sol_2.y[:18]
Ac_2, Fr_2, Frg_2, At_2, Bia_2, Bib_2, Big_2, Po_2, Tl_2, Pb_2 = sol_2.y[18:]

t = np.concatenate((sol_0.t, sol_1.t, sol_2.t))
Ac = np.concatenate((Ac_0, Ac_1, Ac_2))
Fr = np.concatenate((Fr_0, Fr_1, Fr_2))
Frg = np.concatenate((Frg_0, Frg_1, Frg_2))
At = np.concatenate((At_0, At_1, At_2))
Bia = np.concatenate((Bia_0, Bia_1, Bia_2))
Bib = np.concatenate((Bib_0, Bib_1, Bib_2))
Big = np.concatenate((Big_0, Big_1, Big_2))
Po = np.concatenate((Po_0, Po_1, Po_2))
Tl = np.concatenate((Tl_0, Tl_1, Tl_2))
Pb = np.concatenate((Pb_0, Pb_1, Pb_2))

activities = [Ac, Fr, Frg, At, Bia, Bib, Big, Po, Tl, Pb]

# Plot results
plt.figure(figsize=(8, 5))
isotopes = ['Ac‑225', 'Fr‑221', 'Fr‑221 γ', 'At‑217',
            'Bi‑213 α', 'Bi‑213 β', 'Bi‑213 γ',
            'Po‑213', 'Tl‑209', 'Pb‑209', 'Experiment start', 'Cell plating']

colors = ['#E41A1C', '#4DAF4A', '#377EB8', '#FF7F00',
          '#A65628', '#66A61E', '#7570B3', '#D95F02',
          '#E7298A', '#999999']

linetypes = ['-', '-', '-', (0, (20, 20)),
             '-', '-', '-',
             (0, (20, 20)), '-', '-']

custom_lines = []

for color, linetype, name, a in zip(colors, linetypes, isotopes, activities):
    plt.plot(t, a, color=color, linestyle=linetype)
    custom_lines.append(Line2D([0], [0], linestyle='-', color=color))
plt.axvline(0.5, color='gray', linestyle=':', label='Experiment Start')
plt.axvline(3.5, color='gray', linestyle='--', label='Cell Plating')
custom_lines.append(Line2D([0], [0], linestyle=':', color='gray'))
custom_lines.append(Line2D([0], [0], linestyle='--', color='gray'))

plt.yscale('log')
plt.xscale('log')
plt.xlim([0.1, 171.5])
plt.ylim([0.1, 3e3])
plt.grid(True, ls="--", linewidth=0.5)
plt.xlabel('Time (h)')
plt.ylabel('Activity (Bq)')
plt.title('Ac-225 Decay Chain Activities')
plt.legend(custom_lines, isotopes, loc='upper right', ncol=2)
plt.tight_layout()
plt.show()
