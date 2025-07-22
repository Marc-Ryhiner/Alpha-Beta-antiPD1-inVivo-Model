import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

# Decay constants (λ in h⁻¹)
λ = {
    'Ac225': 2.91e-3,
    'Fr221': 8.64,
    'At217': 7.60e4,
    'Bi213': 0.911,
    'Po213': 6.73e5,
    'Tl209': 3.57,
    'Pb209': 0.215,
}
# Branching fractions
p = {
    'Fr221_γ': 0.114,
    'Bi213_α': 0.0214,
    'Bi213_β': 0.9786,
    'Bi213_γ': 0.259,
}

def chain_deriv(t, A):
    Ac225, Fr221, Fr221g, At217, Bi213a, Bi213b, Bi213g, Po213, Tl209, Pb209 = A

    dAc225 = -λ['Ac225'] * Ac225
    dFr221 =  λ['Fr221'] * (Ac225 - Fr221)
    dFr221g = p['Fr221_γ'] * dFr221
    dAt217 =  λ['At217'] * (Fr221 - At217)
    dBi213a = λ['Bi213'] * (p['Bi213_α'] * At217 - Bi213a)
    dBi213b = λ['Bi213'] * (p['Bi213_β'] * At217 - Bi213b)
    dBi213g = p['Bi213_γ'] * λ['Bi213'] * (At217 - Bi213g)
    dPo213 = λ['Po213'] * (Bi213b - Po213)
    dTl209 = λ['Tl209'] * (Bi213a - Tl209)
    dPb209 = λ['Pb209'] * (Po213 + Tl209 - Pb209)

    return [dAc225, dFr221, dFr221g, dAt217,
            dBi213a, dBi213b, dBi213g,
            dPo213, dTl209, dPb209]

# Initial condition: 2000 Bq of Ac-225 and zero for others
A0 = [2000] + [0]*9

# Time span from 0 to 1.5 h
t_eval = np.linspace(0, 3.5, 1002)

# Solve ODE system
sol = solve_ivp(chain_deriv, [0, t_eval[-1]], A0, t_eval=t_eval)  # SciPy’s solve_ivp :contentReference[oaicite:1]{index=1}

# Plot results
plt.figure(figsize=(8, 5))
isotopes = ['Ac‑225', 'Fr‑221', 'Fr‑221 γ', 'At‑217',
            'Bi‑213 α', 'Bi‑213 β', 'Bi‑213 γ',
            'Po‑213', 'Tl‑209', 'Pb‑209']

colors = ['#E41A1C', '#4DAF4A', '#377EB8', '#FF7F00',
          '#A65628', '#66A61E', '#7570B3', '#D95F02',
          '#E7298A', '#999999']

linetypes = ['-', '-', '-', (0, (20, 20)),
             '-', '-', '-',
             (0, (20, 20)), '-', '-']

custom_lines = []

for idx, (color, linetype, name) in enumerate(zip(colors, linetypes, isotopes)):
    plt.plot(sol.t, sol.y[idx], color=color, linestyle=linetype)
    custom_lines.append(Line2D([0], [0], linestyle='-', color=color))

# Vertical line at Experiment start = 0.5 h
plt.axvline(0.5, color='k', linestyle='--', linewidth=1)
plt.text(0.51, 1.1, 'Experiment start', rotation=90, va='bottom', ha='left', fontsize=6)
plt.yscale('log')
plt.xlim([0, 3.5])
plt.ylim([1, 3e3])
plt.grid(True, ls="--", linewidth=0.5)
plt.xlabel('Time (h)')
plt.ylabel('Activity (Bq)')
plt.title('Ac-225 Decay Chain Activities')
plt.legend(custom_lines, isotopes, loc='lower right', fontsize='small', ncol=2)
plt.tight_layout()
plt.show()
