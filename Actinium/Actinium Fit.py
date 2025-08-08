import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Global cache for interpolated functions (used inside memory_integral)
interpolators = {}

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


# Define memory kernel function for G_x^num
def memory_integral(t, dD_func, px_f, px_s, lambd_f, lambd_s):
    def integrand(t1):
        def inner_integrand(t2):
            return dD_func(t2) * (px_f * np.exp(-lambd_f * (t1 - t2)) + px_s * np.exp(-lambd_s * (t1 - t2)))
        inner_val, _ = quad(inner_integrand, 0, t1)
        return dD_func(t1) * inner_val

    result, _ = quad(integrand, 0, t)
    return 2 * result


def ode_help(t, y, ci0, A0, p):
    i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb, Ac, Fr, Frg, At, Bia, Bib, Big, Po, Tl, Pb = y

    kon, koff, kint, krel = p['k_on'], p['k_off'], p['k_int'], p['k_rel']
    N, R = p['Ncell'], p['R']

    di225ac = koff * m225ac - (kon * (R - m225ac - m_u) + mu225ac) * i225ac
    dm225ac = kon * (R - m225ac - m_u) * i225ac + krel * c225ac - (koff + kint + mu225ac) * m225ac
    dc225ac = kint * m225ac - (krel + mu225ac) * c225ac

    diu = koff * m_u + mu225ac * i225ac - kon * (R - m225ac - m_u) * iu
    dm_u = kon * (R - m225ac - m_u) * iu + krel * cu + mu225ac * m225ac - (koff + kint) * m_u
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

    dAc = -mu225ac * Ac
    dFr = mu221fr * (Ac - Fr)
    dFrg = pr['Fr221_g'] * dFr
    dAt = mu217at * (Fr - At)
    dBia = mu213bi * (pr['Bi213_a'] * At - Bia)
    dBib = mu213bi * (pr['Bi213_b'] * At - Bib)
    dBig = pr['Bi213_g'] * mu213bi * (At - Big)
    dPo = mu213po * (Bib - Po)
    dTl = mu209tl * (Bia - Tl)
    dPb = mu209pb * (Po + Tl - Pb)

    return [di225ac, dm225ac, dc225ac, diu, dm_u, dcu, di221fr, dc221fr, di217at, dc217at, di213bi, dc213bi, di209tl, dc209tl, di213po, dc213po, di209pb, dc209pb,
            dAc, dFr, dFrg, dAt, dBia, dBib, dBig, dPo, dTl, dPb]


def ode_help_2(t, y, ci0, A0, p):
    i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb, Ac, Fr, Frg, At, Bia, Bib, Big, Po, Tl, Pb = y

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

    dAc = -mu225ac * Ac
    dFr = mu221fr * (Ac - Fr)
    dFrg = pr['Fr221_g'] * dFr
    dAt = mu217at * (Fr - At)
    dBia = mu213bi * (pr['Bi213_a'] * At - Bia)
    dBib = mu213bi * (pr['Bi213_b'] * At - Bib)
    dBig = pr['Bi213_g'] * mu213bi * (At - Big)
    dPo = mu213po * (Bib - Po)
    dTl = mu209tl * (Bia - Tl)
    dPb = mu209pb * (Po + Tl - Pb)

    return [di225ac, dm225ac, dc225ac, diu, dm_u, dcu, di221fr, dc221fr, di217at, dc217at, di213bi, dc213bi, di209tl, dc209tl, di213po, dc213po, di209pb, dc209pb,
            dAc, dFr, dFrg, dAt, dBia, dBib, dBig, dPo, dTl, dPb]


def ode_system(t, y, ci0, A0, alpha, beta, px_f, px_s, lambd_f, lambd_s, alpha_low, p):
    eps = 1e-12

    i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb, Ac, Fr, Frg, At, Bia, Bib, Big, Po, Tl, Pb, D_high, D_low, S = y

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
             Po * (S_values['S_i_209Pb'] + S_values['S_c_209Pb'] * c209pb / (N * (i209pb + c209pb) + eps))

    # Define dD function for integration
    def dD_func(tau):
        Ac_tau = interpolators['Ac1'](tau)
        Fr_tau = interpolators['Fr1'](tau)
        At_tau = interpolators['At1'](tau)
        Bia_tau = interpolators['Bia1'](tau)
        Po_tau = interpolators['Po1'](tau)
        i225ac_tau = interpolators['i225ac1'](tau)
        i221fr_tau = interpolators['i221fr1'](tau)
        i217at_tau = interpolators['i217at1'](tau)
        i213bi_tau = interpolators['i213bi1'](tau)
        i213po_tau = interpolators['i213po1'](tau)
        m225ac_tau = interpolators['m225ac1'](tau)
        c225ac_tau = interpolators['c225ac1'](tau)
        c221fr_tau = interpolators['c221fr1'](tau)
        c217at_tau = interpolators['c217at1'](tau)
        c213bi_tau = interpolators['c213bi1'](tau)
        c213po_tau = interpolators['c213po1'](tau)
        return Ac_tau * (S_values['S_i_225Ac'] + (S_values['S_m_225Ac'] * m225ac_tau + S_values['S_c_225Ac'] * c225ac_tau) / (N * (i225ac_tau + m225ac_tau + c225ac_tau) + eps)) + \
               Fr_tau * (S_values['S_i_221Fr'] + S_values['S_c_221Fr'] * c221fr_tau / (N * (i221fr_tau + c221fr_tau) + eps)) + \
               At_tau * (S_values['S_i_217At'] + S_values['S_c_217At'] * c217at_tau / (N * (i217at_tau + c217at_tau) + eps)) + \
               Bia_tau * (S_values['S_i_213Bia'] + S_values['S_c_213Bia'] * c213bi_tau / (N * (i213bi_tau + c213bi_tau) + eps)) + \
               Po_tau * (S_values['S_i_213Po'] + S_values['S_c_213Po'] * c213po_tau / (N * (i213po_tau + c213po_tau) + eps))

    G_num = memory_integral(t, dD_func, px_f, px_s, lambd_f, lambd_s)
    dG_num = 2 * dD_high * quad(lambda t2: dD_func(t2) * np.exp(-lambd_f * (t - t2)) * np.exp(-lambd_s * (t - t2)), 0, t)[0]

    sum_term = alpha * dD_high + beta * dG_num
    exp_term = - (alpha * D_high + beta * G_num)
    dS = -sum_term * np.exp(exp_term) - alpha_low * dD_low * np.exp(-alpha_low * D_low)

    return [di225ac, dm225ac, dc225ac, diu, dm_u, dcu, di221fr, dc221fr, di217at, dc217at, di213bi, dc213bi, di209tl, dc209tl, di213po, dc213po, di209pb, dc209pb,
            dAc, dFr, dFrg, dAt, dBia, dBib, dBig, dPo, dTl, dPb,
            dD_low, dD_high, dS]


def ode_system_2(t, y, ci0, A0, alpha, beta, px_f, px_s, lambd_f, lambd_s, alpha_low, p):
    eps = 1e-12

    i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb, Ac, Fr, Frg, At, Bia, Bib, Big, Po, Tl, Pb, D_high, D_low, S = y

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

    # Define dD function for integration
    def dD_func(tau):
        Ac_tau = interpolators['Ac2'](tau)
        Fr_tau = interpolators['Fr2'](tau)
        At_tau = interpolators['At2'](tau)
        Bia_tau = interpolators['Bia2'](tau)
        Po_tau = interpolators['Po2'](tau)
        i225ac_tau = interpolators['i225ac2'](tau)
        i221fr_tau = interpolators['i221fr2'](tau)
        i217at_tau = interpolators['i217at2'](tau)
        i213bi_tau = interpolators['i213bi2'](tau)
        i213po_tau = interpolators['i213po2'](tau)
        m225ac_tau = interpolators['m225ac2'](tau)
        c225ac_tau = interpolators['c225ac2'](tau)
        c221fr_tau = interpolators['c221fr2'](tau)
        c217at_tau = interpolators['c217at2'](tau)
        c213bi_tau = interpolators['c213bi2'](tau)
        c213po_tau = interpolators['c213po2'](tau)
        return Ac_tau * ((S_values['S_m_225Ac'] * (i225ac_tau + m225ac_tau) + S_values['S_c_225Ac'] * c225ac_tau) / (N * np.exp(t * 0.0277) * (i225ac_tau + m225ac_tau + c225ac_tau) + eps)) + \
               Fr_tau * ((S_values['S_m_221Fr'] * i221fr_tau + S_values['S_c_221Fr'] * c221fr_tau) / (N * np.exp(t * 0.0277) * (i221fr_tau + c221fr_tau) + eps)) + \
               At_tau * ((S_values['S_m_217At'] * i217at_tau + S_values['S_c_217At'] * c217at_tau) / (N * np.exp(t * 0.0277) * (i217at_tau + c217at_tau) + eps)) + \
               Bia_tau * ((S_values['S_m_213Bia'] * i213bi_tau + S_values['S_c_213Bia'] * c213bi_tau) / (N * np.exp(t * 0.0277) * (i213bi_tau + c213bi_tau) + eps)) + \
               Po_tau * ((S_values['S_m_213Po'] * i213po_tau + S_values['S_c_213Po'] * c213po_tau) / (N * np.exp(t * 0.0277) * (i213po_tau + c213po_tau) + eps))

    G_num = memory_integral(t, dD_func, px_f, px_s, lambd_f, lambd_s)
    dG_num = 2 * dD_high * quad(lambda t2: dD_func(t2) * np.exp(-lambd_f * (t - t2)) * np.exp(-lambd_s * (t - t2)), 0, t)[0]

    sum_term = alpha * dD_high + beta * dG_num
    exp_term = - (alpha * D_high + beta * G_num)
    dS = -sum_term * np.exp(exp_term) - alpha_low * dD_low * np.exp(-alpha_low * D_low)

    return [di225ac, dm225ac, dc225ac, diu, dm_u, dcu, di221fr, dc221fr, di217at, dc217at, di213bi, dc213bi, di209tl, dc209tl, di213po, dc213po, di209pb, dc209pb,
            dAc, dFr, dFrg, dAt, dBia, dBib, dBig, dPo, dTl, dPb,
            dD_low, dD_high, dS]


# Simulation wrapper
def simulate_survival(ci0, A0, alpha, beta):
    y0 = [ci0 * 0.9985, 0, 0, ci0 * 0.001454, 0, 0, ci0 * 3.320e-4, 0, ci0 * 3.774e-8, 0, ci0 * 9.343e-4, 0, ci0 * 2.530e-6, 0, ci0 * 1.238e-9, 0, ci0 * 1.794e-4, 0,
          A0 * 0.9985, A0 * 0.9857, A0 * 0.1124, A0 * 0.9857, A0 * 6.259e-3, A0 * 0.2862, A0 * 0.08664, A0 * 0.2862, A0 * 3.104e-3, A0 * 0.01325,
          0, 0, 1]

    # Dense output for interpolation
    help1 = solve_ivp(
        ode_help, [0, 3], y0[:-3],
        args=(ci0, A0, base_params),
        dense_output=True, method='BDF', rtol=1e-6, atol=1e-8
    )

    # Create interpolators
    t_vals = np.linspace(0, 3, 300)
    sol_vals = help1.sol(t_vals)
    i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb, Ac, Fr, Frg, At, Bia, Bib, Big, Po, Tl, Pb = sol_vals
    interpolators['Ac1'] = interp1d(t_vals, Ac, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['Fr1'] = interp1d(t_vals, Fr, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['At1'] = interp1d(t_vals, At, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['Bia1'] = interp1d(t_vals, Bia, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['Po1'] = interp1d(t_vals, Po, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['i225ac1'] = interp1d(t_vals, i225ac, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['i221fr1'] = interp1d(t_vals, i221fr, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['i217at1'] = interp1d(t_vals, i217at, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['i213bi1'] = interp1d(t_vals, i213bi, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['i213po1'] = interp1d(t_vals, i213po, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['m225ac1'] = interp1d(t_vals, m225ac, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['c225ac1'] = interp1d(t_vals, c225ac, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['c221fr1'] = interp1d(t_vals, c221fr, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['c217at1'] = interp1d(t_vals, c217at, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['c213bi1'] = interp1d(t_vals, c213bi, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['c213po1'] = interp1d(t_vals, c213po, kind='cubic', bounds_error=False, fill_value='extrapolate')

    # Solve phase 1
    sol1 = solve_ivp(
        ode_system, [0, 3], y0,
        args=(ci0, A0, alpha, beta, px_f, px_s, lambd_f, lambd_s, alpha_low, base_params),
        dense_output=True, method='BDF', rtol=1e-6, atol=1e-8
    )

    i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb, Ac, Fr, Frg, At, Bia, Bib, Big, Po, Tl, Pb, D_high, D_low, S = sol1.y[:, -1]

    # Phase 2
    eps = 1e-12
    i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb = [0, sol1.y[1, -1], sol1.y[2, -1], 0, sol1.y[4, -1], sol1.y[5, -1], 0, sol1.y[7, -1], 0, sol1.y[9, -1], 0, sol1.y[11, -1], 0, sol1.y[13, -1], 0, sol1.y[15, -1], 0, sol1.y[17, -1]]
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
    y0_phase2 = [i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb, Ac2, Fr2, Frg2, At2, Bia2, Bib2, Big2, Po2, Tl2, Pb2, D_high, D_low, S]

    help2 = solve_ivp(
        ode_help_2, [0, 168], y0_phase2[:-3],
        args=(0, A0, base_params),
        t_eval=[168], method='BDF', rtol=1e-6, atol=1e-8, dense_output=True
    )

    # Create interpolators
    t_vals = np.linspace(0, 168, 1000)
    sol_vals = help2.sol(t_vals)
    i225ac, m225ac, c225ac, iu, m_u, cu, i221fr, c221fr, i217at, c217at, i213bi, c213bi, i209tl, c209tl, i213po, c213po, i209pb, c209pb, Ac, Fr, Frg, At, Bia, Bib, Big, Po, Tl, Pb = sol_vals
    interpolators['Ac2'] = interp1d(t_vals, Ac, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['Fr2'] = interp1d(t_vals, Fr, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['At2'] = interp1d(t_vals, At, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['Bia2'] = interp1d(t_vals, Bia, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['Po2'] = interp1d(t_vals, Po, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['i225ac2'] = interp1d(t_vals, i225ac, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['i221fr2'] = interp1d(t_vals, i221fr, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['i217at2'] = interp1d(t_vals, i217at, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['i213bi2'] = interp1d(t_vals, i213bi, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['i213po2'] = interp1d(t_vals, i213po, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['m225ac2'] = interp1d(t_vals, m225ac, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['c225ac2'] = interp1d(t_vals, c225ac, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['c221fr2'] = interp1d(t_vals, c221fr, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['c217at2'] = interp1d(t_vals, c217at, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['c213bi2'] = interp1d(t_vals, c213bi, kind='cubic', bounds_error=False, fill_value='extrapolate')
    interpolators['c213po2'] = interp1d(t_vals, c213po, kind='cubic', bounds_error=False, fill_value='extrapolate')

    sol2 = solve_ivp(
        ode_system_2, [0, 168], y0_phase2,
        args=(0, A0, alpha, beta, px_f, px_s, lambd_f, lambd_s, alpha_low, base_params),
        t_eval=[168], method='BDF', rtol=1e-6, atol=1e-8
    )

    return float(sol2.y[-1, -1])


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
        chi2 += ((log_model - log_obs) / (log_sigma)) ** 2
    return chi2

# Initial guesses and bounds for alpha and beta
x0 = [0.15, 0.02]
bounds = [
    (0.1, 5), (0.0001, 0.5)
]

# Optimization
res = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B', options={'disp': True})
print("Fitted parameters:")
print(f"alpha = {res.x[0]:.4f}, beta = {res.x[1]:.4f}")