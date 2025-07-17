import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # enables 3D

# Your parameters
kon, koff = 131.4, 1.611    # mL·nmol⁻¹·h⁻¹, h⁻¹
kint, krel = 0.7602, 2.138  # h⁻¹
Si_per_N = 5.25e-8 / 100000 # Gy Bq-1 h-1 cell-1
Sb, Sc = 0.374, 0.713       # Gy Bq-1 h-1


# Define grid
R = np.logspace(np.log10(1e-3), np.log10(1e-1), 300)
N = R * 19379845
Si = N * Si_per_N
ci0 = np.logspace(np.log10(2.5e-3), np.log10(2.5e-1), 300)
A = ci0 * 40
Rg, ci0g = np.meshgrid(R, ci0)
Ng, Ag = np.meshgrid(N, A)
Sig, Ag = np.meshgrid(Si, A)

# Solve quadratic for ci∞
a = kon
b = koff + kon * (Rg - ci0g)
c = -koff * ci0g
ci_inf = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

# Compute cb∞
cb_inf = (kon * ci_inf * Rg) / (
    (koff + kon * ci_inf) * (1 + kint / krel)
)

# Compute cc∞
cc_inf = (kint * cb_inf) / krel


# p_con
p_con = Sig / (Sig + (Sb * cb_inf + Sc * cc_inf) / (Ng * ci0)) * 100

# Map to colors
cmap = plt.get_cmap('magma')
colors = cmap((p_con - p_con.min()) / (p_con.max() - p_con.min()))

# Plot
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Ng, Ag, np.zeros_like(p_con),
                facecolors=colors, shade=False)

# Top-down view
ax.view_init(elev=90, azim=-90)
ax.xaxis.pane.set_edgecolor('none')
ax.yaxis.pane.set_edgecolor('none')
ax.zaxis.pane.set_edgecolor('none')
ax.xaxis.pane.set_facecolor((1,1,1,0))
ax.yaxis.pane.set_facecolor((1,1,1,0))
ax.zaxis.pane.set_facecolor((1,1,1,0))
ax.set_zticks([])
ax.grid(False)
ax.zaxis.pane.set_visible(False)
ax.set_xlabel('Number of cancer cells')
ax.set_ylabel('Initial activity (MBq)')
ax.set_zlim(0, 0.01)
ax.set_xlim([0, 2e6])
ax.set_ylim([0, 10])
ax.set_xticks([0,1e6,2e6])
ax.set_yticks([0,5,10])
ax.tick_params(axis='y', which='major', pad=10)
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 20

# Colorbar
mappable = plt.cm.ScalarMappable(cmap=cmap)
mappable.set_array(p_con)
cbar = plt.colorbar(mappable, ax=ax, pad=0.001, shrink=0.6, aspect=10)
cbar.set_label('Clinical contribution of convergent effects (%)', labelpad=15)

plt.tight_layout()
plt.show()
