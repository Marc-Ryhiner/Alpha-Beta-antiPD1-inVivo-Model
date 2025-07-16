
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # enables 3D

# Your parameters
kon, koff = 131.4, 1.611    # mL·nmol⁻¹·h⁻¹, h⁻¹
kint, krel = 0.7602, 2.138  # h⁻¹

# Define grid
R = np.logspace(-3, -1, 300)
ci0 = np.logspace(np.log10(2.5e-3), np.log10(2.5e-1), 300)
Rg, ci0g = np.meshgrid(R, ci0)

# Solve quadratic for ci∞
a = kon
b = koff + kon * (Rg - ci0g)
c = -koff * ci0g
ci_inf = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

# Compute cb∞
cb_inf = (kon * ci_inf * Rg) / (
    (koff + kon * ci_inf) * (1 + kint / krel)
)

# Compute R_occ
R_occ = cb_inf / Rg * 100
R_occ[Rg == 0] = 1

# Map to colors
cmap = plt.get_cmap('plasma')
colors = cmap((R_occ - R_occ.min()) / (R_occ.max() - R_occ.min()))

# Plot
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Rg, ci0g, np.zeros_like(R_occ),
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
ax.set_xlabel('Receptor concentration (nmol per mL)')
ax.set_ylabel('Initial ligand concentration (nmol per mL)')
ax.set_zlim(0, 0.01)
ax.set_xlim([0, 0.1])
ax.set_ylim([0, 0.25])
ax.set_xticks([0,0.05,0.1])
ax.set_yticks([0,0.125,0.25])
ax.tick_params(axis='y', which='major', pad=10)
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 20

# Colorbar
mappable = plt.cm.ScalarMappable(cmap=cmap)
mappable.set_array(R_occ)
cbar = plt.colorbar(mappable, ax=ax, pad=0.001, shrink=0.6, aspect=10)
cbar.set_label('Receptor Occupancy (%)', labelpad=15)

plt.tight_layout()
plt.show()
