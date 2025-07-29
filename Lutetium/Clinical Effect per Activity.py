from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from mpl_toolkits.mplot3d import Axes3D  # enables 3D

# Your parameters
kon, koff = 131.4, 1.611    # mL·nmol⁻¹·h⁻¹, h⁻¹
kint, krel = 0.7602, 2.138  # h⁻¹
Si_per_N = 5.25e-8 / 100000 # Gy Bq-1 h-1 cell-1
Sb, Sc = 0.374, 0.713       # Gy Bq-1 h-1
alpha = 0.367               # Gy-1
la = 0.004323               # h-1

# Define grid
R = np.logspace(np.log10(1e-3), np.log10(1e-0), 300)
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
E = (Sig + (Sb * cb_inf + Sc * cc_inf) / (Ng * ci0)) * alpha / la

# Map to colors
cmap = plt.get_cmap('BrBG')
colors = cmap((E - E.min()) / (E.max() - E.min()))

# Plot
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(np.log10(Ng), Ag, np.zeros_like(E), rstride=1, cstride=1,
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
ax.set_xlim([np.log10(1.8e4), np.log10(2e7)])
ax.set_ylim([0, 10])
ax.set_xticks(np.log10([2e4, 2e5,2e6,2e7]))
ax.set_xticklabels(['2e4', '2e5', '2e6', '2e7'])
ax.set_yticks([0,5,10])
ax.tick_params(axis='y', which='major', pad=10)
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 20

# Your levels and discrete grayscale colors
custom_levels = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

ax.contour(np.log10(Ng), Ag, E, zdir='z', offset=0, levels=custom_levels, colors='#002147', linewidths=1, linestyles=':', alpha=0.3)
ax.text2D(0.812, 0.83, '●', transform=ax.transAxes, ha='center', fontsize=5, color='black')
ax.text2D(0.799, 0.83, '■', transform=ax.transAxes, ha='center', fontsize=5, color='black')
ax.text2D(0.225, 0.83, '■', transform=ax.transAxes, ha='center', fontsize=5, color='black')
ax.text2D(0.786, 0.83, '▲', transform=ax.transAxes, ha='center', fontsize=5, color='black')
ax.text2D(0.245, 0.83, '▲', transform=ax.transAxes, ha='center', fontsize=5, color='black')
ax.text2D(0.770, 0.83, '▼', transform=ax.transAxes, ha='center', fontsize=5, color='black')
ax.text2D(0.269, 0.83, '▼', transform=ax.transAxes, ha='center', fontsize=5, color='black')
ax.text2D(0.749, 0.83, '◀', transform=ax.transAxes, ha='center', fontsize=5, color='black')
ax.text2D(0.299, 0.83, '◀', transform=ax.transAxes, ha='center', fontsize=5, color='black')
ax.text2D(0.723, 0.83, '▶', transform=ax.transAxes, ha='center', fontsize=5, color='black')
ax.text2D(0.339, 0.83, '▶', transform=ax.transAxes, ha='center', fontsize=5, color='black')
ax.text2D(0.679, 0.83, '◆', transform=ax.transAxes, ha='center', fontsize=5, color='black')
ax.text2D(0.393, 0.83, '◆', transform=ax.transAxes, ha='center', fontsize=5, color='black')
ax.text2D(0.559, 0.83, '★', transform=ax.transAxes, ha='center', fontsize=5, color='black')
ax.text2D(0.536, 0.83, '★', transform=ax.transAxes, ha='center', fontsize=5, color='black')

ax.text2D(0.97, 0.745, '●', transform=ax.transAxes, ha='center', fontsize=15, color='black')
ax.text2D(0.97, 0.674, '■', transform=ax.transAxes, ha='center', fontsize=15, color='black')
ax.text2D(0.97, 0.603, '▲', transform=ax.transAxes, ha='center', fontsize=15, color='black')
ax.text2D(0.97, 0.532, '▼', transform=ax.transAxes, ha='center', fontsize=15, color='black')
ax.text2D(0.97, 0.461, '◀', transform=ax.transAxes, ha='center', fontsize=15, color='black')
ax.text2D(0.97, 0.390, '▶', transform=ax.transAxes, ha='center', fontsize=15, color='black')
ax.text2D(0.97, 0.319, '◆', transform=ax.transAxes, ha='center', fontsize=15, color='black')
ax.text2D(0.97, 0.248, '★', transform=ax.transAxes, ha='center', fontsize=15, color='black')

# Create corresponding labels
legend_labels = [f"{level:.4f}" for level in custom_levels]

# Colorbar
mappable = plt.cm.ScalarMappable(cmap=cmap)
mappable.set_array(E)
cbar = plt.colorbar(mappable, ax=ax, pad=0.001, shrink=0.6, aspect=10)
cbar.set_label('Clinical effect per initial activity (per MBq)', labelpad=15)

plt.tight_layout()
plt.show()
