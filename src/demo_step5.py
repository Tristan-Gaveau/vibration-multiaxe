"""Étape 5 : génération d'un signal triaxial depuis S(f) cible, puis ré-estimation (boucle fermée)."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

from generation import example_target, synthesize_triaxial
from psd_csd import csd_matrix

FS = 5000.0      # Hz
N = 300_000      # échantillons (60 s)
NPERSEG = 4096
SEED = 42

f_gen = np.fft.rfftfreq(N, 1.0 / FS)
f_est = np.fft.rfftfreq(NPERSEG, 1.0 / FS)
s_target = example_target(f_est)

x = synthesize_triaxial(example_target(f_gen), FS, N, SEED)
_, s_est = csd_matrix(x, FS, NPERSEG)

rms_target = np.sqrt(trapezoid(np.real(np.diagonal(s_target, axis1=1, axis2=2)), f_est, axis=0))
rms_meas = np.std(x, axis=1)
for name, rt, rm in zip("XYZ", rms_target, rms_meas):
    print(f"Axe {name} : RMS cible {rt:.2f} g | généré {rm:.2f} g | écart {abs(rm - rt) / rt * 100:.2f} %")

sum_est, sum_target = s_est[1:-1].sum(axis=0), s_target[1:-1].sum(axis=0)
for (i, j, name) in ((0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")):
    ratio = sum_est[i, j] / sum_target[i, j]
    print(f"S_{name} : module estimé/cible = {abs(ratio):.3f} | phase estimé - cible = {np.degrees(np.angle(ratio)):.2f}°")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for i, (name, color) in enumerate(zip("XYZ", ("tab:blue", "tab:orange", "tab:green"))):
    ax1.plot(f_est, s_target[:, i, i].real, "--", color=color, linewidth=2, label=rf"cible $S_{{{name}{name}}}$")
    ax1.plot(f_est, s_est[:, i, i].real, "-", color=color, alpha=0.6, label=rf"estimée $S_{{{name}{name}}}$")
ax1.set_xlabel("Fréquence (Hz)")
ax1.set_ylabel(r"PSD ($g^2$/Hz)")
ax1.set_title("Étape 5 : PSD diagonales")
ax1.legend(ncol=2, fontsize=8)
ax1.grid(True, alpha=0.4)

ax2.plot(f_est, np.degrees(np.angle(s_target[:, 0, 1])), "k--", linewidth=2, label=r"cible phase $S_{xy}$")
ax2.plot(f_est, np.degrees(np.angle(s_est[:, 0, 1])), "r-", alpha=0.6, label=r"estimée phase $S_{xy}$")
ax2.set_xlabel("Fréquence (Hz)")
ax2.set_ylabel("Phase (degrés)")
ax2.set_title(r"Étape 5 : phase de l'interspectre $S_{xy}$")
ax2.legend()
ax2.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()