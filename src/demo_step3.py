"""Étape 3 : matrice interspectrale 3x3, cohérence et phase entre axes."""
import matplotlib.pyplot as plt
import numpy as np

from psd_csd import coherence, correlated_triaxial_noise, csd_matrix, n_averages

FS = 5000.0      # Hz
DURATION = 60.0  # s
SIGMA = 10.0     # g
RHO = 0.7        # corrélation x-y  ->  gamma² cible = RHO² = 0,49
NPERSEG = 2048

signals = correlated_triaxial_noise(SIGMA, RHO, FS, DURATION, seeds=(1, 2, 3))
f, s = csd_matrix(signals, FS, NPERSEG)
k = n_averages(signals.shape[1], NPERSEG)

pairs = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
print(f"K = {k} moyennes  (biais attendu sur des signaux indépendants : 1/K = {1 / k:.4f})")
for name, (i, j) in pairs.items():
    print(f"gamma²_{name} moyenne = {coherence(s, i, j)[1:-1].mean():.4f}")
print(f"gamma²_xy cible        = {RHO**2:.4f}")
print(f"Phase moyenne S_xy     = {np.degrees(np.angle(s[1:-1, 0, 1].mean())):.3f}° (attendu 0°)")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for name, (i, j) in pairs.items():
    ax1.plot(f, coherence(s, i, j), alpha=0.7, label=rf"$\gamma^2_{{{name}}}(f)$")
ax1.axhline(RHO**2, color="r", linestyle="--", label=rf"cible $\rho^2$ = {RHO**2:.2f}")
ax1.set_ylabel(r"Cohérence $\gamma^2$")
ax1.set_ylim(0, 1.05)
ax1.set_title("Étape 3 : cohérence et phase inter-axes")
ax1.legend()
ax1.grid(True, alpha=0.4)

ax2.plot(f, np.degrees(np.angle(s[:, 0, 1])), color="purple", alpha=0.7, label=r"Phase $\phi_{xy}(f)$")
ax2.axhline(0, color="k", linestyle="--")
ax2.set_xlabel("Fréquence (Hz)")
ax2.set_ylabel("Phase (degrés)")
ax2.set_ylim(-45, 45)
ax2.legend()
ax2.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()