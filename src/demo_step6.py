"""Étape 6 : FDS de deux spécifications de même RMS, et additivité pour b = 2."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

from damage import fatigue_damage, fds, relative_displacement_rms

Q = 10.0           # facteur de qualité de l'oscillateur
DURATION = 3600.0  # s (relatif : seuls les rapports ont un sens, C = 1)
F = np.linspace(0.0, 2500.0, 2501)
FN = np.geomspace(30.0, 1000.0, 60)

spec_a = np.where((F >= 20.0) & (F <= 2000.0), 0.04, 0.0)          # plateau 20-2000 Hz
shape = np.exp(-(((F - 600.0) / 150.0) ** 2))
spec_b = shape * trapezoid(spec_a, F) / trapezoid(shape, F)        # cloche à 600 Hz, même énergie

print(f"RMS A = {np.sqrt(trapezoid(spec_a, F)):.2f} g | RMS B = {np.sqrt(trapezoid(spec_b, F)):.2f} g (identiques)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
for ax, b in zip(axes, (4, 8)):
    d_a, d_b = fds(F, spec_a, FN, Q, DURATION, b), fds(F, spec_b, FN, Q, DURATION, b)
    ax.loglog(FN, d_a, label="A : plateau 20-2000 Hz")
    ax.loglog(FN, d_b, label="B : cloche à 600 Hz")
    ax.set_title(f"FDS, b = {b} (dommage relatif, C = 1)")
    ax.set_xlabel(r"Fréquence propre $f_n$ (Hz)")
    ax.set_ylabel("D (unité arbitraire)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)
    for fn in (100.0, 600.0):
        ratio = fds(F, spec_b, [fn], Q, DURATION, b)[0] / fds(F, spec_a, [fn], Q, DURATION, b)[0]
        print(f"b={b}, fn={fn:.0f} Hz : D(B)/D(A) = {ratio:.2f}")
plt.tight_layout()
plt.show()

# Additivité : D(P1 + P2) contre D(P1) + D(P2)
p1 = np.where((F > 100.0) & (F < 400.0), 0.05, 0.0)
p2 = np.where((F > 150.0) & (F < 1500.0), 0.03, 0.0)
for b in (2, 4):
    def damage(p):
        return fatigue_damage(relative_displacement_rms(F, p, 200.0, Q), 200.0, DURATION, b)
    print(f"b={b} : D(P1+P2) / (D(P1)+D(P2)) = {damage(p1 + p2) / (damage(p1) + damage(p2)):.6f}")