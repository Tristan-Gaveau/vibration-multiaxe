"""Étape 4 : une enveloppe terme à terme de deux matrices valides peut être non physique."""
import matplotlib.pyplot as plt
import numpy as np

from validity import build_matrix, clip_negative_eigenvalues, eigenvalues, envelope, is_valid

LEVEL = 0.04        # g²/Hz sur chaque axe
R_MAX = 0.9         # corrélation maximale
F_CENTER = 1000.0   # Hz
F_WIDTH = 300.0     # Hz

f = np.linspace(10.0, 2000.0, 500)
r = R_MAX * np.exp(-(((f - F_CENTER) / F_WIDTH) ** 2))

s_a = build_matrix(LEVEL, r, 0.0, 0.0)   # cas A : x et y corrélés
s_b = build_matrix(LEVEL, 0.0, 0.0, r)   # cas B : y et z corrélés
s_env = envelope(s_a, s_b)
s_fix = clip_negative_eigenvalues(s_env, keep_diagonal=False)
s_fix_diag = clip_negative_eigenvalues(s_env, keep_diagonal=True)

invalid = ~is_valid(s_env)
print(f"A valide partout : {bool(is_valid(s_a).all())} | B valide partout : {bool(is_valid(s_b).all())}")
print(f"Enveloppe invalide à {invalid.sum()} fréquences sur {len(f)} "
      f"({f[invalid].min():.0f}-{f[invalid].max():.0f} Hz)")
print(f"lambda_min minimale de l'enveloppe : {eigenvalues(s_env)[:, 0].min():.4f} g²/Hz")
print(f"Corrigées valides : écrêtage seul {bool(is_valid(s_fix).all())}, "
      f"écrêtage + diagonale {bool(is_valid(s_fix_diag).all())}")

k = np.argmax(r)  # fréquence de corrélation maximale
for name, m in (("enveloppe", s_env), ("écrêtage seul", s_fix), ("écrêtage + diagonale", s_fix_diag)):
    corr_xy = m[k, 0, 1] / np.sqrt(m[k, 0, 0] * m[k, 1, 1])
    corr_xz = m[k, 0, 2] / np.sqrt(m[k, 0, 0] * m[k, 2, 2])
    print(f"  {name:22s}: PSD_x={m[k, 0, 0]:.4f}, corr_xy={corr_xy:.3f}, corr_xz={corr_xz:.3f}")

plt.figure(figsize=(9, 4))
plt.plot(f, eigenvalues(s_env)[:, 0], "r-", label=r"$\lambda_{min}$ enveloppe brute")
plt.plot(f, eigenvalues(s_fix)[:, 0], "g--", linewidth=2, label=r"$\lambda_{min}$ après écrêtage")
plt.axhline(0, color="k", linestyle=":")
plt.fill_between(f, eigenvalues(s_env)[:, 0], 0, where=invalid, color="red", alpha=0.2,
                 label=r"zone non physique ($\lambda < 0$)")
plt.xlabel("Fréquence (Hz)")
plt.ylabel(r"$\lambda_{min}$ ($g^2$/Hz)")
plt.title("Étape 4 : détection et correction d'une matrice interspectrale non physique")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()