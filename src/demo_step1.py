"""Étape 1 : PSD d'un bruit blanc, validation (Parseval, moyenne, variance de l'estimation)."""
import matplotlib.pyplot as plt
import numpy as np

from psd_csd import (
    estimate_psd,
    generate_white_noise,
    n_averages,
    parseval_relative_error,
    white_noise_psd,
)

FS = 5000.0      # Hz
DURATION = 60.0  # s
SIGMA = 10.0     # g
SEED = 42

x = generate_white_noise(SIGMA, FS, DURATION, SEED)
psd_theory = white_noise_psd(SIGMA, FS)

plt.figure(figsize=(10, 4.5))
for nperseg in (4096, 2048):
    f, psd = estimate_psd(x, FS, nperseg)
    k = n_averages(len(x), nperseg)
    inner = psd[1:-1]  # on exclut 0 Hz et Nyquist (non doublés en unilatéral)
    rel_std = np.std(inner) / np.mean(inner)
    print(f"nperseg={nperseg}: df={FS / nperseg:.2f} Hz, K={k}")
    print(f"  Parseval        : {parseval_relative_error(x, f, psd) * 100:.3f} %")
    print(f"  Moyenne PSD     : {np.mean(inner):.4f} g²/Hz (théorie {psd_theory:.4f})")
    print(f"  Écart-type rel. : {rel_std * 100:.2f} % (1/sqrt(K) = {100 / np.sqrt(k):.2f} %)")
    plt.plot(f, psd, alpha=0.75, label=rf"nperseg={nperseg} ($\Delta f$={FS / nperseg:.2f} Hz)")

plt.axhline(psd_theory, color="r", linestyle="--", label=f"PSD théorique ({psd_theory:.2f} $g^2$/Hz)")
plt.xlabel("Fréquence (Hz)")
plt.ylabel(r"PSD ($g^2$/Hz)")
plt.title(rf"Étape 1 : compromis résolution / variance ($\sigma$ = {SIGMA} g, $f_s$ = {FS:.0f} Hz)")
plt.ylim(0, psd_theory * 1.6)
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()