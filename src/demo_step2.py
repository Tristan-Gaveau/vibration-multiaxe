"""Étape 2 : réponse d'un oscillateur 1 DDL à un bruit blanc, comparée à l'équation de Miles."""
import matplotlib.pyplot as plt

from oscillator import miles_grms, oscillator_response, rms
from psd_csd import band_mean_psd, estimate_psd, generate_white_noise, white_noise_psd

FS = 5000.0      # Hz
DURATION = 60.0  # s
SIGMA = 10.0     # g
SEED = 42
FN = 200.0       # Hz
Q = 10.0         # -
NPERSEG = 2048

a_base = generate_white_noise(SIGMA, FS, DURATION, SEED)
a_mass = oscillator_response(a_base, FN, Q, FS)

f_in, psd_in = estimate_psd(a_base, FS, NPERSEG)
psd_ref_theory = white_noise_psd(SIGMA, FS)                       # vérité (synthétique)
psd_ref_band = band_mean_psd(f_in, psd_in, FN, FN / Q)            # estimation moyennée (cas réel)

grms_sim = rms(a_mass)
for label, psd_ref in (("PSD théorique", psd_ref_theory), ("PSD moyennée sur fn ± fn/Q", psd_ref_band)):
    grms_miles = miles_grms(FN, Q, psd_ref)
    err = abs(grms_sim - grms_miles) / grms_miles * 100
    print(f"Référence = {label:28s}: PSD={psd_ref:.4f} g²/Hz, Miles={grms_miles:.2f} g, "
          f"simulé={grms_sim:.2f} g, écart={err:.2f} %")

f_out, psd_out = estimate_psd(a_mass, FS, NPERSEG)
plt.figure(figsize=(9, 4))
plt.semilogy(f_in, psd_in, label="PSD excitation (base)", alpha=0.7)
plt.semilogy(f_out, psd_out, label=rf"PSD réponse ($f_n$={FN:.0f} Hz, Q={Q:.0f})", color="orange")
plt.axvline(FN, color="r", linestyle="--", label=rf"$f_n$ = {FN:.0f} Hz")
plt.xlabel("Fréquence (Hz)")
plt.ylabel(r"PSD ($g^2$/Hz)")
plt.title("Étape 2 : amplification par résonance (équation de Miles)")
plt.xlim(0, 1000)
plt.legend()
plt.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.show()