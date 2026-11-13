"""Dommage de fatigue d'un oscillateur 1 DDL excité à la base : FDS (Fatigue Damage Spectrum).

Modèle : contrainte proportionnelle au déplacement relatif z de l'oscillateur, loi de Basquin
N . sigma^b = C, règle de Miner D = sum(n_i / N_i), signal gaussien (pics de Rayleigh) :

    D(fn) = fn . T . (sqrt(2) . z_rms)^b . Gamma(1 + b/2) / C

- fn . T : nombre de cycles (bande étroite : un cycle par période propre) ;
- E[A^b] = (sqrt(2) . z_rms)^b . Gamma(1 + b/2) : moment d'ordre b d'une amplitude de Rayleigh.

Avec C = 1 et z_rms en mm, D est un dommage RELATIF (unité arbitraire) : seuls les rapports
entre deux spécifications ont un sens.
"""
import numpy as np
from math import gamma
from scipy.integrate import trapezoid

G = 9.80665  # m/s² par g


def relative_displacement_rms(f, psd, fn, Q):
    """z_rms (mm) du déplacement relatif d'un oscillateur (fn en Hz, Q sans unité).

    f : grille de fréquences (Hz) ; psd : PSD unilatérale de l'accélération de base (g²/Hz) sur f.
    La PSD est supposée nulle hors de [f[0], f[-1]]. L'intégrale est faite sur une grille fine
    (fn / (20 Q) : 20 points dans la largeur à mi-puissance), pour bien résoudre la résonance.
    z(f) = 1 / (wn² - w² + j.(wn/Q).w) x accélération de base (m/s²).
    """
    f = np.asarray(f, dtype=float)
    f_fine = np.arange(0.0, f[-1] + 1e-9, fn / (20.0 * Q))
    psd_fine = np.interp(f_fine, f, np.asarray(psd, dtype=float), left=0.0, right=0.0)
    w, wn = 2.0 * np.pi * f_fine, 2.0 * np.pi * fn
    h = 1.0 / (wn**2 - w**2 + 1j * (wn / Q) * w)
    variance_m2 = trapezoid(np.abs(h) ** 2 * psd_fine * G**2, f_fine)
    return 1e3 * np.sqrt(variance_m2)


def fatigue_damage(z_rms_mm, fn, duration, b, C=1.0):
    """Dommage relatif D = fn . T . (sqrt(2) z_rms)^b . Gamma(1 + b/2) / C."""
    return fn * duration * (np.sqrt(2.0) * z_rms_mm) ** b * gamma(1.0 + b / 2.0) / C


def fds(f, psd, fn_values, Q, duration, b, C=1.0):
    """Spectre de dommage de fatigue : D en fonction de la fréquence propre, pour une PSD donnée."""
    fn_values = np.asarray(fn_values, dtype=float)
    z = np.array([relative_displacement_rms(f, psd, fn, Q) for fn in fn_values])
    return fatigue_damage(z, fn_values, duration, b, C)