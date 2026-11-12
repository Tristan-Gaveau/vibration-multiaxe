"""Oscillateur à 1 degré de liberté excité à la base, et équation de Miles.

Grandeur de sortie : accélération ABSOLUE de la masse (celle que donne Miles).
"""
import numpy as np
from scipy.signal import TransferFunction, lsim


def oscillator_response(a_base, fn, Q, fs):
    """Accélération absolue (g) de la masse pour une accélération de base `a_base` (g).

    fn : fréquence propre (Hz), Q : facteur de qualité (-), fs : fréquence d'échantillonnage (Hz).
    Transmissibilité : T(s) = (2*zeta*wn*s + wn²) / (s² + 2*zeta*wn*s + wn²), zeta = 1/(2Q).
    """
    zeta = 1.0 / (2.0 * Q)
    wn = 2.0 * np.pi * fn
    system = TransferFunction([2.0 * zeta * wn, wn**2], [1.0, 2.0 * zeta * wn, wn**2])
    t = np.arange(len(a_base)) / fs  # évite le piège np.arange(0, T, dt) en flottants
    _, a_mass, _ = lsim(system, U=a_base, T=t)
    return a_mass


def miles_grms(fn, Q, psd_fn):
    """Équation de Miles : G_rms = sqrt(pi/2 * fn * Q * PSD(fn)).

    Valable pour une PSD d'entrée à peu près plate autour de fn (bande ~ fn/Q).
    """
    return np.sqrt(np.pi / 2.0 * fn * Q * psd_fn)


def rms(x):
    """Valeur efficace sqrt(moyenne(x²))."""
    return np.sqrt(np.mean(np.square(x)))