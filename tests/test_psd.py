"""Tests de l'estimation de PSD (étape 1).

Tolérances : 4 x l'erreur statistique attendue (voir commentaires), pas de valeur arbitraire.
"""
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
NPERSEG = 2048


def test_white_noise_psd_formula():
    # variance = PSD x bande : 100 g² = PSD x 2500 Hz  ->  0,04 g²/Hz
    assert np.isclose(white_noise_psd(SIGMA, FS), 0.04, rtol=1e-12)


def test_parseval_random_signal():
    # Erreur statistique de la variance : sqrt(2/N) ~ 0,26 % (N = 300 000) -> tolérance 1 %.
    x = generate_white_noise(SIGMA, FS, DURATION, SEED)
    f, psd = estimate_psd(x, FS, NPERSEG)
    assert parseval_relative_error(x, f, psd) < 0.01


def test_mean_psd_level():
    # Même erreur statistique ~ 0,26 % -> tolérance 1 %.
    x = generate_white_noise(SIGMA, FS, DURATION, SEED)
    _, psd = estimate_psd(x, FS, NPERSEG)
    assert np.isclose(psd[1:-1].mean(), white_noise_psd(SIGMA, FS), rtol=0.01)


def test_relative_std_matches_one_over_sqrt_k():
    # L'écart-type relatif de la PSD estimée doit valoir ~ 1/sqrt(K). Tolérance 15 % (relative).
    x = generate_white_noise(SIGMA, FS, DURATION, SEED)
    _, psd = estimate_psd(x, FS, NPERSEG)
    inner = psd[1:-1]
    measured = np.std(inner) / np.mean(inner)
    expected = 1.0 / np.sqrt(n_averages(len(x), NPERSEG))
    assert np.isclose(measured, expected, rtol=0.15)


def test_parseval_deterministic_sine():
    # Test déterministe : sinusoïde d'amplitude A, variance A²/2, fréquence sur un bin de la FFT.
    amplitude = 3.0  # g
    f0 = 128 * FS / NPERSEG  # 312,5 Hz, centre d'un bin
    t = np.arange(int(FS * DURATION)) / FS
    x = amplitude * np.sin(2.0 * np.pi * f0 * t)
    f, psd = estimate_psd(x, FS, NPERSEG)
    assert np.isclose(np.var(x), amplitude**2 / 2.0, rtol=1e-6)
    assert parseval_relative_error(x, f, psd) < 1e-3