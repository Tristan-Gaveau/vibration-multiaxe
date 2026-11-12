"""Tests de la matrice interspectrale 3x3 et de la cohérence (étape 3).

Tolérances calibrées sur 30 graines (voir commentaires) : 4 x la dispersion mesurée + biais connu.
"""
import numpy as np

from psd_csd import (
    coherence,
    correlated_triaxial_noise,
    csd_matrix,
    generate_white_noise,
    n_averages,
)

FS = 5000.0      # Hz
DURATION = 60.0  # s
SIGMA = 10.0     # g
RHO = 0.7        # coefficient de corrélation x-y  ->  gamma² = RHO² = 0,49
NPERSEG = 2048



def test_hermitian_symmetry():
    signals = correlated_triaxial_noise(SIGMA, RHO, FS, DURATION)
    _, s = csd_matrix(signals, FS, NPERSEG)
    # S_ji = conj(S_ij). Les 9 termes sont calculés séparément : écart de l'ordre de 1e-18.
    assert np.allclose(s, np.conj(np.swapaxes(s, 1, 2)), rtol=0.0, atol=1e-12)


def test_diagonal_real_positive():
    signals = correlated_triaxial_noise(SIGMA, RHO, FS, DURATION)
    _, s = csd_matrix(signals, FS, NPERSEG)
    diag = np.stack([s[:, i, i] for i in range(3)])
    # Partie imaginaire = bruit d'arrondi (~1e-20) : on compare à la partie réelle, pas à 0 exactement.
    assert np.all(np.abs(diag.imag) <= 1e-12 * diag.real)
    assert np.all(diag.real > 0.0)


def test_coherence_correlated_pair():
    # Sur 30 graines : moyenne de gamma²_xy = 0,491, dispersion entre graines 0,0015,
    # biais ~ +0,001. Tolérance : 4 x 0,0015 + 0,001 -> 0,01.
    signals = correlated_triaxial_noise(SIGMA, RHO, FS, DURATION)
    _, s = csd_matrix(signals, FS, NPERSEG)
    assert np.isclose(coherence(s, 0, 1)[1:-1].mean(), RHO**2, atol=0.01)


def test_coherence_independent_pairs_close_to_one_over_k():
    # Signaux indépendants : gamma² estimée ~ 1/K. Tolérance 3/K (mesuré : 1,1/K).
    signals = correlated_triaxial_noise(SIGMA, RHO, FS, DURATION)
    _, s = csd_matrix(signals, FS, NPERSEG)
    k = n_averages(signals.shape[1], NPERSEG)
    assert coherence(s, 0, 2)[1:-1].mean() < 3.0 / k
    assert coherence(s, 1, 2)[1:-1].mean() < 3.0 / k


def test_phase_of_correlated_pair_is_zero():
    # Corrélation réelle -> phase moyenne nulle. Mesuré : dispersion 0,002 rad -> tolérance 0,01 rad.
    signals = correlated_triaxial_noise(SIGMA, RHO, FS, DURATION)
    _, s = csd_matrix(signals, FS, NPERSEG)
    assert abs(np.angle(s[1:-1, 0, 1].mean())) < 0.01


def test_coherence_is_one_with_a_single_segment():
    # Le piège : sans moyenne (K = 1), gamma² = 1 même pour deux signaux indépendants.
    x = generate_white_noise(SIGMA, FS, DURATION, 1)
    z = generate_white_noise(SIGMA, FS, DURATION, 3)
    _, s = csd_matrix(np.vstack([x, z]), FS, nperseg=len(x))
    assert np.allclose(coherence(s, 0, 1), 1.0, atol=1e-9)


def test_phase_convention_with_delay():
    # y = x retardé de 5 échantillons -> phase de S_xy = -2*pi*f*tau (convention conj(X).Y).
    x = generate_white_noise(SIGMA, FS, DURATION, 7)
    delay = 5
    y = np.roll(x, delay)
    f, s = csd_matrix(np.vstack([x, y]), FS, NPERSEG)
    expected = -2.0 * np.pi * f[10:500] * delay / FS
    # Comparaison sur le cercle unité (évite le saut de phase en +-pi).
    assert np.allclose(np.exp(1j * np.angle(s[10:500, 0, 1])), np.exp(1j * expected), atol=0.05)