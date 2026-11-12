"""Tests de la génération de signaux triaxiaux (étape 5) : boucle fermée génération -> estimation.

Tolérances calibrées sur 20 à 30 graines (voir commentaires) : environ 2 x le maximum observé.
"""
import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import kurtosis

from generation import example_target, synthesize_triaxial
from psd_csd import csd_matrix
from validity import is_valid

FS = 5000.0       # Hz
N = 300_000       # échantillons (60 s)
NPERSEG = 4096
F_EST = np.fft.rfftfreq(NPERSEG, 1.0 / FS)   # grille de l'estimation
F_GEN = np.fft.rfftfreq(N, 1.0 / FS)         # grille de la génération


def _generate(seed=0):
    return synthesize_triaxial(example_target(F_GEN), FS, N, seed)


def test_target_is_valid():
    assert np.all(is_valid(example_target(F_EST)))


def test_rms_per_axis_matches_target():
    # Erreur mesurée : écart-type 0,17 % sur 20 graines -> tolérance 1 %.
    s_target = example_target(F_EST)
    rms_target = np.sqrt(trapezoid(np.real(np.diagonal(s_target, axis1=1, axis2=2)), F_EST, axis=0))
    assert np.allclose(np.std(_generate(), axis=1), rms_target, rtol=0.01)


def test_closed_loop_all_nine_terms():
    # Erreur normalisée |sum(S_est) - sum(S_cible)| / sqrt(S_ii S_jj) : max 1,4 % sur 30 graines -> 3 %.
    s_target = example_target(F_EST)
    _, s_est = csd_matrix(_generate(), FS, NPERSEG)
    sum_est, sum_target = s_est[1:-1].sum(axis=0), s_target[1:-1].sum(axis=0)
    diag = np.real(np.diag(sum_target))
    error = np.abs(sum_est - sum_target) / np.sqrt(np.outer(diag, diag))
    assert error.max() < 0.03


def test_cross_term_phase_is_not_conjugated():
    # Cible complexe : phase de S_xy et S_yz. Erreur de phase max 0,011 rad sur 30 graines -> 0,03 rad.
    s_target = example_target(F_EST)
    _, s_est = csd_matrix(_generate(), FS, NPERSEG)
    for i, j in ((0, 1), (1, 2)):
        ratio = s_est[1:-1, i, j].sum() / s_target[1:-1, i, j].sum()
        assert abs(np.angle(ratio)) < 0.03


def test_pure_imaginary_coupling_has_plus_90_degrees():
    # Test déterministe de la convention : S_xy = +j.c  ->  phase estimée = +90 degrés.
    s = np.zeros((len(F_GEN), 2, 2), dtype=complex)
    s[:, 0, 0] = s[:, 1, 1] = 0.1
    s[:, 0, 1], s[:, 1, 0] = 0.05j, -0.05j
    _, s_est = csd_matrix(synthesize_triaxial(s, FS, N, 0), FS, NPERSEG)
    assert abs(np.degrees(np.angle(s_est[1:-1, 0, 1].mean())) - 90.0) < 1.0


def test_singular_matrix_is_accepted():
    # y = x exactement (matrice de rang 1) : Cholesky échouerait, la décomposition en valeurs propres non.
    s = np.full((len(F_GEN), 2, 2), 0.04, dtype=complex)
    x = synthesize_triaxial(s, FS, N, 1)
    assert np.allclose(x[0], x[1], atol=1e-9)


def test_signal_is_gaussian():
    # Kurtosis de Pearson = 3 pour une loi gaussienne. Écart-type théorique ~ sqrt(24/N) = 0,009 -> 0,05.
    assert abs(kurtosis(_generate()[0], fisher=False) - 3.0) < 0.05


def test_wrong_grid_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        synthesize_triaxial(example_target(F_EST), FS, N, 0)