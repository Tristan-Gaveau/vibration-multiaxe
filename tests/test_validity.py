"""Tests de validité, d'enveloppe et de correction d'une matrice interspectrale (étape 4)."""
import numpy as np

from psd_csd import correlated_triaxial_noise, csd_matrix
from validity import (
    build_matrix,
    clip_negative_eigenvalues,
    eigenvalues,
    envelope,
    is_valid,
)

LEVEL = 0.04    # g²/Hz
R = 0.9         # corrélation de chaque cas A et B
# A : seuls x et y corrélés. B : seuls y et z corrélés.
S_A = build_matrix(LEVEL, R, 0.0, 0.0)
S_B = build_matrix(LEVEL, 0.0, 0.0, R)


def test_a_and_b_are_valid():
    assert is_valid(S_A) and is_valid(S_B)


def test_envelope_is_invalid_with_analytic_eigenvalue():
    env = envelope(S_A, S_B)
    assert not is_valid(env)
    # Matrice de corrélation [[1,R,0],[R,1,R],[0,R,1]] : valeurs propres 1, 1 +- R*sqrt(2).
    expected_min = LEVEL * (1.0 - R * np.sqrt(2.0))
    assert np.isclose(eigenvalues(env)[0], expected_min, rtol=1e-9)


def test_envelope_stays_hermitian():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    b = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    a, b = a + a.conj().T, b + b.conj().T
    env = envelope(a, b)
    assert np.allclose(env, env.conj().T, atol=1e-12)


def test_clipping_makes_matrix_valid():
    env = envelope(S_A, S_B)
    for keep in (False, True):
        assert is_valid(clip_negative_eigenvalues(env, keep_diagonal=keep))


def test_keep_diagonal_preserves_psd_levels():
    fixed = clip_negative_eigenvalues(envelope(S_A, S_B), keep_diagonal=True)
    assert np.allclose(np.diag(fixed), LEVEL, rtol=1e-12)


def test_clipping_leaves_a_valid_matrix_unchanged():
    assert np.allclose(clip_negative_eigenvalues(S_A, keep_diagonal=False), S_A, atol=1e-15)


def test_invalid_band_matches_analytic_threshold():
    # Corrélation r(f) en cloche autour de 1000 Hz ; l'enveloppe est invalide ssi r > 1/sqrt(2).
    f = np.linspace(10.0, 2000.0, 500)
    r = R * np.exp(-(((f - 1000.0) / 300.0) ** 2))
    env = envelope(build_matrix(LEVEL, r, 0.0, 0.0), build_matrix(LEVEL, 0.0, 0.0, r))
    assert np.array_equal(~is_valid(env), r > 1.0 / np.sqrt(2.0))


def test_welch_estimate_is_valid():
    # Une matrice estimée avec K = 291 moyennes est valide : l'invalidité vient de l'enveloppe.
    signals = correlated_triaxial_noise(10.0, 0.7, 5000.0, 60.0)
    _, s = csd_matrix(signals, 5000.0, 2048)
    assert np.all(is_valid(s))