"""Additivité du dommage (étape 6) : exacte pour b = 2, sur-additive pour b > 2."""
import numpy as np

from damage import fatigue_damage, relative_displacement_rms

FN, Q, DURATION = 200.0, 10.0, 60.0
F = np.linspace(0.0, 2500.0, 2501)


def _damage(psd, b):
    return fatigue_damage(relative_displacement_rms(F, psd, FN, Q), FN, DURATION, b)


def test_b2_is_additive_for_disjoint_spectra():
    # b = 2 : D proportionnel à la variance de z, donc à l'intégrale de la PSD (linéaire). Exact.
    p1 = np.where((F > 100) & (F < 400), 0.05, 0.0)
    p2 = np.where((F > 500) & (F < 1500), 0.03, 0.0)
    assert np.isclose(_damage(p1 + p2, 2), _damage(p1, 2) + _damage(p2, 2), rtol=1e-9)


def test_b2_is_additive_for_overlapping_spectra():
    p1 = np.where((F > 100) & (F < 400), 0.05, 0.0)
    p2 = np.where((F > 150) & (F < 1500), 0.03, 0.0)
    assert np.isclose(_damage(p1 + p2, 2), _damage(p1, 2) + _damage(p2, 2), rtol=1e-9)


def test_damage_scales_as_psd_to_the_b_over_two():
    # z_rms ~ sqrt(PSD)  ->  D(k.P) = k^(b/2) . D(P). Valable pour tout b, exact.
    psd = np.where((F > 20) & (F < 2000), 0.04, 0.0)
    for b in (2, 4, 6, 8):
        assert np.isclose(_damage(2.0 * psd, b), 2.0 ** (b / 2.0) * _damage(psd, b), rtol=1e-9)


def test_b_greater_than_2_is_superadditive():
    # Deux spectres identiques superposés : D(2P) = 2^(b/2) D(P) > 2 D(P) dès que b > 2.
    psd = np.where((F > 20) & (F < 2000), 0.04, 0.0)
    for b in (4, 6, 8):
        assert _damage(2.0 * psd, b) > 2.0 * _damage(psd, b) * (1.0 + 1e-9)