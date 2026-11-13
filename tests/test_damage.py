"""Tests du FDS (étape 6) : cas analytiques bruit blanc, puis comparaison à une simulation temporelle.

Tolérances : analytique 1e-3 ; simulation calibrée sur 5 graines (voir commentaires).
"""
import numpy as np
import pytest
from scipy.signal import TransferFunction, find_peaks, lsim

from damage import G, fatigue_damage, fds, relative_displacement_rms
from psd_csd import generate_white_noise, white_noise_psd

FS = 5000.0
SIGMA = 10.0
Q = 10.0
F = np.linspace(0.0, FS / 2.0, 2501)
PSD = np.full_like(F, white_noise_psd(SIGMA, FS))   # 0,04 g²/Hz sur 0-2500 Hz


@pytest.mark.parametrize("fn", [50.0, 200.0, 400.0])
def test_z_rms_matches_analytic_white_noise(fn):
    # Bruit blanc : z_rms = sqrt(pi/2 . fn . Q . PSD) / wn² (m), exact. Écart mesuré < 0,01 %.
    wn = 2.0 * np.pi * fn
    expected_mm = 1e3 * np.sqrt(np.pi / 2.0 * fn * Q * PSD[0] * G**2) / wn**2
    assert np.isclose(relative_displacement_rms(F, PSD, fn, Q), expected_mm, rtol=1e-3)


def test_b2_damage_formula():
    # Gamma(2) = 1 : D = fn . T . 2 . z_rms²
    assert np.isclose(fatigue_damage(0.07, 200.0, 60.0, 2), 200.0 * 60.0 * 2.0 * 0.07**2, rtol=1e-12)


@pytest.mark.parametrize("b", [2, 4, 6, 8])
def test_fds_slope_for_white_noise(b):
    # z_rms² ~ fn^-3  ->  D ~ fn^(1 - 3b/2). Pente mesurée à 1e-4 près.
    fn_values = np.geomspace(50.0, 400.0, 12)
    slope = np.polyfit(np.log(fn_values), np.log(fds(F, PSD, fn_values, Q, 60.0, b)), 1)[0]
    assert np.isclose(slope, 1.0 - 1.5 * b, atol=0.01)


@pytest.fixture(scope="module")
def simulation():
    """Simulation temporelle (300 s) du déplacement relatif, en mm."""
    fn, duration = 200.0, 300.0
    a = generate_white_noise(SIGMA, FS, duration, seed=0) * G           # m/s²
    zeta, wn = 1.0 / (2.0 * Q), 2.0 * np.pi * fn
    system = TransferFunction([1.0], [1.0, 2.0 * zeta * wn, wn**2])    # z / a_base
    _, z, _ = lsim(system, U=a, T=np.arange(len(a)) / FS)
    return fn, duration, 1e3 * z


def test_z_rms_matches_time_simulation(simulation):
    # Écart mesuré 0,4 à 0,6 % sur 5 graines (interpolation linéaire de lsim) -> tolérance 2 %.
    fn, _, z = simulation
    assert np.isclose(z.std(), relative_displacement_rms(F, PSD, fn, Q), rtol=0.02)


@pytest.mark.parametrize("b, tol", [(2, 0.03), (4, 0.08)])
def test_formula_matches_peak_counting(simulation, b, tol):
    # Comptage de pics positifs de z(t), amplitude A_i : D_sim = sum(A_i^b).
    # Écarts mesurés sur 5 graines : -1,5 % (b=2) et -3,7 % (b=4, de -2,5 à -4,6 %).
    # La formule (pics de Rayleigh, bande étroite) est légèrement conservatrice : D_sim <= D_formule.
    fn, duration, z = simulation
    peaks = z[find_peaks(z)[0]]
    d_sim = np.sum(peaks[peaks > 0.0] ** b)
    d_formula = fatigue_damage(relative_displacement_rms(F, PSD, fn, Q), fn, duration, b)
    assert abs(d_sim / d_formula - 1.0) < tol
    assert d_sim <= d_formula