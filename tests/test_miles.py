"""Tests de la réponse d'un oscillateur 1 DDL et de l'équation de Miles (étape 2).

Tolérance 5 % : erreur statistique du RMS ~ 1,6 % + biais de Miles ~ 1 % (à vérifier),
soit environ 2 x l'incertitude totale. Référence = PSD théorique connue (cas synthétique).
"""
import numpy as np

from oscillator import miles_grms, oscillator_response, rms
from psd_csd import generate_white_noise, white_noise_psd

FS = 5000.0      # Hz
DURATION = 60.0  # s
SIGMA = 10.0     # g
SEED = 42
FN = 200.0       # Hz


def _simulated_grms(Q, seed=SEED):
    a_base = generate_white_noise(SIGMA, FS, DURATION, seed)
    return rms(oscillator_response(a_base, FN, Q, FS))


def test_miles_value():
    # G_rms = sqrt(pi/2 x 200 x 10 x 0,04) = 11,21 g
    assert np.isclose(miles_grms(FN, 10.0, 0.04), 11.2099, rtol=1e-4)


def test_simulation_matches_miles():
    expected = miles_grms(FN, 10.0, white_noise_psd(SIGMA, FS))
    assert np.isclose(_simulated_grms(10.0), expected, rtol=0.05)


def test_q_ratio_is_sqrt_two():
    # Miles : G_rms proportionnel à sqrt(Q). Même bruit pour les deux simulations.
    ratio = _simulated_grms(20.0) / _simulated_grms(10.0)
    assert np.isclose(ratio, np.sqrt(2.0), rtol=0.05)