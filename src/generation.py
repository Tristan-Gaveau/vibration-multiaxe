"""Génération de signaux triaxiaux gaussiens à partir d'une matrice interspectrale cible S(f).

Convention (identique à psd_csd.csd_matrix) : S[k, i, j] = E[conj(X_i) . X_j], PSD unilatérale en g²/Hz.
Principe : S = H.H^H, X(f) = facteur . conj(H(f)) . eps(f), eps = bruit complexe gaussien de variance 1,
puis irfft. conj(H) (et non H) donne la bonne phase pour la convention conj(X_i).X_j.
"""
import numpy as np


def example_target(f, base=0.05, peak=0.25, f_peak=600.0, width=150.0):
    """Matrice cible d'exemple (n_freq, 3, 3) : profil fréquentiel en cloche x matrice de corrélation complexe.

    base, peak : niveaux de PSD (g²/Hz) ; f_peak, width : centre et largeur de la cloche (Hz).
    La matrice de corrélation est hermitienne définie positive (valeurs propres 0,11 ; 0,96 ; 1,93).
    """
    f = np.asarray(f, dtype=float)
    profile = base + peak * np.exp(-(((f - f_peak) / width) ** 2))
    m_corr = np.array(
        [
            [1.0, 0.7 + 0.2j, 0.1],
            [0.7 - 0.2j, 1.2, -0.4 + 0.3j],
            [0.1, -0.4 - 0.3j, 0.8],
        ],
        dtype=complex,
    )
    return profile[:, None, None] * m_corr


def synthesize_triaxial(S, fs, n_samples, seed):
    """Signal temporel (n_voies, n_samples), en g, de matrice interspectrale S(f) (g²/Hz).

    S est donnée sur la grille np.fft.rfftfreq(n_samples, 1/fs), de forme (n_samples//2 + 1, n, n).
    La décomposition en valeurs propres (et non Cholesky) accepte les matrices semi-définies
    positives singulières (valeur propre nulle), par exemple une matrice corrigée à l'étape 4.
    """
    n_bins = n_samples // 2 + 1
    S = np.asarray(S)
    if S.shape[0] != n_bins:
        raise ValueError(f"S doit avoir {n_bins} fréquences (rfftfreq de n_samples), reçu {S.shape[0]}.")
    n_ch = S.shape[-1]

    w, v = np.linalg.eigh(S)
    h = v * np.sqrt(np.clip(w, 0.0, None))[..., None, :]      # S = H H^H

    rng = np.random.default_rng(seed)
    eps = (rng.standard_normal((n_bins, n_ch)) + 1j * rng.standard_normal((n_bins, n_ch))) / np.sqrt(2.0)

    # E|X_k|² = PSD x fs x N / 2 pour une PSD unilatérale (voir démonstration dans le README).
    x = np.sqrt(fs * n_samples / 2.0) * np.einsum("kij,kj->ki", np.conj(h), eps)
    x[0] = 0.0                                                # pas de composante continue
    if n_samples % 2 == 0:
        x[-1] = np.sqrt(2.0) * x[-1].real                     # Nyquist réel, variance conservée
    return np.fft.irfft(x, n=n_samples, axis=0).T