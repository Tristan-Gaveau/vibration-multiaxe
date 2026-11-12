"""Estimation de PSD (méthode de Welch) et outils de validation.

Conventions du projet :
- PSD unilatérale, en g²/Hz (accélération en g).
- Intégrale de la PSD sur la fréquence = variance du signal = RMS² (signal centré).
"""
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import csd, welch


def generate_white_noise(sigma, fs, duration, seed):
    """Bruit blanc gaussien centré, écart-type `sigma` (g), fréquence `fs` (Hz), `duration` (s).

    `seed` est obligatoire : un test doit être reproductible.
    """
    n_samples = int(round(fs * duration))
    rng = np.random.default_rng(seed)  # générateur local, sans état global
    return rng.normal(0.0, sigma, n_samples)


def white_noise_psd(sigma, fs):
    """PSD unilatérale théorique (g²/Hz) d'un bruit blanc de variance sigma² échantillonné à fs.

    variance = PSD x bande, avec une bande de 0 à fs/2.
    """
    return sigma**2 / (fs / 2.0)


def estimate_psd(x, fs, nperseg=2048):
    """PSD de Welch : fenêtre de Hann, recouvrement 50 %, unilatérale."""
    return welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="density",
    )


def n_averages(n_samples, nperseg):
    """Nombre K de segments moyennés par Welch (recouvrement 50 %)."""
    step = nperseg // 2
    return (n_samples - nperseg) // step + 1


def parseval_relative_error(x, f, psd):
    """Écart relatif entre la variance du signal et l'intégrale de la PSD."""
    variance_time = np.var(x)
    variance_spectrum = trapezoid(psd, f)
    return abs(variance_spectrum - variance_time) / variance_time


def band_mean_psd(f, psd, f_center, half_width):
    """PSD moyennée sur [f_center - half_width ; f_center + half_width] (Hz).

    Moyenner sur plusieurs points réduit l'erreur statistique d'un point isolé.
    """
    mask = (f >= f_center - half_width) & (f <= f_center + half_width)
    if not mask.any():
        raise ValueError("Aucun point de PSD dans la bande demandée.")
    return psd[mask].mean()


def correlated_triaxial_noise(sigma, rho, fs, duration, seeds=(1, 2, 3)):
    """Signal triaxial [x, y, z] de bruits blancs, de forme (3, N).

    x = n1
    y = rho*n1 + sqrt(1 - rho²)*n2   (corrélé à x : cohérence gamma² = rho²)
    z = n3                           (indépendant)

    n1, n2, n3 DOIVENT avoir des graines différentes, sinon les bruits sont identiques.
    """
    if len(set(seeds)) != 3:
        raise ValueError("Il faut trois graines différentes.")
    n1, n2, n3 = (generate_white_noise(sigma, fs, duration, seed) for seed in seeds)
    return np.vstack([n1, rho * n1 + np.sqrt(1.0 - rho**2) * n2, n3])


def csd_matrix(signals, fs, nperseg=2048):
    """Matrice interspectrale S(f), de forme (n_freq, n_voies, n_voies), en g²/Hz.

    Convention : S[k, i, j] = E[conj(X_i) . X_j], soit scipy.signal.csd(x_i, x_j).
    Si x_j est x_i retardé de tau, la phase de S[k, i, j] vaut -2*pi*f*tau.
    Les 9 termes sont calculés indépendamment : la symétrie hermitienne est testable.
    """
    signals = np.asarray(signals)
    n_ch = signals.shape[0]
    matrix = None
    for i in range(n_ch):
        for j in range(n_ch):
            f, s_ij = csd(
                signals[i],
                signals[j],
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=nperseg // 2,
                scaling="density",
            )
            if matrix is None:
                matrix = np.empty((len(f), n_ch, n_ch), dtype=complex)
            matrix[:, i, j] = s_ij
    return f, matrix


def coherence(matrix, i, j):
    """Cohérence quadratique gamma²_ij(f) = |S_ij|² / (S_ii . S_jj), comprise entre 0 et 1.

    Attention : estimée avec K segments, elle vaut ~1/K pour deux signaux indépendants
    (et exactement 1 pour K = 1).
    """
    return np.abs(matrix[:, i, j]) ** 2 / (matrix[:, i, i].real * matrix[:, j, j].real)