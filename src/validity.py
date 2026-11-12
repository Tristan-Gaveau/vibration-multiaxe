"""Validité physique d'une matrice interspectrale S(f), enveloppe terme à terme et correction.

S(f) est valide si elle est hermitienne ET semi-définie positive : pour tout vecteur w,
w^H . S . w est la PSD de la combinaison w1*a_x + w2*a_y + w3*a_z, donc >= 0.
Toutes les fonctions acceptent une matrice (n, n) ou un tableau (n_freq, n, n).
"""
import numpy as np


def build_matrix(level, r_xy, r_xz, r_yz):
    """Matrice 3x3 réelle : PSD égale à `level` (g²/Hz) sur chaque axe, corrélations r_ij hors diagonale.

    r_xy, r_xz, r_yz peuvent être des scalaires ou des tableaux (un par fréquence).
    """
    r_xy, r_xz, r_yz = np.broadcast_arrays(*(np.asarray(r, dtype=float) for r in (r_xy, r_xz, r_yz)))
    m = np.zeros(r_xy.shape + (3, 3))
    for i in range(3):
        m[..., i, i] = level
    m[..., 0, 1] = m[..., 1, 0] = level * r_xy
    m[..., 0, 2] = m[..., 2, 0] = level * r_xz
    m[..., 1, 2] = m[..., 2, 1] = level * r_yz
    return m


def eigenvalues(S):
    """Valeurs propres (réelles, triées par ordre croissant) de matrices hermitiennes."""
    return np.linalg.eigvalsh(S)


def is_valid(S, rel_tol=1e-9):
    """True si lambda_min >= -rel_tol * lambda_max (tolérance relative au bruit d'arrondi)."""
    ev = eigenvalues(S)
    return ev[..., 0] >= -rel_tol * ev[..., -1]


def envelope(S_a, S_b):
    """Enveloppe terme à terme : pour chaque terme, on garde celui des deux de plus grand module.

    Sur la diagonale (réelle positive) cela donne le maximum. Comme |S_ij| = |S_ji|, le choix est
    le même pour (i, j) et (j, i) : l'enveloppe reste hermitienne.
    """
    return np.where(np.abs(S_b) > np.abs(S_a), S_b, S_a)


def clip_negative_eigenvalues(S, rel_floor=0.0, keep_diagonal=True):
    """Corrige S en écrêtant ses valeurs propres négatives.

    rel_floor : plancher relatif (x lambda_max) pour les valeurs propres. 0 = écrêtage à zéro.
    keep_diagonal : True -> on remet les PSD de chaque axe à leur valeur d'origine
    (S -> D S D avec D diagonale positive, ce qui conserve le caractère semi-défini positif).
    Coût : les corrélations entre axes sont modifiées.
    """
    w, v = np.linalg.eigh(S)
    w = np.maximum(w, rel_floor * w[..., -1:])
    s_clipped = (v * w[..., None, :]) @ np.conj(np.swapaxes(v, -1, -2))
    if keep_diagonal:
        target = np.real(np.diagonal(S, axis1=-2, axis2=-1))
        current = np.real(np.diagonal(s_clipped, axis1=-2, axis2=-1))
        scale = np.sqrt(target / current)
        s_clipped = s_clipped * scale[..., :, None] * scale[..., None, :]
    return s_clipped