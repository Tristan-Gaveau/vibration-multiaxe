# vibration-multiaxe

Prototype Python d'une chaîne de traitement pour la **spécification d'essais vibratoires multiaxes** : estimation de la matrice interspectrale de signaux triaxiaux, vérification de sa validité physique, génération de signaux à partir d'une matrice cible.

Projet personnel d'apprentissage, validé sur des **cas analytiques et des signaux synthétiques**. Ce n'est pas un outil industriel : voir la section [Hypothèses et limites](#hypothèses-et-limites).

Auteur : Tristan, élève-ingénieur ESILV (Modélisation Mécanique et Numérique).

## Chaîne de traitement

```
Accélérations triaxiales a(t) = [ax, ay, az]
            │
            ▼
Matrice S(f) 3×3 par Welch / CSD                 psd_csd.py        (étapes 1 à 3)
            │
            ▼
Test de validité (valeurs propres ≥ 0)           validity.py       (étape 4)
            │
            ├──► Enveloppe terme à terme et correction
            ▼
Génération de signaux triaxiaux depuis S(f)      generation.py     (étape 5)
            │
            ▼
Dommage de fatigue (FDS) mono-direction         damage.py         (étape 6)
```

## Statut

| Étape | Contenu | Statut |
|---|---|---|
| 1 | PSD mono-axe (Welch), validation par Parseval | Fait, testé |
| 2 | Oscillateur 1 DDL, validation par l'équation de Miles | Fait, testé |
| 3 | Matrice interspectrale 3×3, cohérence, phase | Fait, testé |
| 4 | Validité de la matrice, enveloppe invalide, correction | Fait, testé |
| 5 | Génération de signaux triaxiaux, validation en boucle fermée | Fait, testé |
| 6 | Dommage de fatigue (FDS) mono-direction, additivité pour b = 2 | Fait, testé |
| 7-8 | Surface de dommage, essai séquentiel contre simultané | Non faits |

## Installation

Testé sous Python 3.14. Dépendances : NumPy, SciPy, Matplotlib, pytest.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
pytest                           # 46 tests
```

## Utilisation

Chaque étape a un script de démonstration qui affiche les chiffres de validation et les graphiques :

```bash
python src/demo_step1.py   # PSD d'un bruit blanc, compromis résolution / variance
python src/demo_step2.py   # oscillateur 1 DDL et équation de Miles
python src/demo_step3.py   # matrice 3x3, cohérence et phase
python src/demo_step4.py   # enveloppe non physique et correction
python src/demo_step5.py   # génération triaxiale et ré-estimation
python src/demo_step6.py   # FDS de deux spécifications de même RMS, additivité
```

Exemple minimal (génération puis ré-estimation), à lancer depuis le dossier `src/` :

```python
import numpy as np
from generation import example_target, synthesize_triaxial
from psd_csd import csd_matrix

fs, n = 5000.0, 300_000
f = np.fft.rfftfreq(n, 1 / fs)
x = synthesize_triaxial(example_target(f), fs, n, seed=0)   # (3, n), en g
f_est, S = csd_matrix(x, fs, nperseg=4096)                  # (n_freq, 3, 3), en g²/Hz
```

## Conventions

- Accélérations en **g**, PSD **unilatérales** en **g²/Hz**. L'intégrale de la PSD sur la fréquence est la variance du signal.
- Matrice interspectrale de forme `(n_freq, 3, 3)` avec `S[k, i, j] = E[conj(X_i) · X_j]`, c'est-à-dire `scipy.signal.csd(x_i, x_j)`. Si `x_j` est `x_i` retardé de τ, la phase de `S_ij` vaut −2π·f·τ.
- Estimation de Welch : fenêtre de Hann, recouvrement 50 %.

### Normalisation de la génération

Pour un signal réel de N points échantillonné à fs, la PSD unilatérale G vérifie G(f_k) = 2·E|X_k|² / (fs·N), où X_k est la FFT sans fenêtre (le facteur 2 vient du repliement des fréquences négatives). Donc E|X_k|² = G·fs·N/2, et on génère X = √(fs·N/2) · conj(H) · ε avec S = H·Hᴴ et ε un bruit complexe de variance 1. On utilise `conj(H)` pour respecter la convention `conj(X_i)·X_j` ci-dessus : avec `H`, la phase des termes croisés serait inversée. La somme G·Δf (Δf = fs/N) redonne la variance (Parseval).

## Validation

Chaque module est comparé à un résultat analytique. Les tolérances ne sont pas arbitraires : elles sont calibrées sur la dispersion mesurée (20 à 30 graines aléatoires) ou sur l'erreur statistique attendue, avec une marge.

| Étape | Vérification | Résultat | Tolérance du test |
|---|---|---|---|
| 1 | Parseval : intégrale de la PSD = variance | écart 0,12 à 0,14 % | 1 % (erreur statistique ≈ √(2/N) ≈ 0,26 %) |
| 1 | Écart-type relatif de la PSD estimée | 5,86 % pour K = 291, égal à 1/√K | ± 15 % relatif |
| 1 | Sinusoïde (cas déterministe), variance A²/2 | Parseval < 1e-3 | 1e-3 |
| 2 | Équation de Miles (f_n = 200 Hz, Q = 10, 0,04 g²/Hz) | 11,17 g simulé pour 11,21 g, soit 0,3 % | 5 % |
| 2 | Q de 10 à 20 : le RMS est multiplié par √2 | 1,414 | 5 % |
| 3 | Cohérence γ²(x,y) pour ρ = 0,7 (cible 0,49) | 0,490 | ± 0,01 |
| 3 | Cohérence de signaux indépendants | ≈ 0,004, soit environ 1/K avec K = 291 | < 3/K |
| 3 | Un seul segment (K = 1) | γ² = 1 partout, même pour des signaux indépendants | exact |
| 3 | Symétrie hermitienne, diagonale réelle positive | vérifiées | 1e-12 |
| 3 | Convention de phase (signal retardé) | phase = −2π·f·τ | 0,05 |
| 4 | A et B valides, leur enveloppe non | valeur propre minimale −0,0109 g²/Hz, invalide de 855 à 1147 Hz | valeur analytique `L·(1 − 0,9·√2)` à 1e-9 |
| 4 | Zone invalide = zone où r(f) > 1/√2 | identique | égalité exacte |
| 4 | Matrice estimée par Welch (K = 291) | valide | valeur propre ≥ −1e-9·λmax |
| 5 | RMS par axe contre la cible | écart ≤ 0,21 % | 1 % |
| 5 | Les 9 termes, erreur normalisée par √(S_ii·S_jj) | ≤ 1,4 % au pire sur 30 graines | 3 % |
| 5 | Phase des termes croisés | ≤ 0,011 rad | 0,03 rad |
| 5 | Couplage purement imaginaire | +90,00° | ± 1° |
| 5 | Kurtosis (signal gaussien) | 2,995 | 3 ± 0,05 |
| 5 | Matrice singulière (y = x) acceptée | signaux identiques | 1e-9 |
| 6 | z_rms (déplacement relatif) contre la formule exacte du bruit blanc | écart < 0,01 % (fn = 50, 200, 400 Hz) | 1e-3 |
| 6 | Pente du FDS en bruit blanc : D ~ fn^(1 − 3b/2) | pente exacte à 1e-4 près (b = 2, 4, 6, 8) | ± 0,01 |
| 6 | Additivité pour b = 2 : D(P1 + P2) = D(P1) + D(P2) | rapport 1,000000 | 1e-9 |
| 6 | Loi d'échelle : D(2P) = 2^(b/2) · D(P), donc sur-additivité pour b > 2 | exacte | 1e-9 |
| 6 | z_rms calculé contre simulation temporelle (300 s) | écart 0,4 à 0,6 % | 2 % |
| 6 | Formule du FDS contre comptage de pics de la simulation | −1,5 % (b = 2), −3,7 % (b = 4), −5,4 % (b = 6), −7,0 % (b = 8) sur 5 graines | 3 % (b = 2), 8 % (b = 4) |

## Hypothèses et limites

- **Aucune donnée de vol réelle.** Tous les signaux sont synthétiques. Les résultats valident le code, pas une méthode sur des environnements réels.
- **Signaux gaussiens stationnaires.** La génération produit un signal gaussien par construction (kurtosis ≈ 3). Les environnements non gaussiens ou non stationnaires ne sont pas traités.
- **Signal généré = une réalisation périodique** de période N (synthèse par FFT).
- **Cohérence biaisée si peu de moyennes** : elle vaut environ 1/K pour des signaux indépendants, et exactement 1 pour K = 1. Une cohérence élevée sur des enregistrements courts est à interpréter avec prudence.
- **Oscillateur à 1 degré de liberté** : pas de modèle de structure réelle, pas de tenseur de contraintes, pas de pilotage d'un pot vibrant ni d'une table multiaxe.
- **Équation de Miles** : approximation valable pour une PSD plate autour de f_n et un Q suffisamment grand. Biais attendu d'environ 1 % avec Q = 10 (à vérifier).
- **Enveloppe terme à terme** : le choix « plus grand module » (diagonale = maximum) est une convention du projet, pas une méthode normalisée. Une enveloppe de matrices valides peut être non physique (voir étape 4).
- **Correction par écrêtage des valeurs propres** : elle rend la matrice valide, mais modifie les niveaux de PSD. En remettant la diagonale à sa valeur, ce sont les cohérences qui changent (de 0,90 à 0,73 au pic de l'exemple). L'effet de cette correction sur le dommage n'est pas évalué. Des méthodes plus avancées (matrice valide la plus proche) ne sont pas implémentées.
- **Tolérances statistiques** : calibrées sur 20 à 30 graines, elles ne sont pas des garanties.
- **Dommage relatif uniquement** : C = 1 et z_rms en mm, donc seuls les rapports de dommage entre deux spécifications ont un sens. Aucune constante matériau, aucune prédiction de durée de vie.
- **Contrainte supposée proportionnelle au déplacement relatif** de l'oscillateur ; règle de Miner linéaire, exposant b fixé (valeurs usuelles de 4 à 8, à vérifier selon le matériau).
- **Formule du FDS à pics de Rayleigh** (bande étroite, un cycle par période propre) : comparée à un comptage de pics sur simulation, elle est conservatrice de 1,5 % (b = 2) à 7 % (b = 8) pour Q = 10. Ce n'est **pas** un comptage rainflow. L'écart dépend de Q et de la largeur de bande du signal.
- **FDS mono-direction** : le dommage multiaxial (projection sur une direction, surface de dommage) n'est pas traité.
- **Étapes 7 et 8 non faites** : pas de surface de dommage, pas de comparaison essai séquentiel contre essai simultané.

## Structure du dépôt

```
vibration-multiaxe/
├── README.md
├── requirements.txt
├── pytest.ini
├── src/
│   ├── psd_csd.py        # Welch, Parseval, matrice CSD, cohérence, bruit blanc
│   ├── oscillator.py     # oscillateur 1 DDL, équation de Miles
│   ├── validity.py       # valeurs propres, enveloppe, correction
│   ├── generation.py     # synthèse de signaux triaxiaux depuis S(f)
│   ├── damage.py         # FDS mono-direction (déplacement relatif, dommage relatif)
│   └── demo_step1.py … demo_step6.py
└── tests/
    ├── test_psd.py
    ├── test_miles.py
    ├── test_csd.py
    ├── test_validity.py
    ├── test_generation.py
    ├── test_damage.py
    └── test_additivity.py
```

## Perspectives

- **Étapes 7-8** : surface de dommage directionnelle D(θ, φ) par projection uᵀ·S·u, puis comparaison essai séquentiel contre essai simultané (hypothèse à tester numériquement : pour b > 2 et des axes non corrélés, l'essai séquentiel sous-estime le dommage de l'essai simultané).
- Comptage rainflow à la place du comptage de pics, et FDS sur des signaux non gaussiens.
- Données réelles (par exemple une centrale inertielle) : vérifier la fréquence d'échantillonnage, le repliement spectral, la stationnarité et la bande utile.

## Références (à vérifier avant citation)

- C. Lalanne, *Mechanical Vibration and Shock Analysis* (vibrations aléatoires et fatigue).
- MIL-STD-810H, méthode 527 (essais multi-exciteurs).
- Basquin (N·σ^b = C) et Miner (D = Σ nᵢ/Nᵢ) : lois de fatigue classiques, voir Lalanne.
- Documentation SciPy : `scipy.signal.welch`, `csd`, `coherence`.
