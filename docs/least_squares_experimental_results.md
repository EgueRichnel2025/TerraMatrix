# TerraMatrix — Résultats expérimentaux des moindres carrés

> Rapport généré à partir du JSON expérimental. Les valeurs numériques
> et les temps présentés proviennent de l'exécution enregistrée ;
> aucune mesure n'a été reconstituée ou inventée.

## 1. Objectif

Cette campagne compare quatre méthodes de résolution des moindres
carrés : les équations normales, le QR classique sans pivotement,
le QR avec pivotement des colonnes et la SVD avec pseudoinverse.
Elle étudie la précision numérique, le rang, la stationnarité des
solutions et les durées observées.

## 2. Protocole expérimental

- Suite : `TM-LS-COMPARISON-01`.
- Dimensions : `1000 × 20`.
- Type numérique : `float64`.
- Graine de base : `20261022`.
- Écart-type du bruit : `0.001`.
- Nombre d'échauffements : `1`.
- Répétitions chronométrées : `5`.
- Chronomètre : `time.perf_counter_ns`.

Les matrices sont synthétiques. Les expériences de conditionnement
contrôlé utilisent une construction à facteurs orthogonaux et spectre
singulier prescrit. Les trois cas TM-LS-02 reprennent la même graine,
le même vecteur de coefficients de référence et le même bruit ; seule
la matrice change avec le spectre singulier prescrit.

TM-LS-03 est construit avec une colonne dupliquée exactement. Son rang
algébrique attendu est donc `p - 1`, même si les détails de la résolution
en arithmétique flottante dépendent de la méthode employée.

Les empreintes SHA-256 complètes, les paramètres détaillés et les
mesures brutes sont conservés dans le JSON source.

### Métriques

- **Résidu relatif** : norme du résidu divisée par la norme du vecteur
  de réponse, lorsqu'elle est non nulle.
- **Erreur relative sur les coefficients** : écart relatif à `beta_true`.
  Elle n'est pas utilisée pour évaluer la récupération des coefficients
  dans TM-LS-03, où la solution n'est pas unique.
- **Stationnarité relative** : mesure normalisée de la norme de
  `X.T @ (X @ beta_hat - b)`.
- **Temps médian et IQR** : statistiques des répétitions chronométrées.

Les durées mesurent les appels aux solveurs publics, incluant leur
validation des entrées et la préparation mémoire, mais excluant la
génération des données et le calcul des métriques après résolution.
Une synthèse temporelle n'est pas publiée lorsqu'un échauffement ou
une répétition échoue.

## 3. Résultats

### TM-LS-01 — Matrice bien conditionnée

- Graine : `20261022`.
- Rang numérique : `20/20`.
- Tolérance de rang : `2.2204e-13`.
- Conditionnement mesuré : `2.0000e+00`.
- Conditionnement ciblé : `—`.
- Construction : `orthogonal_factors_with_prescribed_spectrum`.
- Empreinte matrice (SHA-256, préfixe) : `b3ae237963630443`.
- Empreinte réponse (SHA-256, préfixe) : `f00234fe35d8d69b`.

| Méthode | Statut | Résidu relatif | Erreur relative β | Norme β | Stationnarité relative | Médiane (ms) | IQR (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| Équations normales | success | 9.9369e-03 | 1.7863e-03 | 3.8624e+00 | 3.0582e-14 | 0.087 | 0.008 |
| QR avec pivotement | success | 9.9369e-03 | 1.7863e-03 | 3.8624e+00 | 3.6091e-14 | 0.711 | 0.026 |
| QR sans pivotement | success | 9.9369e-03 | 1.7863e-03 | 3.8624e+00 | 2.6103e-14 | 0.710 | 0.018 |
| SVD + pseudoinverse | success | 9.9369e-03 | 1.7863e-03 | 3.8624e+00 | 9.0103e-14 | 0.865 | 0.009 |

### TM-LS-02 — Conditionnement ciblé 1e+02 (`TM-LS-02-kappa-1e2`)

- Graine : `20261022`.
- Rang numérique : `20/20`.
- Tolérance de rang : `2.2204e-13`.
- Conditionnement mesuré : `1.0000e+02`.
- Conditionnement ciblé : `1.0000e+02`.
- Construction : `orthogonal_factors_with_prescribed_spectrum`.
- Empreinte matrice (SHA-256, préfixe) : `df1b873c6b729f26`.
- Empreinte réponse (SHA-256, préfixe) : `14aaa0ab6d2ed297`.

| Méthode | Statut | Résidu relatif | Erreur relative β | Norme β | Stationnarité relative | Médiane (ms) | IQR (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| Équations normales | success | 2.0574e-02 | 5.7015e-02 | 3.8982e+00 | 7.8311e-15 | 0.083 | 0.007 |
| QR avec pivotement | success | 2.0574e-02 | 5.7015e-02 | 3.8982e+00 | 1.6149e-14 | 0.712 | 0.379 |
| QR sans pivotement | success | 2.0574e-02 | 5.7015e-02 | 3.8982e+00 | 9.1360e-15 | 0.698 | 0.016 |
| SVD + pseudoinverse | success | 2.0574e-02 | 5.7015e-02 | 3.8982e+00 | 4.9023e-14 | 0.774 | 0.029 |

### TM-LS-02 — Conditionnement ciblé 1e+04 (`TM-LS-02-kappa-1e4`)

- Graine : `20261022`.
- Rang numérique : `20/20`.
- Tolérance de rang : `2.2204e-13`.
- Conditionnement mesuré : `1.0000e+04`.
- Conditionnement ciblé : `1.0000e+04`.
- Construction : `orthogonal_factors_with_prescribed_spectrum`.
- Empreinte matrice (SHA-256, préfixe) : `812ec7634cc8e053`.
- Empreinte réponse (SHA-256, préfixe) : `f6e97d292b193758`.

| Méthode | Statut | Résidu relatif | Erreur relative β | Norme β | Stationnarité relative | Médiane (ms) | IQR (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| Équations normales | success | 2.9682e-02 | 5.0875e+00 | 2.0861e+01 | 3.9501e-14 | 0.080 | 0.003 |
| QR avec pivotement | success | 2.9682e-02 | 5.0875e+00 | 2.0861e+01 | 4.6614e-14 | 0.765 | 0.063 |
| QR sans pivotement | success | 2.9682e-02 | 5.0875e+00 | 2.0861e+01 | 1.1740e-14 | 0.769 | 0.050 |
| SVD + pseudoinverse | success | 2.9682e-02 | 5.0875e+00 | 2.0861e+01 | 1.0358e-13 | 0.852 | 0.049 |

### TM-LS-02 — Conditionnement ciblé 1e+06 (`TM-LS-02-kappa-1e6`)

- Graine : `20261022`.
- Rang numérique : `20/20`.
- Tolérance de rang : `2.2204e-13`.
- Conditionnement mesuré : `1.0000e+06`.
- Conditionnement ciblé : `1.0000e+06`.
- Construction : `orthogonal_factors_with_prescribed_spectrum`.
- Empreinte matrice (SHA-256, préfixe) : `2c40235997484cb8`.
- Empreinte réponse (SHA-256, préfixe) : `ce153a90d6b9bdd6`.

| Méthode | Statut | Résidu relatif | Erreur relative β | Norme β | Stationnarité relative | Médiane (ms) | IQR (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| Équations normales | success | 3.8234e-02 | 4.9339e+02 | 1.9068e+03 | 2.0407e-12 | 0.078 | 0.002 |
| QR avec pivotement | success | 3.8234e-02 | 4.9339e+02 | 1.9068e+03 | 7.9511e-12 | 0.668 | 0.026 |
| QR sans pivotement | success | 3.8234e-02 | 4.9339e+02 | 1.9068e+03 | 1.2730e-12 | 0.675 | 0.028 |
| SVD + pseudoinverse | success | 3.8234e-02 | 4.9339e+02 | 1.9068e+03 | 7.8198e-12 | 0.796 | 0.128 |

### TM-LS-03 — Matrice déficiente en rang

- Graine : `20261022`.
- Rang numérique : `19/20`.
- Tolérance de rang : `1.0178e-11`.
- Conditionnement mesuré : `—`.
- Conditionnement ciblé : `—`.
- Construction : `random_matrix_with_exact_duplicate_column`.
- Empreinte matrice (SHA-256, préfixe) : `9c64bf34c114a36b`.
- Empreinte réponse (SHA-256, préfixe) : `612fb41b0126c1c5`.

| Méthode | Statut | Résidu relatif | Erreur relative β | Norme β | Stationnarité relative | Médiane (ms) | IQR (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| Équations normales | failure | — | — | — | — | — | — |
| QR avec pivotement | success | 2.1568e-04 | — | 4.6643e+00 | 1.8192e-12 | 0.707 | 0.032 |
| QR sans pivotement | success | 2.1571e-04 | — | 6.5725e+10 | 5.3167e-03 | 0.693 | 0.023 |
| SVD + pseudoinverse | success | 2.1568e-04 | — | 4.6592e+00 | 8.3630e-12 | 0.829 | 0.052 |

- **Échec — Équations normales :** LinAlgError: Singular matrix

## 4. Interprétation

### Conditionnement et récupération des coefficients

Le résidu et l'erreur sur les coefficients répondent à des questions
différentes. Un résidu faible indique que la réponse calculée correspond
bien aux observations au sens des moindres carrés ; il ne garantit pas
que les coefficients retrouvés soient proches des coefficients de
référence lorsque la matrice est très mal conditionnée.

Dans TM-LS-02, les coefficients de référence et le bruit sont maintenus
constants entre les trois niveaux de conditionnement. Les différences
observées peuvent ainsi être étudiées par rapport au changement de
spectre singulier. Les valeurs du tableau doivent néanmoins être
interprétées en tenant compte du bruit ajouté à la réponse.

### Rang déficient

Dans TM-LS-03, le diagnostic numérique doit être interprété avec sa
tolérance. Une résolution qui retourne un vecteur ne prouve pas, à elle
seule, que cette solution soit numériquement satisfaisante. Les résidus,
la mesure de stationnarité, la norme des coefficients et le statut du
solveur doivent être examinés conjointement.

Les équations normales peuvent échouer sur la matrice singulière.
Le QR non pivoté ne révèle pas explicitement le rang dans cette
implémentation. Le QR pivoté estime le rang à partir de la diagonale
de `R`, tandis que la SVD utilise les valeurs singulières et un seuil
explicite. En cas de rang déficient, la solution SVD par pseudoinverse
vise une solution de norme euclidienne minimale dans le sous-espace
singulier conservé.

### Durées mesurées

Les temps sont ceux de cette exécution et de cet environnement précis.
Les différences peuvent inclure le coût des enveloppes Python, de la
validation et de la préparation mémoire, pas seulement celui des
routines BLAS ou LAPACK. Ces mesures ne suffisent donc pas à établir
un classement universel de performance entre méthodes ou bibliothèques.

## 5. Reproductibilité et limites

- Rapport brut : `results/experiments/least_squares_comparison.json`.
- Nombre d'expériences : `5`.
- Configuration chronométrée : `1` échauffement(s), 5 répétitions par méthode.
- L'environnement détaillé, les informations de bibliothèque numérique,
  les métadonnées de source et les durées individuelles se trouvent dans
  la clé `environment` et les enregistrements de chaque méthode du JSON.
- Les entrées sont synthétiques ; aucun jeu de données satellite réel
  n'est utilisé dans cette campagne.
- Les résultats restent spécifiques aux paramètres, au matériel, aux
  bibliothèques chargées et à la configuration logicielle enregistrés.
