# TerraMatrix — Comparaison PCA : OpenBLAS et Intel MKL

## 1. Objectif

Comparer la PCA par diagonalisation de la matrice de covariance et la PCA par SVD directe, en étudiant l'influence des dimensions et du backend BLAS.

## 2. Protocole expérimental

- Précision : `float64`.
- Backends : OpenBLAS et Intel MKL / oneMKL.
- 5 dimensions : 1000×20, 1000×50, 1000×100, 5000×20, 10000×20.
- 3 cas par dimension : bien conditionné, mal conditionné et rang déficient.
- 3 échauffements puis 20 répétitions chronométrées par méthode.
- Le chronométrage porte sur l'appel PCA, hors génération des matrices et vérification numérique.
- Les résultats sont les observations de cette machine et de ces environnements à 10 threads ; ils ne constituent pas une règle universelle sur les bibliothèques.

## 3. Résultats par dimension

| Matrice | Méthode | OpenBLAS (ms) | MKL (ms) | Ratio OpenBLAS/MKL | Plus rapide selon le ratio médian |
|---|---|---:|---:|---:|---|
| 1000x20 | covariance | 0.410450 | 1.948350 | 0.233× | OpenBLAS |
| 1000x20 | svd | 2.066400 | 0.982000 | 1.732× | MKL |
| 1000x50 | covariance | 3.027150 | 1.815950 | 1.667× | MKL |
| 1000x50 | svd | 8.845700 | 6.016050 | 1.638× | MKL |
| 1000x100 | covariance | 16.804500 | 5.558800 | 2.842× | MKL |
| 1000x100 | svd | 50.611300 | 9.755400 | 5.172× | MKL |
| 5000x20 | covariance | 2.651250 | 2.502700 | 1.119× | MKL |
| 5000x20 | svd | 8.172200 | 2.850600 | 2.844× | MKL |
| 10000x20 | covariance | 3.797950 | 4.085350 | 0.930× | OpenBLAS |
| 10000x20 | svd | 12.882400 | 11.674550 | 1.005× | MKL |

## 4. Synthèse par méthode

Convention : ratio supérieur à 1 favorise MKL ; ratio inférieur à 1 favorise OpenBLAS.

- **covariance** : 15 comparaisons ; ratio médian OpenBLAS/MKL = 1.219× ; MKL plus rapide dans 9 cas et OpenBLAS plus rapide dans 6 cas.
- **svd** : 15 comparaisons ; ratio médian OpenBLAS/MKL = 2.079× ; MKL plus rapide dans 14 cas et OpenBLAS plus rapide dans 1 cas.

## 5. Vérifications numériques

- Contrôles de rang attendus satisfaits : 30/30.
- Erreur relative maximale de reconstruction : 2.114e-15.
- Erreur maximale d'orthogonalité des composantes : 2.065e-14.
- Différence relative maximale des variances expliquées entre les méthodes : 2.089e-15.

## 6. Figures

- `results/figures/pca_performance_runtime_by_size_t10_r20.png`
- `results/figures/pca_performance_backend_ratio_t10_r20.png`

## 7. Fichiers de données

- `results/experiments/pca_performance_comparison_t10_r20.csv` : temps et diagnostics par méthode, dimension et backend.
- `results/experiments/pca_performance_speedup_t10_r20.csv` : comparaisons appariées.
- `results/experiments/pca_performance_speedup_summary_t10_r20.csv` : synthèse par dimension et méthode.

## 8. Limites

Les ratios dépendent du processeur, des versions des bibliothèques, du nombre de threads et de la variabilité du système. Les médianes sont rapportées avec les IQR dans les données détaillées ; les ratios ne doivent pas être interprétés comme des accélérations garanties. La PCA par covariance forme une matrice de taille variables × variables, tandis que la SVD directe travaille sur la matrice des données ; leurs coûts peuvent donc évoluer différemment lorsque les dimensions changent.
