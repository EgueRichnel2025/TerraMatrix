\# TerraMatrix — Environnements scientifiques



TerraMatrix utilise Conda pour isoler les environnements de calcul scientifique.



Deux backends sont utilisés pour les expériences comparatives :



\- OpenBLAS

\- Intel MKL / oneMKL



Les deux environnements de comparaison utilisent Python 3.12.15,

NumPy 2.5.4 et SciPy 1.18.1.



\## Prérequis



Installer Miniforge ou une distribution Conda compatible.



Les fichiers de verrouillage fournis dans `environments/` ciblent

Windows 64 bits et contiennent les références exactes aux paquets Conda.



\## Recréer l'environnement OpenBLAS sous Windows 64 bits



Depuis la racine du dépôt :



&#x20;   conda create -n terramatrix-openblas-match --file environments/terramatrix-openblas-win-64.txt



Puis :



&#x20;   conda activate terramatrix-openblas-match

&#x20;   python -m pip install -e . --no-deps



\## Recréer l'environnement Intel MKL sous Windows 64 bits



Depuis la racine du dépôt :



&#x20;   conda create -n terramatrix-mkl-match --file environments/terramatrix-mkl-win-64.txt



Puis :



&#x20;   conda activate terramatrix-mkl-match

&#x20;   python -m pip install -e . --no-deps



\## Recréer les environnements avec les fichiers YAML



OpenBLAS :



&#x20;   conda env create -f environment-openblas.yml



Intel MKL :



&#x20;   conda env create -f environment-mkl.yml



Les fichiers YAML décrivent les dépendances principales et la sélection

du backend. Les fichiers explicites pour Windows 64 bits permettent une

reproduction plus précise des paquets Conda enregistrés.



\## Vérification



Dans chaque environnement, exécuter :



&#x20;   python -m pytest -q

&#x20;   ruff check src tests



Puis vérifier le backend chargé :



&#x20;   python -c "from threadpoolctl import threadpool\_info; print(threadpool\_info())"



Les résultats attendus doivent confirmer le backend correspondant.

Ne pas conclure à partir du seul nom de l'environnement : inspecter

les bibliothèques effectivement chargées.



\## Comparaisons scientifiques



Les comparaisons OpenBLAS/MKL doivent utiliser des données, dimensions,

graines aléatoires et paramètres identiques.



Le nombre de threads doit être contrôlé et enregistré. Les mesures

de performance doivent être accompagnées de validations numériques.



Les fichiers de verrouillage actuels ciblent Windows 64 bits.

Les autres systèmes nécessitent des environnements adaptés.

