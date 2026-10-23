"""Summarize and plot reproducible PCA benchmarks for OpenBLAS and MKL."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "results" / "experiments"
FIGURES = ROOT / "results" / "figures"
REPORT = ROOT / "docs" / "pca_backend_comparison.md"

METHODS = ("covariance", "svd")
BACKENDS = ("openblas", "mkl")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    files = sorted(EXPERIMENTS.glob("pca_performance_size_*_t10_r20.json"))
    if len(files) != 10:
        raise ValueError(f"10 benchmark JSON files expected; found {len(files)}.")

    rows: list[dict[str, Any]] = []
    paired: dict[tuple[int, int, str, str], dict[str, float]] = defaultdict(dict)
    comparison_errors: list[float] = []
    rank_checks: list[bool] = []

    for path in files:
        match = re.fullmatch(
            r"pca_performance_size_n(\d+)_p(\d+)_(openblas|mkl)_t10_r20\.json",
            path.name,
        )
        if match is None:
            raise ValueError(f"Unexpected benchmark filename: {path.name}")

        n_samples, n_features, backend = (
            int(match.group(1)),
            int(match.group(2)),
            match.group(3),
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        detected = " ".join(
            data["environment"]["runtime"].get("blas_backends", [])
        ).lower()

        if backend == "openblas" and "openblas" not in detected:
            raise ValueError(f"OpenBLAS backend mismatch: {path.name}")
        if backend == "mkl" and not any(
            marker in detected for marker in ("mkl", "onemkl")
        ):
            raise ValueError(f"MKL backend mismatch: {path.name}")

        params = data["parameters"]
        if (
            params["n_samples"] != n_samples
            or params["n_features"] != n_features
        ):
            raise ValueError(f"Dimension mismatch: {path.name}")

        for experiment in data["experiments"]:
            rank = experiment["rank_diagnostics"]
            rank_checks.append(bool(rank["expectation_matches"]))
            comparison_errors.append(
                float(
                    experiment["comparison"][
                        "relative_explained_variance_difference_l2"
                    ]
                )
            )

            for method in METHODS:
                result = experiment["methods"][method]
                timing = result["timing"]
                raw = timing["raw_execution_times_seconds"]

                if timing["timed_repetitions"] != 20 or len(raw) != 20:
                    raise ValueError(
                        f"Invalid repetition count: {path.name}, "
                        f"{experiment['experiment_id']}, {method}"
                    )
                if not all(math.isfinite(t) and t > 0 for t in raw):
                    raise ValueError(f"Invalid timing values: {path.name}")

                median_ms = float(timing["median_seconds"]) * 1000
                key = (
                    n_samples,
                    n_features,
                    experiment["experiment_id"],
                    method,
                )
                paired[key][backend] = median_ms

                rows.append({
                    "backend": backend,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "shape": f"{n_samples}x{n_features}",
                    "experiment_id": experiment["experiment_id"],
                    "case": experiment["case"],
                    "method": method,
                    "median_ms": median_ms,
                    "mean_ms": float(timing["mean_seconds"]) * 1000,
                    "iqr_ms": float(timing["iqr_seconds"]) * 1000,
                    "min_ms": float(timing["min_seconds"]) * 1000,
                    "max_ms": float(timing["max_seconds"]) * 1000,
                    "timed_repetitions": len(raw),
                    "reconstruction_error": result[
                        "reconstruction_relative_error_fro"
                    ],
                    "orthogonality_error": result[
                        "component_orthogonality_error_fro"
                    ],
                    "numerical_rank": rank["numerical_rank"],
                    "expected_rank": rank["expected_numerical_rank"],
                    "rank_expectation_matches": rank["expectation_matches"],
                })

    if len(rows) != 60:
        raise ValueError(f"Expected 60 method measurements, found {len(rows)}.")
    if not all(rank_checks):
        raise ValueError("At least one numerical-rank validation failed.")

    comparison_rows: list[dict[str, Any]] = []
    for key, backends in sorted(paired.items()):
        if set(backends) != set(BACKENDS):
            raise ValueError(f"Missing paired backend measurements: {key}")

        openblas_ms = backends["openblas"]
        mkl_ms = backends["mkl"]
        ratio = openblas_ms / mkl_ms

        comparison_rows.append({
            "n_samples": key[0],
            "n_features": key[1],
            "experiment_id": key[2],
            "method": key[3],
            "openblas_median_ms": openblas_ms,
            "mkl_median_ms": mkl_ms,
            "openblas_over_mkl": ratio,
            "faster_backend": "MKL" if ratio > 1 else "OpenBLAS",
        })

    if len(comparison_rows) != 30:
        raise ValueError(
            f"Expected 30 paired comparisons; found {len(comparison_rows)}."
        )

    summary_rows: list[dict[str, Any]] = []
    dimensions = sorted({
        (int(row["n_samples"]), int(row["n_features"])) for row in rows
    })

    for n_samples, n_features in dimensions:
        for method in METHODS:
            subset = [
                row for row in comparison_rows
                if row["n_samples"] == n_samples
                and row["n_features"] == n_features
                and row["method"] == method
            ]
            ratios = [float(row["openblas_over_mkl"]) for row in subset]
            openblas_times = [
                float(row["openblas_median_ms"]) for row in subset
            ]
            mkl_times = [float(row["mkl_median_ms"]) for row in subset]
            ratio_median = statistics.median(ratios)

            summary_rows.append({
                "n_samples": n_samples,
                "n_features": n_features,
                "shape": f"{n_samples}x{n_features}",
                "method": method,
                "paired_cases": len(subset),
                "openblas_median_ms": statistics.median(openblas_times),
                "mkl_median_ms": statistics.median(mkl_times),
                "median_openblas_over_mkl": ratio_median,
                "mkl_wins": sum(r > 1 for r in ratios),
                "openblas_wins": sum(r < 1 for r in ratios),
                "faster_backend_by_median": (
                    "MKL" if ratio_median > 1 else "OpenBLAS"
                ),
            })

    write_csv(
        EXPERIMENTS / "pca_performance_comparison_t10_r20.csv", rows
    )
    write_csv(
        EXPERIMENTS / "pca_performance_speedup_t10_r20.csv",
        comparison_rows,
    )
    write_csv(
        EXPERIMENTS / "pca_performance_speedup_summary_t10_r20.csv",
        summary_rows,
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(dimensions))
    width = 0.19

    fig, ax = plt.subplots(figsize=(12, 6))
    series = [
        ("covariance", "openblas", "Covariance · OpenBLAS"),
        ("covariance", "mkl", "Covariance · MKL"),
        ("svd", "openblas", "SVD · OpenBLAS"),
        ("svd", "mkl", "SVD · MKL"),
    ]

    for index, (method, backend, label) in enumerate(series):
        values = []
        for n_samples, n_features in dimensions:
            selected = [
                float(row["median_ms"]) for row in rows
                if int(row["n_samples"]) == n_samples
                and int(row["n_features"]) == n_features
                and row["method"] == method
                and row["backend"] == backend
            ]
            values.append(statistics.median(selected))
        ax.bar(x + (index - 1.5) * width, values, width, label=label)

    ax.set_xticks(x, [f"{n}×{p}" for n, p in dimensions])
    ax.set_xlabel("Dimensions de la matrice (observations × variables)")
    ax.set_ylabel("Temps médian par appel (ms)")
    ax.set_title("PCA : performances selon les dimensions et le backend")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "pca_performance_runtime_by_size_t10_r20.png",
        dpi=160,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for index, method in enumerate(METHODS):
        values = [
            float(row["median_openblas_over_mkl"])
            for row in summary_rows if row["method"] == method
        ]
        ax.bar(x + (index - 0.5) * 0.32, values, 0.32, label=method)

    ax.axhline(1, linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{n}×{p}" for n, p in dimensions])
    ax.set_xlabel("Dimensions de la matrice (observations × variables)")
    ax.set_ylabel("Ratio médian OpenBLAS / MKL (échelle logarithmique)")
    ax.set_title("Comparaison des temps médians : OpenBLAS face à MKL")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "pca_performance_backend_ratio_t10_r20.png",
        dpi=160,
    )
    plt.close(fig)

    lines = [
        "# TerraMatrix — Comparaison PCA : OpenBLAS et Intel MKL",
        "",
        "## 1. Objectif",
        "",
        "Comparer la PCA par diagonalisation de la matrice de covariance "
        "et la PCA par SVD directe, en étudiant l'influence des dimensions "
        "et du backend BLAS.",
        "",
        "## 2. Protocole expérimental",
        "",
        "- Précision : `float64`.",
        "- Backends : OpenBLAS et Intel MKL / oneMKL.",
        "- 5 dimensions : " + ", ".join(
            f"{n}×{p}" for n, p in dimensions
        ) + ".",
        "- 3 cas par dimension : bien conditionné, mal conditionné et rang déficient.",
        "- 3 échauffements puis 20 répétitions chronométrées par méthode.",
        "- Le chronométrage porte sur l'appel PCA, hors génération des matrices "
        "et vérification numérique.",
        "- Les résultats sont les observations de cette machine et de ces "
        "environnements à 10 threads ; ils ne constituent pas une règle "
        "universelle sur les bibliothèques.",
        "",
        "## 3. Résultats par dimension",
        "",
        (
            "| Matrice | Méthode | OpenBLAS (ms) | MKL (ms) | "
            "Ratio OpenBLAS/MKL | Plus rapide selon le ratio médian |"
        ),
        "|---|---|---:|---:|---:|---|",
    ]

    for row in summary_rows:
        lines.append(
            f"| {row['shape']} | {row['method']} "
            f"| {row['openblas_median_ms']:.6f} "
            f"| {row['mkl_median_ms']:.6f} "
            f"| {row['median_openblas_over_mkl']:.3f}× "
            f"| {row['faster_backend_by_median']} |"
        )

    lines.extend([
        "",
        "## 4. Synthèse par méthode",
        "",
        "Convention : ratio supérieur à 1 favorise MKL ; ratio inférieur "
        "à 1 favorise OpenBLAS.",
        "",
    ])

    for method in METHODS:
        selected = [
            row for row in comparison_rows if row["method"] == method
        ]
        ratios = [float(row["openblas_over_mkl"]) for row in selected]
        lines.append(
            f"- **{method}** : {len(ratios)} comparaisons ; "
            f"ratio médian OpenBLAS/MKL = {statistics.median(ratios):.3f}× ; "
            f"MKL plus rapide dans {sum(r > 1 for r in ratios)} cas et "
            f"OpenBLAS plus rapide dans {sum(r < 1 for r in ratios)} cas."
        )

    max_reconstruction = max(
        float(row["reconstruction_error"]) for row in rows
    )
    max_orthogonality = max(
        float(row["orthogonality_error"]) for row in rows
    )
    max_variance_difference = max(comparison_errors)

    lines.extend([
        "",
        "## 5. Vérifications numériques",
        "",
        f"- Contrôles de rang attendus satisfaits : {sum(rank_checks)}/{len(rank_checks)}.",
        f"- Erreur relative maximale de reconstruction : {max_reconstruction:.3e}.",
        f"- Erreur maximale d'orthogonalité des composantes : {max_orthogonality:.3e}.",
        f"- Différence relative maximale des variances expliquées entre les "
        f"méthodes : {max_variance_difference:.3e}.",
        "",
        "## 6. Figures",
        "",
        "- `results/figures/pca_performance_runtime_by_size_t10_r20.png`",
        "- `results/figures/pca_performance_backend_ratio_t10_r20.png`",
        "",
        "## 7. Fichiers de données",
        "",
        "- `results/experiments/pca_performance_comparison_t10_r20.csv` : "
        "temps et diagnostics par méthode, dimension et backend.",
        "- `results/experiments/pca_performance_speedup_t10_r20.csv` : "
        "comparaisons appariées.",
        "- `results/experiments/pca_performance_speedup_summary_t10_r20.csv` : "
        "synthèse par dimension et méthode.",
        "",
        "## 8. Limites",
        "",
        "Les ratios dépendent du processeur, des versions des bibliothèques, "
        "du nombre de threads et de la variabilité du système. Les médianes "
        "sont rapportées avec les IQR dans les données détaillées ; les ratios "
        "ne doivent pas être interprétés comme des accélérations garanties. "
        "La PCA par covariance forme une matrice de taille variables × variables, "
        "tandis que la SVD directe travaille sur la matrice des données ; "
        "leurs coûts peuvent donc évoluer différemment lorsque les dimensions "
        "changent.",
        "",
    ])

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("Analyse PCA terminée.")
    print(f"JSON lus : {len(files)}")
    print(f"Mesures de performance : {len(rows)}")
    print(f"Comparaisons appariées : {len(comparison_rows)}")
    print("Tableaux CSV : 3")
    print(f"Graphiques : 2 dans {FIGURES}")
    print(f"Rapport : {REPORT}")


if __name__ == "__main__":
    main()
