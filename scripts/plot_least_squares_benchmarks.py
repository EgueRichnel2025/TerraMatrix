import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE = Path("results/experiments/least_squares_comparison_t10_r20.csv")
OUTPUT = Path("results/figures")
OUTPUT.mkdir(parents=True, exist_ok=True)

METHODS = [
    "normal_equations",
    "qr_unpivoted",
    "qr_pivoted",
    "svd_pseudoinverse",
]

LABELS = {
    "normal_equations": "Normal equations",
    "qr_unpivoted": "QR without pivoting",
    "qr_pivoted": "QR with pivoting",
    "svd_pseudoinverse": "SVD pseudoinverse",
}

SIZES = [(1000, 20), (5000, 20), (10000, 20), (1000, 50), (1000, 100)]
BACKENDS = ("openblas", "mkl")

if not SOURCE.exists():
    sys.exit(f"Missing benchmark data: {SOURCE}")

with SOURCE.open(newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

pairs = defaultdict(dict)
for row in rows:
    key = (
        int(row["n_samples"]),
        int(row["n_features"]),
        row["experiment_id"],
        row["method"],
    )
    pairs[key][row["backend"]] = row

def valid_time(row):
    if not row or row.get("solver_status") != "success":
        return None
    value = row.get("median_ms", "")
    if not value:
        return None
    return float(value)

# Figure 1: median runtime for the well-conditioned case.
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()
positions = list(range(len(SIZES)))
width = 0.38
xlabels = [f"{n}×{p}" for n, p in SIZES]

for ax, method in zip(axes, METHODS, strict=True):
    open_times, mkl_times = [], []

    for n, p in SIZES:
        open_row = pairs[(n, p, "TM-LS-01", method)].get("openblas")
        mkl_row = pairs[(n, p, "TM-LS-01", method)].get("mkl")
        open_times.append(valid_time(open_row))
        mkl_times.append(valid_time(mkl_row))

    # LS-01 should provide timings for all four methods.
    if any(value is None for value in open_times + mkl_times):
        ax.set_title(f"{LABELS[method]} — missing timing")
        continue

    ax.bar(
        [x - width / 2 for x in positions],
        open_times,
        width,
        label="OpenBLAS",
    )
    ax.bar(
        [x + width / 2 for x in positions],
        mkl_times,
        width,
        label="Intel MKL",
    )
    ax.set_title(LABELS[method])
    ax.set_xticks(positions, xlabels, rotation=20)
    ax.set_ylabel("Median solver-call time (ms)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

fig.suptitle(
    "Least-squares scaling — well-conditioned case (TM-LS-01)\n"
    "10 threads, 3 warm-ups, 20 timed repetitions",
    fontsize=14,
)
fig.tight_layout()
path1 = OUTPUT / "least_squares_runtime_by_size_t10_r20.png"
fig.savefig(path1, dpi=180, bbox_inches="tight")
plt.close(fig)

# Figure 2: paired OpenBLAS/MKL median-time ratios over all cases.
ratios = defaultdict(list)
paired_count = 0
skipped_count = 0

for (_, _, _, method), backends in pairs.items():
    open_row = backends.get("openblas")
    mkl_row = backends.get("mkl")
    open_time = valid_time(open_row)
    mkl_time = valid_time(mkl_row)

    if open_time is None or mkl_time is None or mkl_time <= 0:
        skipped_count += 1
        continue

    ratios[method].append(open_time / mkl_time)
    paired_count += 1

fig, ax = plt.subplots(figsize=(11, 6))
plot_data = [ratios[method] for method in METHODS]

ax.boxplot(plot_data, tick_labels=[LABELS[m] for m in METHODS])
ax.set_yscale("log")
ax.axhline(1.0, linestyle="--", linewidth=1.2, label="Equal median times")
ax.set_ylabel("OpenBLAS median time / MKL median time")
ax.set_title(
    "OpenBLAS vs Intel MKL — paired least-squares timings\n"
    "All matrix sizes and conditioning cases; 10 threads, 20 repetitions"
)
ax.grid(axis="y", which="both", alpha=0.25)
ax.legend()
fig.tight_layout()

path2 = OUTPUT / "least_squares_backend_ratio_t10_r20.png"
fig.savefig(path2, dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"Runtime chart: {path1}")
print(f"Backend-ratio chart: {path2}")
print(f"Paired comparisons plotted: {paired_count}")
print(f"Pairs skipped (failure or missing timing): {skipped_count}")
print("\nSummary by method (ratio = OpenBLAS time / MKL time):")

for method in METHODS:
    values = sorted(ratios[method])
    if not values:
        print(f"{LABELS[method]}: no paired timings")
        continue

    mid = len(values) // 2
    median_ratio = (
        values[mid]
        if len(values) % 2
        else (values[mid - 1] + values[mid]) / 2
    )
    mkl_faster = sum(value > 1 for value in values)
    openblas_faster = sum(value < 1 for value in values)

    print(
        f"{LABELS[method]}: n={len(values)}, "
        f"median ratio={median_ratio:.3f}x, "
        f"MKL faster={mkl_faster}, OpenBLAS faster={openblas_faster}"
    )

print(
    "\nInterpretation: ratio > 1 favours MKL; "
    "ratio < 1 favours OpenBLAS."
)
