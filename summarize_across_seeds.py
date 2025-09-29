"""
Summarize final centralized metrics across multiple seeded runs.

Reads metrics/seed_*/centralized_metrics.csv, takes the last row from each,
aggregates selected metrics across seeds, and writes summary CSV/Markdown to:
- metrics/summary_across_seeds.csv
- metrics/summary_across_seeds.md

Usage:
  python summarize_across_seeds.py

No external dependencies required.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
METRICS_DIR = ROOT / "metrics"

# Metrics to aggregate from centralized_metrics.csv final row of each seed
METRICS_TO_AGG = [
    "central_test_accuracy",
    "central_test_loss",
]

# t-critical values for 95% two-sided CI (alpha=0.05) at df=1..30 (0.975 quantile)
T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

def read_final_row(csv_path: Path) -> Dict[str, float] | None:
    if not csv_path.exists():
        return None
    try:
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            last = None
            for row in reader:
                last = row
            if last is None:
                return None
            # Convert numeric fields when possible
            out = {}
            for k, v in last.items():
                try:
                    out[k] = float(v)
                except (ValueError, TypeError):
                    # keep as-is if not numeric
                    pass
            return out
    except Exception:
        return None


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def sample_std(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    m = mean(values)
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    return math.sqrt(var)


def ci95(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    m = mean(values)
    s = sample_std(values)
    if n == 1 or math.isnan(s):
        return (m, m)
    df = n - 1
    tcrit = T_CRIT_95.get(df, 1.96)  # fallback to normal approx for df>30
    half = tcrit * s / math.sqrt(n)
    return (m - half, m + half)


def main() -> None:
    seed_dirs = sorted([p for p in METRICS_DIR.glob("seed_*") if p.is_dir()])
    if not seed_dirs:
        print(f"No seed_* directories found under {METRICS_DIR}")
        return

    # Collect final rows per seed
    finals: List[Dict[str, float]] = []
    used_seeds: List[str] = []
    for sd in seed_dirs:
        path = sd / "centralized_metrics.csv"
        last = read_final_row(path)
        if last is not None:
            finals.append(last)
            used_seeds.append(sd.name)

    if not finals:
        print("No centralized_metrics.csv found with data in any seed directory.")
        return

    # Aggregate requested metrics
    summary_rows = []
    for metric in METRICS_TO_AGG:
        vals = []
        for row in finals:
            if metric in row and isinstance(row[metric], float):
                vals.append(row[metric])
        if not vals:
            continue
        m = mean(vals)
        s = sample_std(vals)
        lo, hi = ci95(vals)
        summary_rows.append({
            "metric": metric,
            "mean": m,
            "std": s,
            "ci95_low": lo,
            "ci95_high": hi,
            "n": len(vals),
        })

    # Write CSV
    out_csv = METRICS_DIR / "summary_across_seeds.csv"
    with out_csv.open("w", newline="") as f:
        fieldnames = ["metric", "mean", "std", "ci95_low", "ci95_high", "n"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"Wrote {out_csv.relative_to(ROOT)}")

    # Write Markdown for easy pasting
    out_md = METRICS_DIR / "summary_across_seeds.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write("| Metric | Mean | Std | 95% CI low | 95% CI high | N |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                f"| {row['metric']} | {row['mean']:.6f} | {row['std']:.6f} | {row['ci95_low']:.6f} | {row['ci95_high']:.6f} | {row['n']} |\n"
            )
    print(f"Wrote {out_md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
