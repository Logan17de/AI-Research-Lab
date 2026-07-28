from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metrics.csv files from comparison runs.")
    parser.add_argument("--run_root", type=str, default="runs/ultrachat_compare")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def values(rows: list[dict[str, str]], name: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        try:
            out.append(float(row[name]))
        except (KeyError, TypeError, ValueError):
            out.append(float("nan"))
    return out


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    output = Path(args.output) if args.output else run_root / "comparison_metrics.png"

    import matplotlib.pyplot as plt

    runs: list[tuple[str, list[dict[str, str]]]] = []
    for metrics_path in sorted(run_root.glob("*/metrics.csv")):
        rows = read_metrics(metrics_path)
        if rows:
            runs.append((metrics_path.parent.name, rows))
    if not runs:
        raise FileNotFoundError(f"No metrics.csv files found below {run_root}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for name, rows in runs:
        steps = values(rows, "step")
        axes[0, 0].plot(steps, values(rows, "loss"), label=name)
        axes[0, 1].plot(steps, values(rows, "eval_assistant_ppl"), label=name)
        axes[1, 0].plot(steps, values(rows, "ppl"), label=name)
        axes[1, 1].plot(steps, values(rows, "tokens_per_sec"), label=name)

    axes[0, 0].set_title("Train Loss")
    axes[0, 1].set_title("Assistant-content Eval PPL")
    axes[1, 0].set_title("Train PPL")
    axes[1, 1].set_title("Tokens/sec")
    for axis in axes.flat:
        axis.set_xlabel("step")
        axis.grid(True, alpha=0.25)
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
