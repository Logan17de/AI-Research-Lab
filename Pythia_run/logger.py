from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class MetricLogger:
    def __init__(self, run_dir: str, *, resume_step: int | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self.plot_path = self.run_dir / "metrics.png"
        self.fieldnames = [
            "step",
            "epoch",
            "progress_pct",
            "loss",
            "ppl",
            "eval_assistant_ppl",
            "eval_combined_ppl",
            "lr",
            "modifier_lr",
            "lora_lr",
            "plastic_lr",
            "plastic_layernorm_lr",
            "plastic_drift",
            "tokens_per_sec",
            "target_tokens_per_sec",
            "target_tokens",
            "samples_processed",
            "tokens_processed",
            "target_tokens_processed",
            "effective_epochs",
            "micro_batch_size",
            "grad_accum_steps",
            "effective_batch_size",
            "elapsed_seconds",
            "eos_accuracy",
            "first_answer_accuracy",
            "target_top1_accuracy",
            "base_logit_norm",
            "output_mod_logit_norm",
            "output_mod_logit_ratio",
            "checkpoint_path",
            "gpu",
        ]
        if resume_step is None:
            self._write_rows([])
        else:
            self._trim_to_step(resume_step)

    def _write_rows(self, rows: list[dict[str, Any]]) -> None:
        with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows({name: row.get(name, "") for name in self.fieldnames} for row in rows)

    def _trim_to_step(self, resume_step: int) -> None:
        if not self.csv_path.exists():
            self._write_rows([])
            return
        retained: list[dict[str, Any]] = []
        with open(self.csv_path, "r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    step = int(float(row.get("step", "")))
                except (TypeError, ValueError):
                    continue
                if step <= resume_step:
                    retained.append(row)
        self._write_rows(retained)

    def log(self, metrics: dict[str, Any]) -> None:
        row = {name: metrics.get(name, "") for name in self.fieldnames}
        with open(self.csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(row)

    def plot(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        rows: list[dict[str, str]] = []
        with open(self.csv_path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows.extend(reader)
        if not rows:
            return

        steps = [float(row["step"]) for row in rows if row.get("step")]
        if not steps:
            return

        def values(name: str) -> list[float]:
            out: list[float] = []
            for row in rows:
                try:
                    out.append(float(row[name]))
                except (KeyError, TypeError, ValueError):
                    out.append(float("nan"))
            return out

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(steps, values("loss"), label="train loss")
        axes[0, 0].set_title("Loss")
        axes[0, 1].plot(steps, values("ppl"), label="train ppl")
        axes[0, 1].plot(steps, values("eval_assistant_ppl"), label="eval assistant ppl")
        axes[0, 1].plot(steps, values("eval_combined_ppl"), label="eval combined ppl")
        axes[0, 1].set_title("Perplexity")
        axes[0, 1].legend()
        axes[1, 0].plot(steps, values("tokens_per_sec"), label="tok/s")
        axes[1, 0].set_title("Throughput")
        axes[1, 1].plot(steps, values("lr"), label="base lr")
        axes[1, 1].plot(steps, values("modifier_lr"), label="modifier lr")
        axes[1, 1].plot(steps, values("lora_lr"), label="lora lr")
        axes[1, 1].plot(steps, values("plastic_lr"), label="plastic lr")
        axes[1, 1].plot(steps, values("plastic_layernorm_lr"), label="plastic norm lr")
        axes[1, 1].set_title("Learning Rates")
        axes[1, 1].legend()

        for axis in axes.flat:
            axis.set_xlabel("step")
            axis.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=160)
        plt.close(fig)
