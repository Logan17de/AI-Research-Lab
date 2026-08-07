from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class CategoryTotals:
    count: int = 0
    nll_sum: float = 0.0
    top1_correct: int = 0
    top5_correct: int = 0
    rank_sum: float = 0.0
    probability_sum: float = 0.0

    def add(self, nll: torch.Tensor, correct: torch.Tensor, top5: torch.Tensor, ranks: torch.Tensor) -> None:
        if nll.numel() == 0:
            return
        self.count += int(nll.numel())
        self.nll_sum += float(nll.double().sum().item())
        self.top1_correct += int(correct.sum().item())
        self.top5_correct += int(top5.sum().item())
        self.rank_sum += float(ranks.double().sum().item())
        self.probability_sum += float(torch.exp(-nll.double()).sum().item())

    def metrics(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"count": 0, "nll_sum": 0.0, "nll": float("nan"), "ppl": float("nan"), "top1": float("nan"), "top5": float("nan"), "rank": float("nan"), "prob": float("nan")}
        mean_nll = self.nll_sum / self.count
        return {
            "count": self.count,
            "nll_sum": self.nll_sum,
            "nll": mean_nll,
            "ppl": math.exp(min(mean_nll, 20.0)),
            "top1": self.top1_correct / self.count,
            "top5": self.top5_correct / self.count,
            "rank": self.rank_sum / self.count,
            "prob": self.probability_sum / self.count,
        }


@dataclass
class EvalTotals:
    reasoning_open_ids: tuple[tuple[int, ...], ...] = ()
    reasoning_close_ids: tuple[tuple[int, ...], ...] = ()
    final_open_ids: tuple[tuple[int, ...], ...] = ()
    final_close_ids: tuple[tuple[int, ...], ...] = ()
    categories: dict[str, CategoryTotals] = field(
        default_factory=lambda: {
            name: CategoryTotals()
            for name in (
                "combined", "assistant_content", "first_answer", "remaining_content",
                "reasoning", "final_answer", "final_answer_first", "eos",
            )
        }
    )
    first_eos_probability_sum: float = 0.0
    first_eos_top1: int = 0
    first_count: int = 0

    @staticmethod
    def _find_any(sequence: list[int], variants: tuple[tuple[int, ...], ...], start: int = 0):
        for position in range(max(start, 0), len(sequence)):
            for variant in variants:
                end = position + len(variant)
                if variant and tuple(sequence[position:end]) == variant:
                    return position, end
        return None

    def _reasoning_final_masks(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reasoning = torch.zeros_like(input_ids, dtype=torch.bool, device="cpu")
        final = torch.zeros_like(input_ids, dtype=torch.bool, device="cpu")
        if not all((self.reasoning_open_ids, self.reasoning_close_ids, self.final_open_ids, self.final_close_ids)):
            return reasoning.to(input_ids.device), final.to(input_ids.device)
        for row, sequence in enumerate(input_ids.detach().cpu().tolist()):
            reasoning_open = self._find_any(sequence, self.reasoning_open_ids)
            if reasoning_open is None:
                continue
            reasoning_close = self._find_any(sequence, self.reasoning_close_ids, reasoning_open[1])
            if reasoning_close is None:
                continue
            final_open = self._find_any(sequence, self.final_open_ids, reasoning_close[1])
            if final_open is None:
                continue
            final_close = self._find_any(sequence, self.final_close_ids, final_open[1])
            if final_close is None:
                continue
            reasoning[row, reasoning_open[1] : reasoning_close[0]] = True
            final[row, final_open[1] : final_close[0]] = True
        return reasoning.to(input_ids.device), final.to(input_ids.device)

    def update(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        assistant_id: int,
        eos_id: int,
    ) -> None:
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        current_tokens = input_ids[:, :-1]
        flat_nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)
        predictions = shift_logits.argmax(dim=-1)
        correct = predictions.eq(shift_labels)
        safe_labels = shift_labels.clamp_min(0)
        gold_scores = shift_logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        ranks = shift_logits.gt(gold_scores.unsqueeze(-1)).sum(dim=-1).add(1)
        top5 = shift_logits.topk(min(5, shift_logits.size(-1)), dim=-1).indices.eq(safe_labels.unsqueeze(-1)).any(dim=-1)
        supervised = shift_labels.ne(-100)
        eos = shift_labels.eq(eos_id)
        content = supervised & ~eos
        previous_supervised = torch.cat(
            [torch.zeros_like(supervised[:, :1]), supervised[:, :-1]], dim=1
        )
        first = content & ~previous_supervised
        reasoning_positions, final_positions = self._reasoning_final_masks(input_ids)
        reasoning = reasoning_positions[:, 1:] & supervised
        final = final_positions[:, 1:] & supervised
        previous_final = torch.cat([torch.zeros_like(final[:, :1]), final[:, :-1]], dim=1)
        final_first = final & ~previous_final
        if first.any():
            first_logits = shift_logits[first]
            self.first_eos_probability_sum += float(first_logits.softmax(dim=-1)[:, eos_id].double().sum().item())
            self.first_eos_top1 += int(first_logits.argmax(dim=-1).eq(eos_id).sum().item())
            self.first_count += int(first.sum().item())
        masks = {
            "combined": supervised,
            "assistant_content": content,
            "first_answer": first,
            "remaining_content": content & ~first,
            "reasoning": reasoning,
            "final_answer": final,
            "final_answer_first": final_first,
            "eos": eos,
        }
        for name, mask in masks.items():
            self.categories[name].add(
                flat_nll[mask].detach().cpu(),
                correct[mask].detach().cpu(),
                top5[mask].detach().cpu(),
                ranks[mask].detach().cpu(),
            )

    def metrics(self) -> dict[str, dict[str, float | int]]:
        result = {name: totals.metrics() for name, totals in self.categories.items()}
        result["first_answer"]["eos_prob"] = (
            self.first_eos_probability_sum / self.first_count if self.first_count else float("nan")
        )
        result["first_answer"]["eos_top1"] = self.first_eos_top1 / self.first_count if self.first_count else float("nan")
        return result
