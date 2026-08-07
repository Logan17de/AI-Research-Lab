from __future__ import annotations

import torch

from evaluate import tagged_section
from evaluation import EvalTotals


def test_qa_first_answer_and_reasoning_final_categories_are_causally_shifted() -> None:
    token_ids = [
        1, 2,
        10, 11, 12,
        30,
        13, 11, 12,
        10, 14, 12,
        40, 41,
        13, 14, 12,
        0,
    ]
    input_ids = torch.tensor([token_ids])
    labels = torch.tensor([[-100, -100, *token_ids[2:]]])
    logits = torch.full((1, len(token_ids), 50), -8.0)
    for position in range(len(token_ids) - 1):
        target = int(labels[0, position + 1])
        if target >= 0:
            logits[0, position, target] = 8.0

    totals = EvalTotals(
        reasoning_open_ids=((10, 11, 12),),
        reasoning_close_ids=((13, 11, 12),),
        final_open_ids=((10, 14, 12),),
        final_close_ids=((13, 14, 12),),
    )
    totals.update(logits, input_ids, labels, assistant_id=49, eos_id=0)
    metrics = totals.metrics()

    assert metrics["first_answer"]["count"] == 1
    assert metrics["first_answer"]["top1"] == 1.0
    assert metrics["reasoning"]["count"] == 1
    assert metrics["final_answer"]["count"] == 2
    assert metrics["final_answer_first"]["count"] == 1
    assert metrics["final_answer"]["top1"] == 1.0


def test_tagged_section_extracts_final_answer() -> None:
    text = "<thinking>reasoning here</thinking> <output>The final answer.</output>"
    assert tagged_section(text, "thinking") == "reasoning here"
    assert tagged_section(text, "output") == "The final answer."
    assert tagged_section("unfinished", "output") is None
