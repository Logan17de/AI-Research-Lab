from __future__ import annotations

from convert_sharegpt_to_qa import convert_records, extract_direct_answer


def test_extract_direct_answer_discards_thinking_and_tags() -> None:
    answer = (
        "<thinking>private reasoning that must not enter training</thinking> "
        "<output>  The direct answer.\nWith a second sentence.  </output>"
    )
    assert extract_direct_answer(answer) == "The direct answer. With a second sentence."


def test_extract_direct_answer_stops_at_first_output_close() -> None:
    answer = (
        "<thinking>reasoning</thinking><output>Direct answer.</output>"
        "<thinking><output>reflection that must be discarded</output></thinking>"
    )
    assert extract_direct_answer(answer) == "Direct answer."


def test_conversion_accepts_malformed_thinking_when_output_is_complete() -> None:
    records = [
        {
            "conversations": [
                {"from": "human", "value": "What is two plus two?"},
                {"from": "gpt", "value": "broken thinking <output>Four.</output>"},
            ]
        }
    ]
    converted, skipped = convert_records(records)
    assert converted == [("What is two plus two?", "Four.")]
    assert skipped == []


def test_conversion_rejects_missing_output() -> None:
    records = [
        {
            "conversations": [
                {"from": "human", "value": "Question"},
                {"from": "gpt", "value": "<thinking>reasoning only</thinking>"},
            ]
        }
    ]
    converted, skipped = convert_records(records)
    assert converted == []
    assert skipped[0]["reason"] == "missing, incomplete, or empty output section"
