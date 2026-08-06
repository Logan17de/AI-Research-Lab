from __future__ import annotations

import unittest

import torch

from chat import select_pattern as select_chat_pattern
from evaluate import select_pattern as select_eval_pattern
from init_base import numeric_vocabulary_text
from tiny_pl.model import LearnerLayout, TinyConfig, TinyPatternLM
from tiny_pl.tokenizer import LiveWordTokenizer


class TinyPatternLearnerTests(unittest.TestCase):
    def test_word_tokenizer_splits_only_on_whitespace(self) -> None:
        tokenizer = LiveWordTokenizer.build(["Hello, world! 日本語 テスト 12 + 3"])
        self.assertEqual(
            tokenizer.split("Hello,   world!\n日本語 テスト"),
            ["Hello,", "world!", "日本語", "テスト"],
        )
        self.assertEqual(tokenizer.decode(tokenizer.encode("日本語 テスト")), "日本語 テスト")

    def test_numeric_vocabulary_does_not_encode_arithmetic_examples(self) -> None:
        text = numeric_vocabulary_text(0, 5)
        self.assertEqual(text, "0 1 2 3 4 5")
        self.assertNotIn("+", text)
        with self.assertRaises(ValueError):
            numeric_vocabulary_text(5, 0)

    def test_input_and_output_embeddings_are_untied(self) -> None:
        model = TinyPatternLM(TinyConfig(vocab_size=32))
        self.assertIsNot(model.token_embedding.weight, model.lm_head.weight)
        self.assertNotEqual(
            model.token_embedding.weight.data_ptr(),
            model.lm_head.weight.data_ptr(),
        )

    def test_single_and_all_layer_learners_are_parameter_matched(self) -> None:
        config = TinyConfig(vocab_size=32)
        single = TinyPatternLM(config, LearnerLayout("single", 128))
        single.add_pattern("addition")
        all_layers = TinyPatternLM(config, LearnerLayout("all", 32))
        all_layers.add_pattern("addition")
        single_count = single.learner_parameter_count("addition")
        all_count = all_layers.learner_parameter_count("addition")
        self.assertLess(abs(all_count / single_count - 1.0), 0.001)

    def test_new_learner_has_zero_initial_effect(self) -> None:
        torch.manual_seed(1)
        model = TinyPatternLM(
            TinyConfig(vocab_size=32, dropout=0.0),
            LearnerLayout("all", 8),
        ).eval()
        model.add_pattern("addition")
        inputs = torch.tensor([[1, 4, 5, 2]])
        active_logits = model(inputs)["logits"]
        model.set_active_pattern(None)
        base_logits = model(inputs)["logits"]
        self.assertTrue(torch.equal(active_logits, base_logits))

    def test_base_alias_disables_learner(self) -> None:
        model = TinyPatternLM(TinyConfig(vocab_size=32), LearnerLayout("single", 8))
        model.add_pattern("addition")
        select_chat_pattern(model, "base")
        self.assertIsNone(model.active_pattern)
        select_eval_pattern(model, "addition")
        self.assertEqual(model.active_pattern, "addition")
        select_eval_pattern(model, "none")
        self.assertIsNone(model.active_pattern)

    def test_answer_only_loss_is_finite(self) -> None:
        model = TinyPatternLM(
            TinyConfig(vocab_size=32),
            LearnerLayout("single", 16),
        )
        model.add_pattern("addition")
        inputs = torch.tensor([[1, 5, 6, 7, 2]])
        labels = torch.tensor([[-100, -100, -100, 7, 2]])
        loss = model(inputs, labels=labels)["loss"]
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
