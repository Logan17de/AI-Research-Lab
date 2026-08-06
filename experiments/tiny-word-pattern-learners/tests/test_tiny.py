from __future__ import annotations

import unittest

import torch

from chat import select_pattern as select_chat_pattern
from evaluate import select_pattern as select_eval_pattern
from generate_arithmetic_data import build_groups, split_groups
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

    def test_controlled_split_has_no_commuted_pair_leakage(self) -> None:
        train, validation = split_groups(
            build_groups("addition", 0, 9),
            validation_ratio=0.2,
            seed=42,
        )
        train_prompts = {row["prompt"] for row in train}
        train_answers = {row["answer"] for row in train}
        for row in validation:
            tokens = row["prompt"].split()
            reverse = " ".join([tokens[0], tokens[1], tokens[4], tokens[3], tokens[2], tokens[5]])
            self.assertNotIn(reverse, train_prompts)
            self.assertIn(row["answer"], train_answers)

    def test_input_and_output_embeddings_are_untied(self) -> None:
        model = TinyPatternLM(TinyConfig(vocab_size=32))
        self.assertIsNot(model.token_embedding.weight, model.lm_head.weight)
        self.assertNotEqual(
            model.token_embedding.weight.data_ptr(),
            model.lm_head.weight.data_ptr(),
        )

    def test_three_new_learner_cores_are_parameter_matched(self) -> None:
        config = TinyConfig(vocab_size=128)
        layouts = {
            "standalone": LearnerLayout(
                mode="all",
                bottleneck_dim=32,
                kind="standalone",
                standalone_layers=1,
                standalone_ffn_size=32,
            ),
            "pre_activation": LearnerLayout(
                mode="all",
                bottleneck_dim=30,
                kind="pre_activation",
            ),
            "post_activation": LearnerLayout(
                mode="all",
                bottleneck_dim=19,
                kind="post_activation",
            ),
        }
        breakdowns = {}
        for name, layout in layouts.items():
            model = TinyPatternLM(config, layout)
            model.add_pattern("addition")
            breakdowns[name] = model.pattern_parameter_breakdown("addition")

        cores = [breakdown["core"] for breakdown in breakdowns.values()]
        heads = [breakdown["head"] for breakdown in breakdowns.values()]
        self.assertLess(max(cores) / min(cores) - 1.0, 0.02)
        self.assertEqual(len(set(heads)), 1)

    def test_pre_and_post_adjustments_start_at_zero(self) -> None:
        config = TinyConfig(vocab_size=32)
        for kind, dimension in (("pre_activation", 30), ("post_activation", 19)):
            model = TinyPatternLM(
                config,
                LearnerLayout(mode="all", bottleneck_dim=dimension, kind=kind),
            )
            model.add_pattern("addition")
            pattern = model.patterns["addition"]
            modules = (
                pattern.pre_adjustments.values()
                if kind == "pre_activation"
                else pattern.post_adjustments.values()
            )
            for module in modules:
                self.assertEqual(torch.count_nonzero(module.up.weight).item(), 0)

    def test_legacy_residual_learner_remains_zero_effect(self) -> None:
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

    def test_all_three_variants_produce_valid_logits(self) -> None:
        config = TinyConfig(vocab_size=64)
        layouts = (
            LearnerLayout(mode="all", bottleneck_dim=32, kind="standalone"),
            LearnerLayout(mode="all", bottleneck_dim=30, kind="pre_activation"),
            LearnerLayout(mode="all", bottleneck_dim=19, kind="post_activation"),
        )
        inputs = torch.tensor([[1, 4, 5, 2]])
        for layout in layouts:
            model = TinyPatternLM(config, layout)
            model.add_pattern("addition")
            logits = model(inputs)["logits"]
            self.assertEqual(tuple(logits.shape), (1, 4, 64))
            self.assertTrue(torch.isfinite(logits).all())

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
            LearnerLayout(mode="all", bottleneck_dim=19, kind="post_activation"),
        )
        model.add_pattern("addition")
        inputs = torch.tensor([[1, 5, 6, 7, 2]])
        labels = torch.tensor([[-100, -100, -100, 7, 2]])
        loss = model(inputs, labels=labels)["loss"]
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
