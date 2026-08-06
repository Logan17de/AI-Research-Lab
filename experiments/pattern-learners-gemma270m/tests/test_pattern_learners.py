from __future__ import annotations

import unittest

import torch
from torch import nn

from pattern_learners.learner import (
    LearnerLayout,
    PatternLearnerSystem,
    ResidualPatternLearner,
    matched_all_layer_dim,
)
from pattern_learners.optimization import (
    LearningRates,
    TrainabilityConfig,
    build_optimizer,
    parameter_partitions,
)


class FakeLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.linear.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states)


class FakeInnerModel(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer(hidden_size) for _ in range(num_layers)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class FakeModel(nn.Module):
    def __init__(self, hidden_size: int = 8, num_layers: int = 4, vocab_size: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.model = FakeInnerModel(hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.model(self.embedding(input_ids)))


class PatternLearnerTests(unittest.TestCase):
    def test_zero_effect_initialization(self) -> None:
        learner = ResidualPatternLearner(hidden_size=8, bottleneck_dim=4)
        hidden = torch.randn(2, 3, 8)
        self.assertTrue(torch.equal(learner(hidden), hidden))

    def test_single_layer_hook_only_changes_selected_layer(self) -> None:
        model = FakeModel()
        system = PatternLearnerSystem(8, 4, LearnerLayout("single", 4, single_layer=2))
        system.add_pattern("addition")
        system.attach(model)

        module = system.patterns["addition"].modules_by_layer["2"]
        nn.init.constant_(module.down.weight, 0.1)
        nn.init.constant_(module.up.weight, 0.1)
        inputs = torch.tensor([[1, 2, 3]])
        system.set_active_pattern(None)
        baseline = model(inputs)
        system.set_active_pattern("addition")
        changed = model(inputs)
        self.assertFalse(torch.equal(baseline, changed))

    def test_all_layer_parameter_match(self) -> None:
        single = PatternLearnerSystem(640, 18, LearnerLayout("single", 640))
        single.add_pattern("math")
        all_layers = PatternLearnerSystem(
            640,
            18,
            LearnerLayout("all", matched_all_layer_dim(640, 18)),
        )
        all_layers.add_pattern("math")
        ratio = all_layers.learner_parameter_count() / single.learner_parameter_count()
        self.assertLess(abs(ratio - 1.0), 0.02)

    def test_independent_freezing_and_learning_rates(self) -> None:
        model = FakeModel()
        system = PatternLearnerSystem(8, 4, LearnerLayout("single", 4))
        system.add_pattern("math")
        system.attach(model)

        optimizer, summary = build_optimizer(
            model,
            system,
            TrainabilityConfig(
                freeze_embeddings=False,
                freeze_backbone=True,
                freeze_learner=False,
            ),
            LearningRates(embedding=1e-6, backbone=2e-6, learner=3e-4),
        )
        self.assertFalse(summary["embedding"]["frozen"])
        self.assertTrue(summary["backbone"]["frozen"])
        self.assertFalse(summary["learner"]["frozen"])
        self.assertEqual(
            {group["group_name"] for group in optimizer.param_groups},
            {"embedding", "learner"},
        )
        lr_map = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
        self.assertEqual(lr_map["embedding"], 1e-6)
        self.assertEqual(lr_map["learner"], 3e-4)

    def test_tied_embedding_is_not_duplicated(self) -> None:
        model = FakeModel()
        system = PatternLearnerSystem(8, 4, LearnerLayout("single", 4))
        system.add_pattern("math")
        system.attach(model)
        partitions = parameter_partitions(model, system)
        embedding_ids = [id(parameter) for parameter in partitions["embedding"]]
        self.assertEqual(len(embedding_ids), len(set(embedding_ids)))
        self.assertEqual(len(embedding_ids), 1)

    def test_new_named_pattern_is_independent(self) -> None:
        system = PatternLearnerSystem(8, 4, LearnerLayout("single", 4))
        system.add_pattern("addition")
        system.add_pattern("multiplication")
        addition_ids = {id(parameter) for parameter in system.patterns["addition"].parameters()}
        multiplication_ids = {
            id(parameter) for parameter in system.patterns["multiplication"].parameters()
        }
        self.assertFalse(addition_ids & multiplication_ids)


if __name__ == "__main__":
    unittest.main()
