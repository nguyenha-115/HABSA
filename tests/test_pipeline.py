from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from config import load_config
from data_loader import OKEHABSADataset, TextPreprocessor, VocabularyTokenizer, collate_oke_habsa
from losses import OntologicalLoss
from models import OKEHABSANet
from models.embedders import TransEEmbedding
from ontology import OntologyManager
from trainers import StageTrainer


SAMPLES = [
    {
        "token": ["The", "battery", "life", "is", "excellent", "."],
        "head": [3, 3, 4, 0, 4, 4],
        "aspects": [
            {"term": ["battery", "life"], "from": 1, "to": 3, "polarity": "positive"}
        ],
    },
    {
        "token": ["The", "touchpad", "is", "not", "responsive", "."],
        "head": [2, 5, 5, 5, 0, 5],
        "aspects": [
            {"term": ["touchpad"], "from": 1, "to": 2, "polarity": "negative"}
        ],
    },
]


class DummyTrainerModel(nn.Module):
    def __init__(self, num_entities: int):
        super().__init__()
        self.text_encoder = nn.Module()
        self.text_encoder.transformer = nn.Sequential(
            nn.Linear(4, 4),
            nn.LayerNorm(4),
        )
        self.classifier = nn.Linear(4, 2)
        self.output_norm = nn.LayerNorm(2)
        self.transe = TransEEmbedding(num_entities, 1, 4)


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(
            overrides={
                "model": {
                    "embedding_dim": 16,
                    "hidden_dim": 32,
                    "ontology_dim": 16,
                    "num_heads": 4,
                    "dropout": 0.0,
                },
                "data": {"max_length": 32},
            }
        )
        self.ontology = OntologyManager("laptop")
        self.tokenizer = VocabularyTokenizer()
        self.tokenizer.fit(SAMPLES, min_frequency=1)
        preprocessor = TextPreprocessor(self.tokenizer, self.ontology, 32)
        self.dataset = OKEHABSADataset(SAMPLES, preprocessor)

    def test_ontology_exports(self):
        self.assertEqual(self.ontology.validate(), [])
        self.assertEqual(self.ontology.map_entity("battery life")[0], "BatteryLifeAspect")
        path = self.ontology.export_owl(Path("outputs/tests/laptop.owl"))
        self.assertTrue(path.exists())
        self.assertIn("BatteryAspect", path.read_text(encoding="utf-8"))

    def test_end_to_end_forward_and_loss(self):
        batch = collate_oke_habsa([self.dataset[0], self.dataset[1]])
        model = OKEHABSANet(self.config, self.ontology, len(self.tokenizer.vocabulary))
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            aspect_mask=batch["aspect_mask"],
            concept_id=batch["concept_id"],
            main_concept_id=batch["main_concept_id"],
            dependency=batch["dependency"],
        )
        self.assertEqual(tuple(outputs["logits"].shape), (2, 5))
        self.assertEqual(
            tuple(outputs["node_logits"].shape), (2, len(self.ontology.names), 5)
        )
        criterion = OntologicalLoss(self.ontology)
        losses = criterion(outputs, batch["label"])
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()

    def test_trainer_optimizer_groups_plm_lr_and_no_decay(self):
        model = DummyTrainerModel(len(self.ontology.names))
        trainer = StageTrainer(
            model,
            self.config,
            self.ontology,
            output_dir=Path("outputs/tests/trainer"),
        )
        parameter_names = {
            id(parameter): name for name, parameter in model.named_parameters()
        }
        grouped_ids = [
            id(parameter)
            for group in trainer.optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(set(grouped_ids), set(parameter_names))

        layer_norm_ids = {
            id(parameter)
            for module in model.modules()
            if isinstance(module, nn.LayerNorm)
            for parameter in module.parameters(recurse=False)
        }
        for group in trainer.optimizer.param_groups:
            for parameter in group["params"]:
                name = parameter_names[id(parameter)]
                expected_lr = (
                    float(self.config.training.plm_learning_rate)
                    if "text_encoder.transformer" in name
                    else float(self.config.training.learning_rate)
                )
                expected_decay = not (
                    name.endswith(".bias") or id(parameter) in layer_norm_ids
                )
                self.assertEqual(group["lr"], expected_lr)
                self.assertEqual(
                    group["weight_decay"],
                    float(self.config.training.weight_decay) if expected_decay else 0.0,
                )

        trainer._set_stage("finetune")
        for group in trainer.optimizer.param_groups:
            self.assertEqual(group["lr"], group["initial_lr"] * 0.1)

    def test_transe_negative_sampling_replaces_head_or_tail(self):
        model = DummyTrainerModel(len(self.ontology.names))
        trainer = StageTrainer(
            model,
            self.config,
            self.ontology,
            output_dir=Path("outputs/tests/trainer"),
        )
        triples = torch.tensor(
            [[0, 0, 1], [1, 0, 2], [2, 0, 3], [3, 0, 4]],
            device=trainer.device,
        )
        with (
            patch(
                "trainers.stage_trainer.torch.rand",
                return_value=torch.tensor(
                    [0.1, 0.9, 0.2, 0.8], device=trainer.device
                ),
            ),
            patch(
                "trainers.stage_trainer.torch.randint",
                return_value=torch.tensor(
                    [4, 5, 6, 7], device=trainer.device
                ),
            ),
        ):
            negative = trainer._sample_negative_triples(triples)

        expected = torch.tensor(
            [[4, 0, 1], [1, 0, 5], [6, 0, 3], [3, 0, 7]],
            device=trainer.device,
        )
        self.assertTrue(torch.equal(negative, expected))


if __name__ == "__main__":
    unittest.main()
