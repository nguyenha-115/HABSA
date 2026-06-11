from __future__ import annotations

import unittest
from pathlib import Path

import torch

from config import load_config
from data_loader import OKEHABSADataset, TextPreprocessor, VocabularyTokenizer, collate_oke_habsa
from losses import OntologicalLoss
from models import OKEHABSANet
from ontology import OntologyManager


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


if __name__ == "__main__":
    unittest.main()
