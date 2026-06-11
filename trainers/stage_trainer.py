from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from losses import OntologicalLoss
from utils.metrics import classification_metrics, ontological_consistency


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


class StageTrainer:
    def __init__(self, model, config, ontology, output_dir: str | Path | None = None):
        self.model = model
        self.config = config
        self.ontology = ontology
        requested = str(config.device)
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested)
        self.model.to(self.device)
        self.output_dir = Path(output_dir or config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        training = config.training
        self.criterion = OntologicalLoss(
            ontology,
            alpha_main=float(training.alpha_main),
            beta_sub=float(training.beta_sub),
            lambda_mono=float(training.lambda_mono),
            lambda_dom=float(training.lambda_dom),
            lambda_cons=float(training.lambda_cons),
            lambda_kge=float(training.lambda_kge),
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(training.learning_rate),
            weight_decay=float(training.weight_decay),
        )
        self.history: list[dict] = []

    def pretrain_transe(self, epochs: int | None = None) -> list[float]:
        epochs = int(epochs if epochs is not None else self.config.ontology.transe_epochs)
        triples = self.model.ontology_triples(self.device)
        if not triples.numel() or epochs <= 0:
            return []
        optimizer = torch.optim.Adam(self.model.transe.parameters(), lr=1e-3)
        losses: list[float] = []
        for _ in range(epochs):
            negative = triples.clone()
            replace_head = torch.rand(len(triples), device=self.device) < 0.5
            random_entities = torch.randint(
                0, len(self.ontology.names), (len(triples),), device=self.device
            )
            negative[replace_head, 0] = random_entities[replace_head]
            negative[~replace_head, 2] = random_entities[~replace_head]
            loss = self.model.transe.margin_loss(
                triples,
                negative,
                float(self.config.ontology.transe_margin),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.model.transe.normalize_()
            losses.append(loss.item())
        return losses

    def _set_stage(self, stage: str) -> None:
        requires_grad = stage != "warmup"
        for parameter in self.model.transe.parameters():
            parameter.requires_grad = requires_grad
        if stage == "finetune":
            for group in self.optimizer.param_groups:
                group["lr"] = float(self.config.training.learning_rate) * 0.1

    def _run_epoch(
        self,
        loader,
        training: bool,
        constraint_scale: float = 1.0,
    ) -> dict[str, float]:
        self.model.train(training)
        totals: dict[str, float] = defaultdict(float)
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_nodes: list[torch.Tensor] = []
        count = 0
        for batch in loader:
            batch = move_batch(batch, self.device)
            with torch.set_grad_enabled(training):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    aspect_mask=batch["aspect_mask"],
                    concept_id=batch["concept_id"],
                    main_concept_id=batch["main_concept_id"],
                    dependency=batch["dependency"],
                )
                kge_loss = None
                if training and constraint_scale > 0:
                    triples = self.model.ontology_triples(self.device)
                    negative = triples.clone()
                    negative[:, 2] = torch.randint(
                        0,
                        len(self.ontology.names),
                        (len(triples),),
                        device=self.device,
                    )
                    kge_loss = self.model.transe.margin_loss(
                        triples,
                        negative,
                        float(self.config.ontology.transe_margin),
                    )
                losses = self.criterion(
                    outputs,
                    batch["label"],
                    constraint_scale=constraint_scale,
                    kge_loss=kge_loss,
                )
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    losses["total"].backward()
                    clip_grad_norm_(
                        self.model.parameters(), float(self.config.training.gradient_clip)
                    )
                    self.optimizer.step()
            batch_size = batch["label"].size(0)
            count += batch_size
            for name, value in losses.items():
                totals[name] += value.detach().item() * batch_size
            all_logits.append(outputs["logits"].detach().cpu())
            all_labels.append(batch["label"].detach().cpu())
            all_nodes.append(outputs["node_logits"].detach().cpu())

        if count == 0:
            raise ValueError("The data loader is empty")
        result = {name: value / count for name, value in totals.items()}
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        result.update(classification_metrics(logits, labels))
        result["ontological_consistency"] = ontological_consistency(
            torch.cat(all_nodes), self.ontology
        )
        return result

    def evaluate(self, loader) -> dict[str, float]:
        return self._run_epoch(loader, training=False)

    def fit(self, train_loader, validation_loader=None) -> list[dict]:
        stages = [
            ("warmup", int(self.config.training.warmup_epochs)),
            ("main", int(self.config.training.main_epochs)),
            ("finetune", int(self.config.training.finetune_epochs)),
        ]
        best_f1 = -1.0
        stale = 0
        total_main = max(1, int(self.config.training.main_epochs))
        for stage, epochs in stages:
            self._set_stage(stage)
            for epoch in range(epochs):
                scale = 0.0 if stage == "warmup" else (
                    (epoch + 1) / total_main if stage == "main" else 1.0
                )
                train_metrics = self._run_epoch(
                    train_loader, training=True, constraint_scale=scale
                )
                validation_metrics = (
                    self.evaluate(validation_loader)
                    if validation_loader is not None
                    else train_metrics
                )
                row = {
                    "stage": stage,
                    "epoch": epoch + 1,
                    "constraint_scale": scale,
                    "train": train_metrics,
                    "validation": validation_metrics,
                }
                self.history.append(row)
                score = validation_metrics["macro_f1"]
                if score > best_f1:
                    best_f1, stale = score, 0
                    self.save_checkpoint(self.output_dir / "best_model.pt")
                else:
                    stale += 1
                if stale >= int(self.config.training.patience):
                    self._write_history()
                    return self.history
        self._write_history()
        return self.history

    def _write_history(self) -> None:
        (self.output_dir / "history.json").write_text(
            json.dumps(self.history, indent=2), encoding="utf-8"
        )

    def save_checkpoint(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config.to_dict(),
                "ontology_domain": self.ontology.domain,
                "history": self.history,
            },
            target,
        )
        return target

    def load_checkpoint(self, path: str | Path, load_optimizer: bool = False) -> dict:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        if load_optimizer and "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.history = checkpoint.get("history", [])
        return checkpoint
