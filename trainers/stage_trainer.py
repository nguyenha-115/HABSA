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
    def __init__(
        self,
        model,
        config,
        ontology,
        output_dir: str | Path | None = None,
        class_weights: torch.Tensor | None = None,
        log_callback=None,
    ):
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
        cfg = config.training
        self.criterion = OntologicalLoss(
            ontology,
            alpha_main=float(cfg.alpha_main),
            beta_sub=float(cfg.beta_sub),
            lambda_mono=float(cfg.lambda_mono),
            lambda_dom=float(cfg.lambda_dom),
            lambda_cons=float(cfg.lambda_cons),
            node_loss_weight=float(cfg.node_loss_weight),
            label_smoothing=float(cfg.label_smoothing),
            class_weights=(
                class_weights.to(self.device)
                if bool(cfg.class_weighting) and class_weights is not None
                else None
            ),
        ).to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.history: list[dict] = []
        self.log_callback = log_callback or (lambda message: print(message, flush=True))

    def _log(self, message: str) -> None:
        self.log_callback(message)

    def _build_scheduler(self):
        cfg = self.config.training
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=float(cfg.scheduler_factor),
            patience=int(cfg.scheduler_patience),
            min_lr=float(cfg.min_learning_rate),
        )

    def _build_optimizer(self):
        cfg = self.config.training
        groups = defaultdict(list)
        no_decay_ids = {
            id(parameter)
            for module in self.model.modules()
            if isinstance(module, torch.nn.LayerNorm)
            for parameter in module.parameters(recurse=False)
        }
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                is_plm = "text_encoder.transformer" in name
                use_decay = not (name.endswith(".bias") or id(parameter) in no_decay_ids)
                groups[(is_plm, use_decay)].append(parameter)
        parameters = []
        for (is_plm, use_decay), values in groups.items():
            lr = float(cfg.plm_learning_rate if is_plm else cfg.learning_rate)
            parameters.append(
                {
                    "params": values,
                    "lr": lr,
                    "initial_lr": lr,
                    "weight_decay": float(cfg.weight_decay) if use_decay else 0.0,
                }
            )
        return torch.optim.AdamW(parameters)

    def _sample_negative_triples(self, triples: torch.Tensor) -> torch.Tensor:
        positives = {tuple(row) for row in triples.detach().cpu().tolist()}
        negative = triples.clone()
        for index in range(len(triples)):
            replace_head = bool(torch.rand((), device=self.device) < 0.5)
            for _ in range(32):
                entity = int(
                    torch.randint(
                        len(self.ontology.names), (), device=self.device
                    ).item()
                )
                candidate = triples[index].clone()
                candidate[0 if replace_head else 2] = entity
                if tuple(candidate.detach().cpu().tolist()) not in positives:
                    negative[index] = candidate
                    break
        return negative

    def pretrain_transe(self, epochs: int | None = None) -> list[float]:
        epochs = int(
            self.config.ontology.transe_epochs if epochs is None else epochs
        )
        triples = self.model.ontology_triples(self.device)
        optimizer = torch.optim.Adam(self.model.transe.parameters(), lr=1e-3)
        losses = []
        self._log(
            f"[TRANSE] start epochs={epochs} triples={len(triples)} "
            f"margin={float(self.config.ontology.transe_margin):.3f}"
        )
        for epoch in range(max(0, epochs)):
            negative = self._sample_negative_triples(triples)
            loss = self.model.transe.margin_loss(
                triples, negative, float(self.config.ontology.transe_margin)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            self.model.transe.normalize_()
            losses.append(float(loss.item()))
            if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % 10 == 0:
                self._log(
                    f"[TRANSE] epoch={epoch + 1:03d}/{epochs:03d} "
                    f"loss={losses[-1]:.6f}"
                )
        torch.save(
            {
                "entity_state": self.model.transe.entity.state_dict(),
                "relation_state": self.model.transe.relation.state_dict(),
                "losses": losses,
            },
            self.output_dir / "transe_best.pt",
        )
        self.model.freeze_transe()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self._log(
            f"[TRANSE] completed final_loss={losses[-1]:.6f}"
            if losses
            else "[TRANSE] skipped"
        )
        return losses

    def _set_stage(self, stage: str) -> None:
        scale = 0.1 if stage == "finetune" else 1.0
        for group in self.optimizer.param_groups:
            group["lr"] = group["initial_lr"] * scale

    def _run_epoch(self, loader, training: bool, constraint_scale: float = 1.0):
        self.model.train(training)
        totals = defaultdict(float)
        logits, labels, node_logits = [], [], []
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
                losses = self.criterion(
                    outputs,
                    batch["label"],
                    batch["node_targets"],
                    constraint_scale,
                )
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError("NaN/Inf encountered in training loss")
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    losses["total"].backward()
                    clip_grad_norm_(
                        self.model.parameters(),
                        float(self.config.training.gradient_clip),
                    )
                    self.optimizer.step()
            batch_size = len(batch["label"])
            count += batch_size
            for name, value in losses.items():
                totals[name] += float(value.detach().item()) * batch_size
            logits.append(outputs["logits"].detach().cpu())
            labels.append(batch["label"].detach().cpu())
            node_logits.append(outputs["node_logits"].detach().cpu())
        if not count:
            raise ValueError("The data loader is empty")
        result = {name: value / count for name, value in totals.items()}
        result.update(classification_metrics(torch.cat(logits), torch.cat(labels)))
        result["ontological_consistency"] = ontological_consistency(
            torch.cat(node_logits), self.ontology
        )
        return result

    def evaluate(self, loader):
        return self._run_epoch(loader, False)

    def fit(self, train_loader, validation_loader):
        stages = [
            ("warmup", int(self.config.training.warmup_epochs)),
            ("main", int(self.config.training.main_epochs)),
            ("finetune", int(self.config.training.finetune_epochs)),
        ]
        best_f1, stale, gap_stale = -1.0, 0, 0
        main_epochs = max(1, int(self.config.training.main_epochs))
        total_epochs = sum(epochs for _, epochs in stages)
        global_epoch = 0
        stop = False
        stop_reason = ""
        for stage, epochs in stages:
            self._set_stage(stage)
            for epoch in range(epochs):
                global_epoch += 1
                constraint_scale = (
                    0.0
                    if stage == "warmup"
                    else (epoch + 1) / main_epochs
                    if stage == "main"
                    else 1.0
                )
                train = self._run_epoch(train_loader, True, constraint_scale)
                validation = self.evaluate(validation_loader)
                gap = float(train["macro_f1"]) - float(validation["macro_f1"])
                row = {
                    "stage": stage,
                    "epoch": epoch + 1,
                    "constraint_scale": constraint_scale,
                    "generalization_gap": gap,
                    "train": train,
                    "validation": validation,
                }
                self.history.append(row)
                score = float(validation["macro_f1"])
                self.scheduler.step(score)
                improved = score > best_f1 + float(self.config.training.min_delta)
                if improved:
                    best_f1, stale = score, 0
                    self.save_checkpoint(self.output_dir / "best_model.pt")
                else:
                    stale += 1
                if gap > float(self.config.training.max_generalization_gap):
                    gap_stale += 1
                else:
                    gap_stale = 0
                self.save_checkpoint(self.output_dir / "last_model.pt")
                self._write_history()
                learning_rates = [group["lr"] for group in self.optimizer.param_groups]
                self._log(
                    f"[EPOCH] {global_epoch:03d}/{total_epochs:03d} "
                    f"stage={stage} local={epoch + 1:02d}/{epochs:02d} "
                    f"scale={constraint_scale:.3f} lr={max(learning_rates):.2e} "
                    f"train_loss={float(train['total']):.4f} "
                    f"train_f1={float(train['macro_f1']):.4f} "
                    f"val_loss={float(validation['total']):.4f} "
                    f"val_f1={score:.4f} val_acc={float(validation['accuracy']):.4f} "
                    f"val_mae={float(validation['mae']):.4f} "
                    f"val_oc={float(validation['ontological_consistency']):.4f} "
                    f"gap={gap:.4f} best_f1={best_f1:.4f} "
                    f"best={'yes' if improved else 'no'} stale={stale} "
                    f"gap_stale={gap_stale}"
                )
                if stale >= int(self.config.training.patience):
                    stop = True
                    stop_reason = "early_stopping"
                elif gap_stale >= int(self.config.training.gap_patience):
                    stop = True
                    stop_reason = "generalization_gap"
                if stop:
                    self._log(
                        f"[TRAIN] stopped reason={stop_reason} "
                        f"epoch={global_epoch}/{total_epochs} best_val_f1={best_f1:.4f}"
                    )
                    break
            if stop:
                break
        if not stop:
            self._log(
                f"[TRAIN] completed epochs={global_epoch}/{total_epochs} "
                f"best_val_f1={best_f1:.4f}"
            )
        return self.history

    def _write_history(self):
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

    def load_checkpoint(self, path: str | Path, load_optimizer: bool = False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.history = checkpoint.get("history", [])
        return checkpoint
