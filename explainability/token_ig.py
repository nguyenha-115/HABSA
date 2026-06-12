from __future__ import annotations

import torch


class TokenIntegratedGradients:
    def __init__(self, model, steps: int = 50):
        self.model = model
        self.steps = steps

    def attribute(self, batch: dict, target_class=None) -> dict:
        self.model.eval()
        embeddings = self.model.text_encoder.embed(batch["input_ids"]).detach()
        baseline = torch.zeros_like(embeddings)
        if target_class is None:
            with torch.no_grad():
                target_class = self.model(**batch)["logits"].argmax(-1)
        if not isinstance(target_class, torch.Tensor):
            target_class = torch.full(
                (embeddings.size(0),),
                int(target_class),
                dtype=torch.long,
                device=embeddings.device,
            )
        gradients = []
        for alpha in torch.linspace(
            1.0 / self.steps, 1.0, self.steps, device=embeddings.device
        ):
            scaled = (baseline + alpha * (embeddings - baseline)).detach()
            scaled.requires_grad_(True)
            output = self.model.forward_from_embeddings(
                scaled,
                batch["attention_mask"],
                batch["aspect_mask"],
                batch["concept_id"],
                batch.get("main_concept_id"),
                batch.get("dependency"),
            )
            selected = output["logits"].gather(1, target_class.unsqueeze(1)).sum()
            gradients.append(torch.autograd.grad(selected, scaled)[0].detach())
        integrated = (embeddings - baseline) * torch.stack(gradients).mean(0)
        scores = integrated.sum(-1) * batch["attention_mask"]
        normalized = scores / scores.abs().amax(1, keepdim=True).clamp_min(1e-8)
        return {
            "attributions": integrated,
            "token_scores": scores,
            "normalized_scores": normalized,
            "target_class": target_class,
        }
