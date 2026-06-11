from __future__ import annotations

import torch


class TokenIntegratedGradients:
    def __init__(self, model, steps: int = 50):
        self.model = model
        self.steps = steps

    def attribute(
        self,
        batch: dict,
        target_class: int | torch.Tensor | None = None,
    ) -> dict:
        self.model.eval()
        input_ids = batch["input_ids"]
        embeddings = self.model.text_encoder.embed(input_ids).detach()
        baseline = torch.zeros_like(embeddings)
        gradients: list[torch.Tensor] = []

        if target_class is None:
            with torch.no_grad():
                prediction = self.model(
                    input_ids=input_ids,
                    attention_mask=batch["attention_mask"],
                    aspect_mask=batch["aspect_mask"],
                    concept_id=batch["concept_id"],
                    main_concept_id=batch.get("main_concept_id"),
                    dependency=batch.get("dependency"),
                )["logits"]
                target_class = prediction.argmax(dim=-1)
        if not isinstance(target_class, torch.Tensor):
            target_class = torch.full(
                (input_ids.size(0),), target_class, device=input_ids.device, dtype=torch.long
            )

        for alpha in torch.linspace(
            1.0 / self.steps, 1.0, self.steps, device=input_ids.device
        ):
            scaled = (baseline + alpha * (embeddings - baseline)).detach()
            scaled.requires_grad_(True)
            output = self.model.forward_from_embeddings(
                scaled,
                attention_mask=batch["attention_mask"],
                aspect_mask=batch["aspect_mask"],
                concept_id=batch["concept_id"],
                main_concept_id=batch.get("main_concept_id"),
                dependency=batch.get("dependency"),
            )
            selected = output["logits"].gather(1, target_class.unsqueeze(1)).sum()
            gradient = torch.autograd.grad(selected, scaled)[0]
            gradients.append(gradient.detach())

        integrated = (embeddings - baseline) * torch.stack(gradients).mean(dim=0)
        token_scores = integrated.sum(dim=-1)
        token_scores = token_scores * batch["attention_mask"].to(token_scores.dtype)
        scale = token_scores.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        normalized = token_scores / scale
        return {
            "attributions": integrated,
            "token_scores": token_scores,
            "normalized_scores": normalized,
            "target_class": target_class,
        }

    def explain_tokens(self, batch: dict, tokens: list[list[str]]) -> list[list[dict]]:
        result = self.attribute(batch)
        explanations: list[list[dict]] = []
        for row, row_tokens in enumerate(tokens):
            explanations.append(
                [
                    {
                        "token": token,
                        "score": float(result["normalized_scores"][row, index].item()),
                    }
                    for index, token in enumerate(row_tokens)
                ]
            )
        return explanations
