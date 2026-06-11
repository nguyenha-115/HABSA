from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import Config, apply_dotted_overrides, load_config
from data_loader import (
    INDEX_TO_SCORE,
    OKEHABSADataset,
    TextPreprocessor,
    VocabularyTokenizer,
    collate_oke_habsa,
    load_records,
)
from explainability import ConceptAttribution, StructuralExplainer, TokenIntegratedGradients
from models import OKEHABSANet
from ontology import OntologyManager
from trainers import StageTrainer, move_batch


DATASET_TO_DOMAIN = {
    "laptops": "laptop",
    "restaurants": "restaurant",
    "restaurants16": "restaurant",
    "tweets": "social",
}
CANONICAL_DATASETS = {
    "laptops": "Laptops",
    "restaurants": "Restaurants",
    "restaurants16": "Restaurants16",
    "tweets": "Tweets",
}
SENTIMENT_NAMES = ["very_negative", "negative", "neutral", "positive", "very_positive"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_domain(dataset: str | None, configured: str) -> str:
    return DATASET_TO_DOMAIN.get((dataset or "").lower(), configured)


def canonical_dataset(dataset: str) -> str:
    try:
        return CANONICAL_DATASETS[dataset.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset {dataset!r}; expected one of {list(CANONICAL_DATASETS.values())}"
        ) from exc


def make_loader(dataset, config, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config.training.batch_size),
        shuffle=shuffle,
        num_workers=int(config.data.num_workers),
        collate_fn=collate_oke_habsa,
    )


def split_records(records: list[dict], ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    validation_size = max(1, int(len(indices) * ratio))
    validation = {index for index in indices[:validation_size]}
    return (
        [record for index, record in enumerate(records) if index not in validation],
        [record for index, record in enumerate(records) if index in validation],
    )


def command_build_ontology(args, config) -> None:
    domain = resolve_domain(args.dataset, str(config.ontology.domain))
    ontology = OntologyManager(domain)
    errors = ontology.validate()
    if errors:
        raise RuntimeError("\n".join(errors))
    output = Path(args.output or f"ontology/{domain}_domain.owl")
    ontology.export_owl(output)
    ontology.export_json(output.with_suffix(".json"))
    ontology.export_tsv(output.with_suffix(".tsv"))
    print(json.dumps({"domain": domain, "concepts": len(ontology.names), "owl": str(output)}))


def command_train(args, config) -> None:
    dataset_name = canonical_dataset(args.dataset or str(config.data.domain))
    domain = resolve_domain(dataset_name, str(config.ontology.domain))
    config.data.domain = dataset_name
    config.ontology.domain = domain
    if args.epochs is not None:
        config.training.warmup_epochs = min(1, args.epochs)
        config.training.main_epochs = max(0, args.epochs - 1)
        config.training.finetune_epochs = 0
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    output_dir = Path(args.output_dir or config.output_dir)
    config.output_dir = str(output_dir)
    set_seed(int(config.seed))

    records = load_records(Path(config.data.root) / dataset_name / "train.json")
    if args.max_records is not None:
        records = records[: args.max_records]
    train_records, validation_records = split_records(
        records, float(config.data.validation_ratio), int(config.seed)
    )
    tokenizer = VocabularyTokenizer()
    tokenizer.fit(
        train_records,
        min_frequency=int(config.data.min_frequency),
        max_size=int(config.model.vocab_size),
    )
    ontology = OntologyManager(domain)
    preprocessor = TextPreprocessor(tokenizer, ontology, int(config.data.max_length))
    train_dataset = OKEHABSADataset(
        train_records, preprocessor, float(config.ontology.mapping_threshold)
    )
    validation_dataset = OKEHABSADataset(
        validation_records, preprocessor, float(config.ontology.mapping_threshold)
    )
    model = OKEHABSANet(config, ontology, len(tokenizer.vocabulary))
    trainer = StageTrainer(model, config, ontology, output_dir)
    kge_losses = trainer.pretrain_transe(args.transe_epochs)
    history = trainer.fit(
        make_loader(train_dataset, config, shuffle=True),
        make_loader(validation_dataset, config),
    )
    tokenizer.save(output_dir / "vocab.json")
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    ontology.export_owl(output_dir / f"{domain}_domain.owl")
    summary = {
        "train_aspects": len(train_dataset),
        "validation_aspects": len(validation_dataset),
        "vocabulary": len(tokenizer.vocabulary),
        "transe_final_loss": kge_losses[-1] if kge_losses else None,
        "last_epoch": history[-1] if history else None,
        "checkpoint": str(output_dir / "best_model.pt"),
    }
    print(json.dumps(summary, indent=2))


def load_artifacts(checkpoint_path: str | Path):
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = Config.wrap(payload["config"])
    ontology = OntologyManager(payload["ontology_domain"])
    tokenizer = VocabularyTokenizer.load(checkpoint_path.parent / "vocab.json")
    model = OKEHABSANet(config, ontology, len(tokenizer.vocabulary))
    trainer = StageTrainer(model, config, ontology, checkpoint_path.parent)
    trainer.load_checkpoint(checkpoint_path)
    return config, ontology, tokenizer, model, trainer


def command_evaluate(args, _config) -> None:
    config, ontology, tokenizer, model, trainer = load_artifacts(args.checkpoint)
    dataset_name = canonical_dataset(args.dataset or str(config.data.domain))
    records = load_records(Path(config.data.root) / dataset_name / "test.json")
    if args.max_records is not None:
        records = records[: args.max_records]
    preprocessor = TextPreprocessor(tokenizer, ontology, int(config.data.max_length))
    dataset = OKEHABSADataset(records, preprocessor, float(config.ontology.mapping_threshold))
    metrics = trainer.evaluate(make_loader(dataset, config))
    metrics["samples"] = len(dataset)
    print(json.dumps(metrics, indent=2))


def make_inference_batch(text: str, aspect: str, tokenizer, ontology, config):
    tokens = tokenizer.tokenize(text)
    aspect_tokens = tokenizer.tokenize(aspect)
    normalized = [tokenizer.normalize(token) for token in tokens]
    target = [tokenizer.normalize(token) for token in aspect_tokens]
    start = next(
        (
            index
            for index in range(len(tokens) - len(target) + 1)
            if normalized[index : index + len(target)] == target
        ),
        0,
    )
    end = min(len(tokens), start + max(1, len(target)))
    record = {
        "token": tokens,
        "aspects": [
            {"term": aspect_tokens, "from": start, "to": end, "polarity": "neutral"}
        ],
    }
    preprocessor = TextPreprocessor(tokenizer, ontology, int(config.data.max_length))
    dataset = OKEHABSADataset([record], preprocessor, float(config.ontology.mapping_threshold))
    return collate_oke_habsa([dataset[0]])


def predict_payload(args):
    config, ontology, tokenizer, model, trainer = load_artifacts(args.checkpoint)
    batch = move_batch(
        make_inference_batch(args.text, args.aspect, tokenizer, ontology, config),
        trainer.device,
    )
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            aspect_mask=batch["aspect_mask"],
            concept_id=batch["concept_id"],
            main_concept_id=batch["main_concept_id"],
            dependency=batch["dependency"],
        )
        probabilities = torch.softmax(outputs["logits"], dim=-1)[0]
    prediction = int(probabilities.argmax().item())
    payload = {
        "text": args.text,
        "aspect": args.aspect,
        "ontology_concept": batch["concept"][0],
        "sentiment": SENTIMENT_NAMES[prediction],
        "score": float(INDEX_TO_SCORE[prediction].item()),
        "probabilities": {
            name: float(probabilities[index].item())
            for index, name in enumerate(SENTIMENT_NAMES)
        },
    }
    return payload, batch, outputs, ontology, model, config


def command_predict(args, _config) -> None:
    payload, *_ = predict_payload(args)
    print(json.dumps(payload, indent=2))


def command_explain(args, _config) -> None:
    payload, batch, outputs, ontology, model, config = predict_payload(args)
    ig = TokenIntegratedGradients(
        model, steps=args.steps or int(config.explainability.integrated_gradients_steps)
    )
    token_result = ig.attribute(batch)
    concept_explainer = ConceptAttribution(model, ontology)
    concept_scores = concept_explainer.aggregate(
        token_result["token_scores"], batch["token_concept_ids"]
    )[0]
    structural = StructuralExplainer(ontology)
    payload["token_attributions"] = [
        {
            "token": token,
            "score": float(token_result["normalized_scores"][0, index].item()),
        }
        for index, token in enumerate(batch["tokens"][0])
    ]
    payload["concept_attributions"] = concept_scores
    payload["propagation_path"] = structural.active_path(
        outputs, batch["concept"][0]
    )
    payload["counterfactual_positive"] = structural.counterfactual(
        outputs, batch["concept"][0], 1.0
    )
    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OKE-HABSA pipeline")
    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="Override config"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ontology_parser = subparsers.add_parser("build-ontology")
    ontology_parser.add_argument("--dataset")
    ontology_parser.add_argument("--output")
    ontology_parser.set_defaults(handler=command_build_ontology)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset")
    train_parser.add_argument("--output-dir")
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--transe-epochs", type=int)
    train_parser.add_argument(
        "--max-records", type=int, help="Limit input sentences for smoke runs"
    )
    train_parser.set_defaults(handler=command_train)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--dataset")
    evaluate_parser.add_argument("--max-records", type=int)
    evaluate_parser.set_defaults(handler=command_evaluate)

    for name, handler in [("predict", command_predict), ("explain", command_explain)]:
        inference_parser = subparsers.add_parser(name)
        inference_parser.add_argument("--checkpoint", required=True)
        inference_parser.add_argument("--text", required=True)
        inference_parser.add_argument("--aspect", required=True)
        if name == "explain":
            inference_parser.add_argument("--steps", type=int)
        inference_parser.set_defaults(handler=handler)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = apply_dotted_overrides(load_config(args.config), args.set)
    args.handler(args, config)


if __name__ == "__main__":
    main()
