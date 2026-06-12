from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from config import Config, load_config
from data_loader import (
    OKEHABSADataset,
    SENTIMENT_VALUES,
    TextPreprocessor,
    TransformerTokenizer,
    VocabularyTokenizer,
    collate_oke_habsa,
    load_records,
)
from explainability import ConceptAttribution, StructuralExplainer, TokenIntegratedGradients
from models import OKEHABSANet
from ontology import OntologyManager
from trainers import StageTrainer, move_batch


SENTIMENT_NAMES = {
    3: ["negative", "neutral", "positive"],
    5: ["very_negative", "negative", "neutral", "positive", "very_positive"],
}


class RunLogger:
    def __init__(self, path: str | Path, reset: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} | {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def canonical_dataset(name: str) -> str:
    choices = {
        "laptop": "Laptops",
        "laptops": "Laptops",
    }
    try:
        return choices[name.lower()]
    except KeyError as exc:
        raise ValueError("The implemented ontology currently supports Laptops only") from exc


def make_loader(dataset, config, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=int(config.training.batch_size),
        shuffle=shuffle,
        num_workers=int(config.data.num_workers),
        collate_fn=collate_oke_habsa,
    )


def split_records(records, ratio: float, seed: int, ontology=None):
    ontology = ontology or OntologyManager("laptop")
    strata = {}
    for index, record in enumerate(records):
        polarities = sorted(
            {aspect.get("polarity", "neutral") for aspect in record.get("aspects", [])}
        )
        concepts = []
        for aspect in record.get("aspects", []):
            concept, _ = ontology.map_entity(aspect.get("term", []))
            branch = next(
                (
                    name
                    for name in reversed(ontology.ancestors(concept))
                    if ontology.concepts[name].depth == 1
                ),
                concept,
            )
            concepts.append(branch)
        signature = ("+".join(polarities), sorted(concepts)[0] if concepts else ontology.root)
        strata.setdefault(signature, []).append(index)
    rng = random.Random(seed)
    validation_ids = set()
    for members in strata.values():
        rng.shuffle(members)
        count = min(len(members) - 1, int(len(members) * ratio))
        validation_ids.update(members[: max(0, count)])
    target = max(1, round(len(records) * ratio))
    remaining = [index for index in range(len(records)) if index not in validation_ids]
    rng.shuffle(remaining)
    validation_ids.update(remaining[: max(0, target - len(validation_ids))])
    train = [(index, record) for index, record in enumerate(records) if index not in validation_ids]
    validation = [(index, record) for index, record in enumerate(records) if index in validation_ids]
    return train, validation


def build_tokenizer(config, train_records=None):
    if str(config.model.text_backend) == "transformer":
        return TransformerTokenizer(
            str(config.model.pretrained_model), bool(config.model.local_files_only)
        )
    tokenizer = VocabularyTokenizer()
    if train_records is not None:
        tokenizer.fit(
            train_records,
            int(config.data.min_frequency),
            int(config.model.vocab_size),
        )
    return tokenizer


def command_build_ontology(args, _config):
    ontology = OntologyManager("laptop")
    errors = ontology.validate()
    if errors:
        raise RuntimeError("\n".join(errors))
    artifacts = ontology.export_all(args.output or "ontology/laptop_domain.owl")
    errors = ontology.validate_owl(artifacts["owl"], run_reasoner=args.reasoner)
    if errors:
        raise RuntimeError("\n".join(errors))
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2))


def command_train(args, config):
    dataset_name = canonical_dataset(args.dataset or str(config.data.domain))
    if args.epochs is not None:
        config.training.warmup_epochs = min(1, args.epochs)
        config.training.main_epochs = max(0, args.epochs - 1)
        config.training.finetune_epochs = 0
    if args.transe_epochs is not None:
        config.ontology.transe_epochs = args.transe_epochs
    output = Path(args.output_dir or config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger = getattr(args, "_logger", None) or RunLogger(output / "run.log", reset=True)
    config.output_dir = str(output)
    config.data.domain = dataset_name
    set_seed(int(config.seed))
    logger.log(
        f"[RUN] train_start dataset={dataset_name} seed={int(config.seed)} "
        f"backend={config.model.text_backend} output={output}"
    )

    records = load_records(Path(config.data.root) / dataset_name / "train.json")
    if args.max_records:
        records = records[: args.max_records]
    ontology = OntologyManager("laptop")
    train_pairs, validation_pairs = split_records(
        records, float(config.data.validation_ratio), int(config.seed), ontology
    )
    train_records = [record for _, record in train_pairs]
    validation_records = [record for _, record in validation_pairs]
    tokenizer = build_tokenizer(config, train_records)
    preprocessor = TextPreprocessor(tokenizer, ontology, int(config.data.max_length))
    common = {
        "preprocessor": preprocessor,
        "mapping_threshold": float(config.ontology.mapping_threshold),
        "num_sentiments": int(config.model.num_sentiments),
    }
    train_dataset = OKEHABSADataset(
        train_records,
        sentence_ids=[index for index, _ in train_pairs],
        **common,
    )
    validation_dataset = OKEHABSADataset(
        validation_records,
        sentence_ids=[index for index, _ in validation_pairs],
        **common,
    )
    vocab_size = (
        len(tokenizer.vocabulary)
        if isinstance(tokenizer, VocabularyTokenizer)
        else len(tokenizer.tokenizer)
    )
    model = OKEHABSANet(config, ontology, vocab_size)
    trainer = StageTrainer(
        model,
        config,
        ontology,
        output,
        class_weights=train_dataset.class_weights(),
        log_callback=logger.log,
    )
    logger.log(
        f"[DATA] train_sentences={len(train_records)} "
        f"validation_sentences={len(validation_records)} "
        f"train_aspects={len(train_dataset)} "
        f"validation_aspects={len(validation_dataset)}"
    )
    kge_losses = trainer.pretrain_transe()
    history = trainer.fit(
        make_loader(train_dataset, config, True),
        make_loader(validation_dataset, config),
    )
    tokenizer.save(output / "tokenizer.json")
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "seed": int(config.seed),
        "train_sentence_ids": [index for index, _ in train_pairs],
        "validation_sentence_ids": [index for index, _ in validation_pairs],
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    artifacts = ontology.export_all(output / "ontology.owl")
    mapping = {
        "samples": len(train_dataset) + len(validation_dataset),
        "root_fallback_rate": sum(
            item["concept"] == ontology.root
            for dataset in (train_dataset, validation_dataset)
            for item in dataset.items
        )
        / max(1, len(train_dataset) + len(validation_dataset)),
        "mean_confidence": sum(
            item["mapping_confidence"]
            for dataset in (train_dataset, validation_dataset)
            for item in dataset.items
        )
        / max(1, len(train_dataset) + len(validation_dataset)),
    }
    (output / "mapping_report.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )
    (output / "explanations").mkdir(exist_ok=True)
    summary = {
        "train_aspects": len(train_dataset),
        "validation_aspects": len(validation_dataset),
        "transe_final_loss": kge_losses[-1] if kge_losses else None,
        "best_checkpoint": str(output / "best_model.pt"),
        "last_checkpoint": str(output / "last_model.pt"),
        "epochs_completed": len(history),
        "best_validation_macro_f1": max(
            (float(row["validation"]["macro_f1"]) for row in history),
            default=None,
        ),
        "ontology_hash": artifacts["sha256"],
    }
    (output / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.log(
        f"[RUN] train_completed epochs={summary['epochs_completed']} "
        f"best_val_f1={summary['best_validation_macro_f1']} "
        f"checkpoint={summary['best_checkpoint']}"
    )
    print(json.dumps(summary, indent=2))
    return summary


def load_artifacts(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = Config.wrap(payload["config"])
    ontology = OntologyManager(payload["ontology_domain"])
    metadata = json.loads(
        (checkpoint_path.parent / "tokenizer.json").read_text(encoding="utf-8")
    )
    if metadata.get("type") == "transformer":
        tokenizer = TransformerTokenizer(
            metadata["model_name"], bool(config.model.local_files_only)
        )
        vocab_size = len(tokenizer.tokenizer)
    else:
        tokenizer = VocabularyTokenizer.load(checkpoint_path.parent / "tokenizer.json")
        vocab_size = len(tokenizer.vocabulary)
    model = OKEHABSANet(config, ontology, vocab_size)
    trainer = StageTrainer(model, config, ontology, checkpoint_path.parent)
    trainer.load_checkpoint(checkpoint_path)
    return config, ontology, tokenizer, model, trainer


def command_evaluate(args, _config):
    config, ontology, tokenizer, _, trainer = load_artifacts(args.checkpoint)
    records = load_records(Path(config.data.root) / canonical_dataset(args.dataset) / "test.json")
    if args.max_records:
        records = records[: args.max_records]
    dataset = OKEHABSADataset(
        records,
        TextPreprocessor(tokenizer, ontology, int(config.data.max_length)),
        float(config.ontology.mapping_threshold),
        int(config.model.num_sentiments),
    )
    metrics = trainer.evaluate(make_loader(dataset, config))
    metrics["samples"] = len(dataset)
    target = Path(args.checkpoint).parent / "test_metrics.json"
    target.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (Path(args.checkpoint).parent / "per_class_metrics.json").write_text(
        json.dumps(metrics["per_class"], indent=2), encoding="utf-8"
    )
    logger = getattr(args, "_logger", None)
    if logger is not None:
        logger.log(
            f"[TEST] samples={metrics['samples']} loss={float(metrics['total']):.4f} "
            f"macro_f1={float(metrics['macro_f1']):.4f} "
            f"accuracy={float(metrics['accuracy']):.4f} "
            f"mae={float(metrics['mae']):.4f} "
            f"oc={float(metrics['ontological_consistency']):.4f}"
        )
    print(json.dumps(metrics, indent=2))
    return metrics


def command_run_pipeline(args, config):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output / "run.log", reset=True)
    started_at = datetime.now().isoformat(timespec="seconds")
    logger.log(
        f"[PIPELINE] start dataset={args.dataset} output={output} "
        f"reasoner={args.reasoner}"
    )

    ontology = OntologyManager("laptop")
    errors = ontology.validate()
    if errors:
        raise RuntimeError("\n".join(errors))
    ontology_artifacts = ontology.export_all(output / "ontology.owl")
    errors = ontology.validate_owl(
        ontology_artifacts["owl"], run_reasoner=args.reasoner
    )
    if errors:
        raise RuntimeError("\n".join(errors))
    logger.log(
        f"[ONTOLOGY] concepts={len(ontology.names)} "
        f"edges={len(ontology.hierarchy_edges())} "
        f"hash={ontology_artifacts['sha256']}"
    )

    train_args = argparse.Namespace(
        dataset=args.dataset,
        output_dir=str(output),
        epochs=None,
        transe_epochs=None,
        max_records=args.max_records,
        _logger=logger,
    )
    train_summary = command_train(train_args, config)
    checkpoint = Path(train_summary["best_checkpoint"])
    if not checkpoint.exists():
        raise RuntimeError(f"Best checkpoint was not created: {checkpoint}")

    evaluate_args = argparse.Namespace(
        checkpoint=str(checkpoint),
        dataset=args.dataset,
        max_records=args.max_test_records,
        _logger=logger,
    )
    test_metrics = command_evaluate(evaluate_args, config)
    results = {
        "status": "completed",
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": canonical_dataset(args.dataset),
        "output_dir": str(output),
        "ontology": {
            "concepts": len(ontology.names),
            "hierarchy_edges": len(ontology.hierarchy_edges()),
            "hash": ontology_artifacts["sha256"],
        },
        "training": train_summary,
        "test": test_metrics,
        "artifacts": {
            "run_log": str(output / "run.log"),
            "history": str(output / "history.json"),
            "best_checkpoint": str(checkpoint),
            "test_metrics": str(output / "test_metrics.json"),
            "per_class_metrics": str(output / "per_class_metrics.json"),
        },
    }
    results_path = output / "pipeline_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.log(
        f"[PIPELINE] completed test_f1={float(test_metrics['macro_f1']):.4f} "
        f"test_acc={float(test_metrics['accuracy']):.4f} "
        f"test_mae={float(test_metrics['mae']):.4f} "
        f"test_oc={float(test_metrics['ontological_consistency']):.4f} "
        f"results={results_path}"
    )
    print(json.dumps(results, indent=2))
    return results


def make_inference_batch(text, aspect, tokenizer, ontology, config):
    tokens = tokenizer.tokenize(text)
    aspect_tokens = tokenizer.tokenize(aspect)
    normalized = [token.lower() for token in tokens]
    target = [token.lower() for token in aspect_tokens]
    start = next(
        (
            index
            for index in range(len(tokens) - len(target) + 1)
            if normalized[index : index + len(target)] == target
        ),
        None,
    )
    if start is None:
        raise ValueError("The aspect must occur verbatim in the input text")
    dataset = OKEHABSADataset(
        [
            {
                "token": tokens,
                "aspects": [
                    {
                        "term": aspect_tokens,
                        "from": start,
                        "to": start + len(target),
                        "polarity": "neutral",
                    }
                ],
            }
        ],
        TextPreprocessor(tokenizer, ontology, int(config.data.max_length)),
        float(config.ontology.mapping_threshold),
        int(config.model.num_sentiments),
    )
    return collate_oke_habsa([dataset[0]])


def predict_payload(args):
    config, ontology, tokenizer, model, trainer = load_artifacts(args.checkpoint)
    batch = move_batch(
        make_inference_batch(args.text, args.aspect, tokenizer, ontology, config),
        trainer.device,
    )
    model.eval()
    with torch.no_grad():
        outputs = model(**batch)
        probabilities = torch.softmax(outputs["logits"], -1)[0]
    prediction = int(probabilities.argmax())
    names = SENTIMENT_NAMES[int(config.model.num_sentiments)]
    return {
        "aspect": args.aspect,
        "ontology_concept": batch["concept"][0],
        "sentiment": names[prediction],
        "score": float(SENTIMENT_VALUES[len(names)][prediction]),
        "probabilities": {
            name: float(probabilities[index]) for index, name in enumerate(names)
        },
    }, batch, outputs, ontology, model, config


def command_predict(args, _config):
    payload, *_ = predict_payload(args)
    print(json.dumps(payload, indent=2))


def command_explain(args, _config):
    payload, batch, outputs, ontology, model, config = predict_payload(args)
    attribution = TokenIntegratedGradients(
        model, args.steps or int(config.explainability.integrated_gradients_steps)
    ).attribute(batch)
    payload["token_attributions"] = attribution["normalized_scores"][0].tolist()
    payload["concept_attributions"] = ConceptAttribution(model, ontology).aggregate(
        attribution["token_scores"], batch["token_concept_ids"]
    )[0]
    payload["propagation_path"] = StructuralExplainer(ontology).active_path(
        outputs, batch["concept"][0]
    )
    print(json.dumps(payload, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "OKE-HABSA pipeline. Run without a subcommand to execute the full "
            "pipeline using config/default_config.yaml."
        )
    )
    parser.add_argument("--config")
    sub = parser.add_subparsers(dest="command")
    build = sub.add_parser("build-ontology")
    build.add_argument("--dataset", default="Laptops")
    build.add_argument("--output")
    build.add_argument("--reasoner", action="store_true")
    build.set_defaults(handler=command_build_ontology)
    train = sub.add_parser("train")
    train.add_argument("--dataset", default="Laptops")
    train.add_argument("--output-dir")
    train.add_argument("--epochs", type=int)
    train.add_argument("--transe-epochs", type=int)
    train.add_argument("--max-records", type=int)
    train.set_defaults(handler=command_train)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--dataset", default="Laptops")
    evaluate.add_argument("--max-records", type=int)
    evaluate.set_defaults(handler=command_evaluate)
    for name, handler in (("predict", command_predict), ("explain", command_explain)):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--text", required=True)
        command.add_argument("--aspect", required=True)
        if name == "explain":
            command.add_argument("--steps", type=int)
        command.set_defaults(handler=handler)
    return parser


def pipeline_args_from_config(config):
    pipeline = config.pipeline
    return argparse.Namespace(
        dataset=str(config.data.domain),
        output_dir=str(pipeline.output_dir),
        epochs=None,
        transe_epochs=None,
        max_records=pipeline.max_records,
        max_test_records=pipeline.max_test_records,
        reasoner=bool(pipeline.reasoner),
    )


def main():
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command is None:
        command_run_pipeline(pipeline_args_from_config(config), config)
    else:
        args.handler(args, config)


if __name__ == "__main__":
    main()
