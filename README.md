# OKE-HABSA

Implementation of **Ontology & Knowledge-Enhanced Hierarchical
Aspect-Based Sentiment Analysis**.

The repository contains:

- Programmatic laptop, restaurant, and social-domain ontologies with OWL,
  JSON, and TSV export.
- Lexical entity mapping and implicit-aspect rules.
- An offline BiGRU encoder or an optional local Hugging Face transformer.
- TransE hierarchy embeddings and sinusoidal ontology position encoding.
- Bidirectional cross-attention between text and ontology concepts.
- Bottom-up ontology GNN propagation with critical-aspect weighting.
- Hierarchical, monotonicity, dominance, consistency, and KGE losses.
- Four-stage training and token, concept, structural, counterfactual, and
  global explanations.

## Setup

```powershell
python -m pip install -r requirements.txt
```

The default `bigru` backend requires no model download. To use XLM-R, set
`model.text_backend=transformer` and make the configured model available in
the local Hugging Face cache, or set `model.local_files_only=false`.

## Commands

Build ontology artifacts:

```powershell
python main.py build-ontology --dataset Laptops
```

Train with the full four-stage schedule:

```powershell
python main.py train --dataset Laptops --output-dir outputs/laptops
```

Run a short integration training:

```powershell
python main.py train --dataset Laptops --epochs 1 --transe-epochs 1 `
  --set data.validation_ratio=0.02 --set training.batch_size=16
```

Evaluate:

```powershell
python main.py evaluate --checkpoint outputs/laptops/best_model.pt
```

Predict and explain:

```powershell
python main.py predict --checkpoint outputs/laptops/best_model.pt `
  --text "The battery life is excellent." --aspect "battery life"

python main.py explain --checkpoint outputs/laptops/best_model.pt `
  --text "The touchpad is not responsive." --aspect "touchpad" --steps 20
```

Configuration lives in `config/default_config.yaml`. Any setting can be
overridden with repeatable `--set section.key=value` arguments.

## Sentiment labels

OKE-HABSA predicts five levels: `very_negative`, `negative`, `neutral`,
`positive`, and `very_positive`, corresponding to `-2..+2`. The bundled
SemEval datasets contain three labels, mapped to `-1`, `0`, and `+1`.

## Tests

```powershell
python -m unittest discover -s tests -v
```
