# Install and Run

## 1) Create a virtual environment

### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Run the complete demo

```bash
python run_complete_pipeline_demo.py
```

This generates a fresh example run under:

```text
examples/generated_demo_run/
```

You will get:
- advanced balanced pipeline outputs
- Pragmatist bridge outputs
- context-error engineering outputs
- a small summary manifest

## 3) Open the notebook version

Open:

```text
advanced_balanced_hybrid_persona_pipeline.ipynb
```

This notebook is already patched to use package-relative paths instead of `/mnt/data`.

## 4) Use the notebook cells in your broader notebook

Copy or import:

```text
pragmatist_hybrid_persona_notebook_cells.py
```

That file shows the expected run order for:
- building the balanced hybrid pipeline
- wrapping it with `PragmatistHybridBridge`
- evaluating context strategies
- syncing the context-error engineering layer

## 5) Replace the demo inputs with your real data

### Anchor cards
Use `templates/anchor_cards_template.csv` as the column guide.

Expected fields include:
- `anchor_persona_id`
- `macro_persona`
- `anchor_name`
- `job_to_be_done`
- `constraints`
- `success_metric`
- `decision_criteria`
- `vocabulary`
- `evidence_sources`
- `confidence_score`
- `priority_topics`

### Observed records
Use `templates/observed_records_template.csv` as the column guide.

Expected minimum fields:
- `text`
- `anchor_persona_id`

## 6) Minimal real-data workflow

```python
from advanced_balanced_hybrid_persona_pipeline import (
    BalancedHybridPipeline,
    PipelineConfig,
    load_codebook_csv,
    load_json_schema,
)

from pragmatist_hybrid_persona_integration import PragmatistHybridBridge

codebook_df = load_codebook_csv("hybrid_persona_label_codebook.csv")
schema = load_json_schema("hybrid_persona_schema.json")

pipeline = BalancedHybridPipeline(codebook_df, schema, config=PipelineConfig())
pipeline.register_anchors(anchor_cards_df)
pipeline.ingest_observed_records(observed_df, source_mode="observed", assign_anchor=False)

candidates = pipeline.generate_candidates()
selected = pipeline.rebalance(candidates)
pipeline.consolidate(selected)

bridge = PragmatistHybridBridge.from_pipeline(
    pipeline,
    user_controls={
        "allow_pending_records": False,
        "allow_synthetic_expansions": True,
    },
    default_strategy="anchor_priority",
)
```

## 7) Sync the context-error engineering layer

After a few hydrated turns:

```python
snapshot = bridge.sync_context_error_layer()
report = bridge.apply_context_error_actions(auto_apply=True)
```

Those calls transform:
- `state.runtime["context_history"]`
- `state.runtime["context_errors"]`

into a more actionable diagnosis and mitigation layer.

## 8) Optional package contents you can inspect

### Reference outputs
The `examples/reference_outputs/` directory contains the previously generated artifacts from successful runs. These are useful when you want to inspect the expected output shapes without running the package first.

### Detailed guides
See:
- `docs/pragmatist_integration_guide.md`
- `docs/context_error_engineering_guide.md`

## Common first edits

The most common things to change first are:
- your real anchor personas
- your observed input records
- the intent/topic balance
- the default strategy in the bridge
- the mitigation thresholds in the context-error layer

