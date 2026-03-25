# Complete Hybrid Persona + Pragmatist Context Engineering Pipeline Package

This package bundles the final working version of the full pipeline you built across the earlier steps:

1. **Grounded hybrid persona generation**
2. **Pragmatist-style runtime integration**
3. **Dedicated context-error engineering**

The package is organized so you can do two things without hunting across older artifacts:

- run the full demo end to end
- swap the demo inputs for your real anchor cards and observed records

## Start here

If you want the fastest path, use this order:

1. Read `ARCHITECTURE_FLOW.md`
2. Read `INSTALL_AND_RUN.md`
3. Run `python run_complete_pipeline_demo.py`
4. Open `advanced_balanced_hybrid_persona_pipeline.ipynb`
5. Use `pragmatist_hybrid_persona_notebook_cells.py` to wire the pipeline into your Pragmatist-style notebook

## What each core file is for

### Grounding and generation
- `advanced_balanced_hybrid_persona_pipeline.py`  
  The main offline synthesis pipeline. It creates anchor-aware candidates, validates them, rebalances them, consolidates them into memory notes, and builds context packets.

- `hybrid_persona_schema.json`  
  The record schema used to validate generated outputs.

- `hybrid_persona_label_codebook.csv`  
  The closed-label taxonomy and cue examples used by the labeler.

- `hybrid_persona_system_guide.md`  
  The original design guide that explains the balanced Script + PersonaHub hybrid logic.

### Runtime integration
- `pragmatist_hybrid_persona_integration.py`  
  The adapter that plugs the balanced hybrid pipeline into a broader Pragmatist-style context engineering workflow.

- `pragmatist_hybrid_persona_notebook_cells.py`  
  Notebook-ready cells for wiring the bridge into an interactive notebook flow.

### Context-error engineering
- `pragmatist_hybrid_context_error_engineering.py`  
  The dedicated layer that turns packet history and packet errors into diagnostics, recommendations, and runtime mitigation actions.

### Guidance and examples
- `ARCHITECTURE_FLOW.md`  
  Plain-language explanation of the essential architecture and how data moves through the system.

- `INSTALL_AND_RUN.md`  
  Setup steps, run order, and minimal commands.

- `docs/pragmatist_integration_guide.md`  
  More detailed notes for the Pragmatist bridge.

- `docs/context_error_engineering_guide.md`  
  More detailed notes for the context-error engineering layer.

- `templates/anchor_cards_template.csv`  
  Example format for your grounded Script persona cards.

- `templates/observed_records_template.csv`  
  Example format for your observed or seed records.

- `examples/reference_outputs/`  
  Saved outputs from previous successful runs, included so you can inspect the expected artifact shapes.

## The architecture in simple terms

Think of the system as a **three-stage loop**.

### 1) Build good synthetic data without losing grounding
You start with evidence-backed anchor personas. The generator expands around them, but every candidate still has to pass drift checks and schema validation.

### 2) Turn that offline data into live runtime context
The selected records become memory notes. When a new query arrives, the system builds a turn-specific packet that selects the most relevant global and session memories.

### 3) Learn from context failures
Every packet is logged. If packets repeatedly show missing context, topic leakage, weak grounding, or overload, the context-error layer recommends and can apply mitigation actions.

That separation is intentional:

- the **generator** makes candidate records
- the **packet builder** chooses what to inject for a specific turn
- the **error layer** improves the packet process over time

## Essential architectural components

### `BalancedHybridPipeline`
This is the main orchestrator. It handles:
- anchor registration
- observed-record ingestion
- candidate generation
- validation and drift checks
- rebalancing
- consolidation into memory

### `HybridPersonaState`
This is the shared data store. It keeps:
- anchor registry
- candidate records
- global memory
- session memory
- runtime context metadata

### `build_context_packet(...)`
This is the turn brief builder. It looks at the current query and prepares:
- predicted labels
- selected global memory slices
- selected session memory slices
- YAML frontmatter
- packet diagnostics

### `compare_context_strategies(...)`
This is the A/B evaluator for context selection. It lets you compare strategies like `anchor_priority` and `semantic_relevance` before you lock in a default.

### `PragmatistHybridBridge`
This is the adapter between the offline pipeline and the live notebook or agent runtime. It wraps the state and exposes:
- turn hydration
- hook generation
- instructions assembly
- strategy evaluation
- post-session consolidation
- context-error synchronization

### `ContextErrorEngineeringLayer`
This is the quality-control layer. It converts packet history into:
- turn logs
- issue event logs
- health summaries
- recommended actions
- optional runtime mitigations

## Recommended run order with real data

1. Prepare your anchor cards in the format shown in `templates/anchor_cards_template.csv`
2. Prepare your observed records in the format shown in `templates/observed_records_template.csv`
3. Build and consolidate the balanced hybrid pipeline
4. Wrap the pipeline with `PragmatistHybridBridge`
5. Compare context selection strategies
6. Hydrate packets during runtime turns
7. Sync the context-error layer from `state.runtime["context_history"]` and `state.runtime["context_errors"]`
8. Apply mitigation actions when repeated failure patterns appear

## Quick command

```bash
python run_complete_pipeline_demo.py
```

That script creates a fresh demo run under `examples/generated_demo_run/`.

## Python version and dependencies

Use **Python 3.10+**.

Install the core packages listed in `requirements.txt`.  
The Agents SDK import used by the bridge is optional; the bridge still works for notebook preparation and offline testing when that SDK is not installed.

## Package design choices

A few choices are deliberate:

- **Ground first, expand second** so synthetic breadth does not erase evidence.
- **Separate global and session memory** so stable facts and fresh run-time context do not get mixed together.
- **Audit packets before trusting them** so weak grounding is visible instead of hidden.
- **Treat context errors as data** so the system can improve its own packet selection policy.

## Where to customize first

If you are adapting this for your own pipeline, the first places to customize are:

- your real anchor persona cards
- your observed seed records
- your allowed topics and intent distribution
- your default context strategy
- the mitigation policy inside the context-error layer

