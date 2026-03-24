# Pragmatist × Hybrid Persona Integration Guide

This guide wires the balanced hybrid persona module into the broader Pragmatist-style contextual-engineering pipeline.

## What this integration changes

Instead of keeping the persona generator separate from the runtime memory system, the new bridge makes persona-aware context packets a first-class part of the turn lifecycle.

The bridge reuses three existing touchpoints from the Pragmatist notebook pattern:

1. **State object** → use `PragmatistHybridState`, which extends `HybridPersonaState`
2. **On-start hook** → use `PragmatistHybridHooks`, which calls `build_pragmatist_context_packet(...)`
3. **A/B evaluation stage** → use `evaluate_context_strategies(...)`, which wraps `compare_context_strategies(...)`

## Files

- `advanced_balanced_hybrid_persona_pipeline.py`
- `pragmatist_hybrid_persona_integration.py`

Place both files in the same directory.

## Minimal notebook wiring

### 1) Imports

```python
from advanced_balanced_hybrid_persona_pipeline import (
    BalancedHybridPipeline,
    PipelineConfig,
    load_codebook_csv,
    load_json_schema,
)

from pragmatist_hybrid_persona_integration import (
    PragmatistHybridBridge,
    PragmatistHybridState,
    PragmatistHybridHooks,
    make_pragmatist_hybrid_instructions,
)
```

### 2) Build the balanced hybrid pipeline as usual

```python
codebook_df = load_codebook_csv("hybrid_persona_label_codebook.csv")
schema = load_json_schema("hybrid_persona_schema.json")

pipeline = BalancedHybridPipeline(codebook_df, schema, config=PipelineConfig())
pipeline.register_anchors(anchor_cards_df)
pipeline.ingest_observed_records(observed_df, source_mode="observed", assign_anchor=False)

candidates = pipeline.generate_candidates()
selected = pipeline.rebalance(candidates)
pipeline.consolidate(selected)
```

### 3) Wrap the existing hybrid state in the Pragmatist bridge

```python
bridge = PragmatistHybridBridge.from_pipeline(
    pipeline,
    user_controls={
        "allow_pending_records": False,
        "allow_synthetic_expansions": True,
    },
    default_strategy="anchor_priority",
)

user_state = bridge.state
```

At this point, `user_state` is a `PragmatistHybridState` that preserves the local-first state shape but now carries:

- `anchor_registry`
- approved synthetic records in `global_memory`
- current candidate/session records in `session_memory`
- runtime fields for context-packet history and context-error tracking

### 4) Replace the notebook's on-start memory hook

```python
memory_hooks = bridge.make_hooks()
```

This hook now does the following every turn:

- extracts the live user query from `ctx.turn_input`
- calls `build_pragmatist_context_packet(...)`
- selects the best memory slices for the turn
- writes `system_frontmatter`, `global_memories_md`, and `session_memories_md` back onto state
- logs packet diagnostics to `state.runtime["context_history"]`

### 5) Replace the notebook's dynamic instructions function

```python
BASE_INSTRUCTIONS = """
You are a helpful research assistant.
Ground your response in the provided persona context when it is relevant.
When the context packet indicates weak grounding, ask a precise follow-up instead of assuming.
"""

instructions = bridge.make_instructions(
    base_instructions=BASE_INSTRUCTIONS,
    extra_policy="Always prefer anchor facts over speculative persona details.",
    include_diagnostics=True,
)
```

The generated instructions function injects:

- YAML frontmatter from the context packet
- global persona memory
- session memory
- the hybrid precedence policy
- diagnostics about packet risks such as `missing_topic_support` or `weak_anchor_grounding`

### 6) Swap in the bridge for context-strategy A/B tests

```python
benchmark_queries = [
    "best zero trust access tools for remote employees with sso",
    "student discount note taking app with offline sync",
    "edr pricing for 100 endpoints",
]

ab_report = bridge.evaluate_strategies(benchmark_queries)
ab_report["recommended_strategy"]
```

This returns both the raw per-query results and a ranked strategy summary. The bridge automatically persists the winning strategy into:

```python
bridge.state.runtime["default_strategy"]
```

### 7) Keep post-session consolidation in the notebook lifecycle

```python
consolidation_report = bridge.consolidate()
```

This uses the original hybrid module's consolidation logic, which already preserves anchor notes, filters stale low-importance notes, deduplicates memory, and runs critic-style safety checks before writing to global memory.

## Recommended replacement map for the Pragmatist notebook

### Replace the state object section

**Old**: `TravelState`

**New**: `PragmatistHybridState`

You no longer need a separate travel-only state if the notebook is now being used as the general contextual-engineering layer for persona-aware synthesis experiments.

### Replace the basic `MemoryHooks` / `SmartMemoryHooks`

**Old behavior**: keyword-only relevance and simple recency

**New behavior**: `build_context_packet(...)` with label-aware selection and `compare_context_strategies(...)` for evaluation

### Replace the A/B harness

**Old behavior**: manual strategy A vs strategy B functions

**New behavior**:

```python
bridge.evaluate_strategies(queries)
```

This gives you:

- lower-level packet diagnostics
- the selected strategy persisted on state
- reusable reports for later context-error engineering

## Why this wiring is the cleanest fit

The integration keeps the exact broader pipeline shape from Pragmatist:

- state-first runtime
- dynamic context injection at turn start
- explicit memory precedence rules
- post-session consolidation
- evaluation of injection strategies

But it replaces the simple keyword-based memory selection with the hybrid persona system's stronger packet builder, which is grounded in:

- anchor personas
- label-aware retrieval
- session/global memory separation
- audit-ready context diagnostics

## Small debugging checklist

If your packets look sparse:

- confirm `pipeline.consolidate(...)` has already been run before the bridge is created
- inspect `bridge.state.global_memory["notes"]`
- run `bridge.evaluate_strategies(...)` and check `issue_count`
- make sure `allow_global_memory` and `allow_session_memory` are still enabled in `user_controls`

If your agent overuses synthetic examples:

```python
bridge.state.user_controls["allow_synthetic_expansions"] = False
```

If you want to block pending candidates from runtime injection:

```python
bridge.state.user_controls["allow_pending_records"] = False
```

## Suggested next step

The next logical extension is to turn `state.runtime["context_history"]` and `state.runtime["context_errors"]` into the first layer of a dedicated context-error engineering dashboard.
