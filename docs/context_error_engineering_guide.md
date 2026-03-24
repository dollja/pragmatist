
# Pragmatist × Hybrid Persona Context-Error Engineering Guide

This layer now sits between packet logging and runtime controls.

## What it consumes

The integration bridge writes every hydrated packet summary into:

- `state.runtime["context_history"]`
- `state.runtime["context_errors"]`

The dedicated layer then normalizes those streams into:

- `state.runtime["context_error_layer"]["turn_log"]`
- `state.runtime["context_error_layer"]["event_log"]`
- `state.runtime["context_error_layer"]["summary"]`
- `state.runtime["context_error_layer"]["recommendations"]`

## What it diagnoses

Current issue families:

- `coverage`: `missing_context`, `missing_topic_support`
- `grounding`: `weak_anchor_grounding`, `persona_mismatch`
- `precision`: `topic_leakage`
- `safety`: `pending_context_risk`
- `budget`: `context_overload`

## Main bridge methods

```python
bridge.sync_context_error_layer()
bridge.context_error_turn_frame()
bridge.context_error_event_frame()
bridge.apply_context_error_actions(auto_apply=True)
bridge.export_context_error_artifacts("some_output_dir")
```

## Typical flow

```python
# 1) Build packets by running normal turns.
_ = bridge.hydrate_turn("best zero trust access tools for remote employees with sso")
_ = bridge.hydrate_turn("best crm for freelancers with cheap pricing")

# 2) Inspect the dedicated error layer.
snapshot = bridge.sync_context_error_layer()
snapshot["summary"]["issue_counts"]

# 3) Review proposed mitigations.
snapshot["recommendations"]

# 4) Apply them back into runtime controls.
report = bridge.apply_context_error_actions(auto_apply=True)
report["applied_actions"]
```

## What auto-mitigation can change

Depending on repeated failures, the layer may:

- disable `allow_pending_records`
- disable `allow_synthetic_expansions`
- reduce `max_context_global_k`
- reduce `max_context_session_k`
- set `runtime["default_strategy"]`
- set `runtime["require_grounded_followup_on_weak_packets"]`
- append structured items into `runtime["coverage_gap_queue"]`

## Why this matches the Pragmatist shape

It preserves the same pipeline order:

1. state-first memory store
2. injection-time packet hydration
3. guardrail-aware prompt assembly
4. evaluation and A/B testing
5. post-hoc diagnosis and mitigation

The new part is that packet audits are no longer just logged; they become a dedicated context-error engineering substrate that can drive remediation.
