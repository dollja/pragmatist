# Verification

The consolidated package was run successfully from the package root using:

```bash
python run_complete_pipeline_demo.py
```

## Result

A fresh demo run was created under:

```text
examples/generated_demo_run/
```

## High-level checks

- The advanced balanced pipeline completed and exported:
  - `generated_candidates.jsonl`
  - `balanced_selected_records.jsonl`
  - `context_packet.yaml`
  - `context_strategy_comparison.csv`
  - `generation_metrics.json`
  - `consolidation_report.json`

- The Pragmatist bridge completed and exported:
  - `pragmatist_context_history.json`
  - `pragmatist_ab_results.csv`
  - `pragmatist_ab_summary.json`
  - `pragmatist_packet_audit.json`
  - `pragmatist_consolidation.json`

- The context-error engineering layer completed and exported:
  - `pragmatist_context_error_snapshot.json`
  - `pragmatist_context_error_actions.json`
  - `context_error_turn_log.csv`
  - `context_error_event_log.csv`
  - `context_error_summary.json`
  - `context_error_recommendations.json`

## Demo summary highlights

- Recommended context strategy: `anchor_priority`
- Context-error health score in the demo run: `100.0`
- Packet audit issue count for the demo hydration: `0`

The generated summary file is:

```text
examples/generated_demo_run/complete_pipeline_demo_summary.json
```

