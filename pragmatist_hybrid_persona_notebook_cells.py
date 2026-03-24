
# %% [markdown]
"""
Pragmatist × Hybrid Persona Notebook Cells (with context-error engineering)
---------------------------------------------------------------------------
Paste these cells into the Pragmatist notebook after the balanced hybrid pipeline
has been generated and consolidated.
"""

# %%
from advanced_balanced_hybrid_persona_pipeline import (
    BalancedHybridPipeline,
    PipelineConfig,
    load_codebook_csv,
    load_json_schema,
)
import pandas as pd

from pragmatist_hybrid_persona_integration import (
    PragmatistHybridBridge,
    PragmatistHybridState,
    PragmatistHybridHooks,
    make_pragmatist_hybrid_instructions,
)

# %%
# 1) Build the balanced hybrid persona pipeline.
codebook_df = load_codebook_csv("hybrid_persona_label_codebook.csv")
schema = load_json_schema("hybrid_persona_schema.json")
pipeline = BalancedHybridPipeline(codebook_df, schema, config=PipelineConfig())

# anchor_cards_df = ...
# observed_df = ...
# pipeline.register_anchors(anchor_cards_df)
# pipeline.ingest_observed_records(observed_df, source_mode="observed", assign_anchor=False)
# candidates = pipeline.generate_candidates()
# selected = pipeline.rebalance(candidates)
# pipeline.consolidate(selected)

# %%
# 2) Wrap the balanced hybrid state inside the Pragmatist-style bridge.
bridge = PragmatistHybridBridge.from_pipeline(
    pipeline,
    user_controls={
        "allow_pending_records": False,
        "allow_synthetic_expansions": True,
    },
    default_strategy="anchor_priority",
)
user_state: PragmatistHybridState = bridge.state

# %%
# 3) A/B test context selection strategies before wiring the agent.
benchmark_queries = [
    "best zero trust access tools for remote employees with sso",
    "student discount note taking app with offline sync",
    "edr pricing for 100 endpoints",
]
ab_report = bridge.evaluate_strategies(benchmark_queries)
print("Recommended strategy:", ab_report["recommended_strategy"])
pd.DataFrame(ab_report["summary"])

# %%
# 4) Stress-test the context layer and inspect the dedicated error-engineering view.
# These controls intentionally make failures easier to observe during debugging.
bridge.state.user_controls["allow_pending_records"] = True
bridge.state.user_controls["max_context_global_k"] = 10
bridge.state.user_controls["max_context_session_k"] = 10
bridge.config.context_token_budget = 350

stress_queries = [
    "best zero trust access tools for remote employees with sso",
    "best crm for freelancers with cheap pricing",
    "help me choose a website builder for a bakery with online ordering",
]

for query in stress_queries:
    _ = bridge.hydrate_turn(query)

error_snapshot = bridge.sync_context_error_layer()
print("Health score:", error_snapshot["summary"]["health_score"])
print("Issue counts:", error_snapshot["summary"]["issue_counts"])
pd.DataFrame(error_snapshot["recommendations"])

# %%
# 5) Drill into turn-level and issue-level logs.
turn_df = bridge.context_error_turn_frame()
event_df = bridge.context_error_event_frame()

turn_df[["turn_id", "query", "strategy", "issue_count", "estimated_tokens", "anchor_share"]]
event_df[["turn_id", "issue_code", "issue_family", "severity", "query"]]

# %%
# 6) Apply the recommended mitigations back into runtime controls.
mitigation_report = bridge.apply_context_error_actions(auto_apply=True)
mitigation_report

# %%
# 7) Replace the notebook's hook and instructions with persona-aware versions.
BASE_INSTRUCTIONS = """
You are a helpful research assistant.
Ground your response in the injected persona context when it is relevant.
If the current context packet is weak or incomplete, ask a precise follow-up instead of assuming.
""".strip()

memory_hooks = bridge.make_hooks()
instructions = bridge.make_instructions(
    base_instructions=BASE_INSTRUCTIONS,
    extra_policy="Always prefer anchor facts over speculative persona details.",
    include_diagnostics=True,
)

# %%
# 8) Use `user_state`, `memory_hooks`, and `instructions` in your final Agent.
# Example:
# from agents import Agent
# agent = Agent(
#     name="Persona-Aware Assistant",
#     instructions=instructions,
#     hooks=memory_hooks,
#     model="your-model-here",
# )
#
# result = await Runner.run(agent, input="best zero trust access tools for remote employees with sso", context=user_state)
# print(result.final_output)

# %%
# 9) Keep consolidation in your post-session stage.
consolidation_report = bridge.consolidate()
consolidation_report

# %%
# 10) Optional: export the error dashboard artifacts.
artifact_paths = bridge.export_context_error_artifacts("pragmatist_context_error_artifacts")
artifact_paths
