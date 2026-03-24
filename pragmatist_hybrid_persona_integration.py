
from __future__ import annotations

"""
Pragmatist-style integration layer for the advanced balanced hybrid persona pipeline.

This module keeps the broader contextual-engineering shape from the user's
Pragmatist-inspired notebook:
- local-first state object
- on-start hook for context hydration
- dynamic instructions assembly
- post-turn / post-session consolidation
- A/B testing of context injection strategies

It reuses the balanced hybrid persona module rather than reimplementing it and now
adds a dedicated context-error engineering layer driven by:
- runtime["context_history"]
- runtime["context_errors"]

Place this file alongside:
- `advanced_balanced_hybrid_persona_pipeline.py`
- `pragmatist_hybrid_context_error_engineering.py`
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
import copy
import importlib.util
import json
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# Resolve local modules from the filesystem.
# ---------------------------------------------------------------------------

def _load_local_module(module_name: str) -> Any:
    try:
        return __import__(module_name)
    except ModuleNotFoundError:
        module_path = Path(__file__).resolve().with_name(f"{module_name}.py")
        if not module_path.exists():
            raise
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {module_name} from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


_balanced = _load_local_module("advanced_balanced_hybrid_persona_pipeline")
_error_engineering = _load_local_module("pragmatist_hybrid_context_error_engineering")

HybridPersonaState = _balanced.HybridPersonaState
HybridMemoryNote = _balanced.HybridMemoryNote
ContextPacket = _balanced.ContextPacket
HybridLabeler = _balanced.HybridLabeler
PipelineConfig = _balanced.PipelineConfig
BalancedHybridPipeline = _balanced.BalancedHybridPipeline
HYBRID_MEMORY_POLICY = _balanced.HYBRID_MEMORY_POLICY
build_context_packet = _balanced.build_context_packet
compare_context_strategies = _balanced.compare_context_strategies
consolidate_state = _balanced.consolidate_state
load_codebook_csv = _balanced.load_codebook_csv
render_frontmatter = _balanced.render_frontmatter
render_notes_md = _balanced.render_notes_md
record_to_memory_note = _balanced.record_to_memory_note
audit_context_packet = _balanced.audit_context_packet
utc_today_iso = _balanced.utc_today_iso

ContextErrorEngineeringConfig = _error_engineering.ContextErrorEngineeringConfig
ContextErrorEngineeringLayer = _error_engineering.ContextErrorEngineeringLayer


# Optional Agents SDK imports. The module still works for notebook preparation and
# offline testing if the SDK is not installed.
try:
    from agents import Agent, AgentHooks, RunContextWrapper
    try:
        from agents import AgentHookContext
    except Exception:
        AgentHookContext = RunContextWrapper
except Exception:  # pragma: no cover - graceful fallback for environments without the SDK
    Agent = Any  # type: ignore[misc,assignment]

    class AgentHooks:  # type: ignore[no-redef]
        pass

    class RunContextWrapper:  # type: ignore[no-redef]
        def __init__(self, context: Any, turn_input: Any = ""):
            self.context = context
            self.turn_input = turn_input

    AgentHookContext = RunContextWrapper


PRAGMATIST_HYBRID_MEMORY_POLICY = """
<persona_context_usage>
Use the routing labels in the YAML frontmatter to choose which memory slices matter,
not as standalone evidence.

When reasoning:
- GLOBAL memory should carry stable anchor personas and approved synthetic examples.
- SESSION memory should carry fresh observations, current-run overrides, and review queues.
- Pending synthetic records can guide follow-up questions but should not outweigh anchor facts.
- If the context packet shows missing topic support or weak grounding, prefer a grounded follow-up
  over filling gaps with invented details.
- Treat anchor summaries as the safest source when there is any disagreement.
</persona_context_usage>
""".strip()

CONTEXT_ERROR_RESPONSE_POLICY = """
<context_error_response_policy>
If runtime controls require grounded follow-up, or the current packet diagnostics show weak grounding,
missing context, or persona mismatch:
- explicitly state what is missing
- ask the smallest grounded follow-up that would resolve the gap
- do not invent persona details to bridge the gap
</context_error_response_policy>
""".strip()


DEFAULT_RUNTIME_CONTROLS = {
    "allow_global_memory": True,
    "allow_session_memory": True,
    "allow_pending_records": False,
    "allow_synthetic_expansions": True,
    "max_context_global_k": None,
    "max_context_session_k": None,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_context_error_engineering_layer(config: Optional[PipelineConfig] = None) -> ContextErrorEngineeringLayer:
    cfg = config or PipelineConfig()
    ee_config = ContextErrorEngineeringConfig(
        context_token_budget=cfg.context_token_budget,
        default_context_global_k=cfg.context_global_k,
        default_context_session_k=cfg.context_session_k,
    )
    return ContextErrorEngineeringLayer(config=ee_config)


@dataclass
class PragmatistHybridState(HybridPersonaState):
    """
    Extension of HybridPersonaState that preserves the Pragmatist notebook's
    broader runtime concerns.
    """

    trip_history: Dict[str, Any] = field(default_factory=lambda: {"trips": []})
    runtime: Dict[str, Any] = field(
        default_factory=lambda: {
            "default_strategy": "anchor_priority",
            "last_query": "",
            "last_context_strategy": None,
            "last_context_packet": None,
            "context_history": [],
            "context_errors": [],
            "context_error_layer": {
                "turn_log": [],
                "event_log": [],
                "summary": {},
                "recommendations": [],
                "history_cursor": 0,
                "error_cursor": 0,
            },
            "last_context_error_snapshot": None,
            "last_context_error_action_report": None,
            "coverage_gap_queue": [],
            "require_grounded_followup_on_weak_packets": False,
        }
    )
    user_controls: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_RUNTIME_CONTROLS))


def _coerce_note(note: HybridMemoryNote | Dict[str, Any]) -> HybridMemoryNote:
    if isinstance(note, HybridMemoryNote):
        return note
    return HybridMemoryNote(**note)


def _extract_user_text(turn_input: Any) -> str:
    if isinstance(turn_input, str):
        return turn_input.strip()

    if isinstance(turn_input, dict):
        content = turn_input.get("content") or turn_input.get("text") or ""
        return str(content).strip()

    if isinstance(turn_input, list):
        if not turn_input:
            return ""
        last_item = turn_input[-1]
        if isinstance(last_item, dict):
            content = last_item.get("content") or last_item.get("text") or ""
            return str(content).strip()
        return str(last_item).strip()

    return str(turn_input or "").strip()


def _is_pending(note: HybridMemoryNote) -> bool:
    return note.metadata.get("review_status") == "pending"


def _is_synthetic(note: HybridMemoryNote) -> bool:
    return note.source_mode in {"personahub_zero_shot", "personahub_few_shot", "hybrid"}


def _clone_config_with_controls(config: Optional[PipelineConfig], state: PragmatistHybridState) -> PipelineConfig:
    base = copy.deepcopy(config or PipelineConfig())
    controls = state.user_controls or {}
    if controls.get("max_context_global_k") is not None:
        base.context_global_k = int(controls["max_context_global_k"])
    if controls.get("max_context_session_k") is not None:
        base.context_session_k = int(controls["max_context_session_k"])
    return base


def _rebuild_frontmatter(
    query: str,
    predicted_labels: Dict[str, Any],
    strategy: str,
    selected_global: Sequence[HybridMemoryNote],
    selected_session: Sequence[HybridMemoryNote],
) -> str:
    payload = {
        "query": query,
        "predicted_labels": {
            "persona_macro": predicted_labels.get("persona_macro"),
            "intent_level": predicted_labels.get("intent_level"),
            "topic_product_category": predicted_labels.get("topic_product_category"),
        },
        "strategy": strategy,
        "global_notes_selected": len(list(selected_global)),
        "session_notes_selected": len(list(selected_session)),
    }
    return render_frontmatter(payload)


def _apply_user_controls(
    packet: ContextPacket,
    state: PragmatistHybridState,
    strategy: str,
    config: Optional[PipelineConfig] = None,
) -> ContextPacket:
    config = config or PipelineConfig()
    controls = state.user_controls or {}

    selected_global = [_coerce_note(n) for n in packet.selected_global]
    selected_session = [_coerce_note(n) for n in packet.selected_session]

    if not controls.get("allow_global_memory", True):
        selected_global = []

    if not controls.get("allow_session_memory", True):
        selected_session = []

    if not controls.get("allow_pending_records", False):
        selected_session = [n for n in selected_session if not _is_pending(n)]

    if not controls.get("allow_synthetic_expansions", True):
        selected_global = [n for n in selected_global if not _is_synthetic(n) or n.note_type == "anchor"]
        selected_session = [n for n in selected_session if not _is_synthetic(n)]

    packet.selected_global = selected_global
    packet.selected_session = selected_session
    packet.global_memories_md = render_notes_md(selected_global, k=config.context_global_k)
    packet.session_memories_md = render_notes_md(selected_session, k=config.context_session_k)
    packet.frontmatter = _rebuild_frontmatter(
        query=packet.query,
        predicted_labels=packet.predicted_labels,
        strategy=strategy,
        selected_global=selected_global,
        selected_session=selected_session,
    )
    packet.audit = audit_context_packet(packet, config=config)
    packet.audit["strategy"] = strategy
    packet.audit["controls_applied"] = {
        "allow_global_memory": controls.get("allow_global_memory", True),
        "allow_session_memory": controls.get("allow_session_memory", True),
        "allow_pending_records": controls.get("allow_pending_records", False),
        "allow_synthetic_expansions": controls.get("allow_synthetic_expansions", True),
        "max_context_global_k": controls.get("max_context_global_k"),
        "max_context_session_k": controls.get("max_context_session_k"),
    }
    return packet


def sync_context_error_engineering(
    state: PragmatistHybridState,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    layer = make_context_error_engineering_layer(config=config)
    snapshot = layer.sync_from_runtime(state.runtime, user_controls=state.user_controls)
    state.runtime["context_error_layer"] = snapshot
    state.runtime["last_context_error_snapshot"] = snapshot.get("summary", {})
    return snapshot


def apply_context_error_engineering_actions(
    state: PragmatistHybridState,
    config: Optional[PipelineConfig] = None,
    *,
    auto_apply: bool = True,
) -> Dict[str, Any]:
    layer = make_context_error_engineering_layer(config=config)
    report = layer.apply_recommendations(
        runtime=state.runtime,
        user_controls=state.user_controls,
        auto_apply=auto_apply,
    )
    if auto_apply:
        sync_context_error_engineering(state, config=config)
    state.runtime["last_context_error_action_report"] = report
    return report


def _log_context_packet(
    state: PragmatistHybridState,
    packet: ContextPacket,
    strategy: str,
    config: Optional[PipelineConfig] = None,
) -> None:
    history = state.runtime.setdefault("context_history", [])
    turn_index = len(history) + 1
    turn_id = f"turn_{turn_index:05d}"

    summary = {
        "turn_id": turn_id,
        "timestamp": _utc_now_iso(),
        "query": packet.query,
        "strategy": strategy,
        "predicted_labels": {
            "persona_macro": packet.predicted_labels.get("persona_macro"),
            "intent_level": packet.predicted_labels.get("intent_level"),
            "topic_product_category": packet.predicted_labels.get("topic_product_category"),
        },
        "audit": packet.audit,
        "selected_global": len(packet.selected_global),
        "selected_session": len(packet.selected_session),
    }
    history.append(summary)
    state.runtime["last_query"] = packet.query
    state.runtime["last_context_strategy"] = strategy
    state.runtime["last_context_packet"] = packet
    if packet.audit.get("issues"):
        state.runtime.setdefault("context_errors", []).append(copy.deepcopy(summary))
    sync_context_error_engineering(state, config=config)


def sync_packet_to_state(
    state: PragmatistHybridState,
    packet: ContextPacket,
    strategy: str,
    config: Optional[PipelineConfig] = None,
) -> PragmatistHybridState:
    state.system_frontmatter = packet.frontmatter
    state.global_memories_md = packet.global_memories_md
    state.session_memories_md = packet.session_memories_md
    state.runtime["last_query"] = packet.query
    state.runtime["last_context_strategy"] = strategy
    state.runtime["last_context_packet"] = packet
    # The original Pragmatist notebook uses this as a one-turn reinjection hint after trimming.
    state.inject_session_memories_next_turn = bool(packet.selected_session)
    _log_context_packet(state, packet, strategy, config=config)
    return state


def build_pragmatist_context_packet(
    state: PragmatistHybridState,
    labeler: HybridLabeler,
    query: str,
    strategy: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
) -> ContextPacket:
    strategy = strategy or state.runtime.get("default_strategy") or "anchor_priority"
    local_config = _clone_config_with_controls(config=config, state=state)
    packet = build_context_packet(
        state=state,
        labeler=labeler,
        query=query,
        strategy=strategy,
        config=local_config,
    )
    packet = _apply_user_controls(packet, state=state, strategy=strategy, config=local_config)
    sync_packet_to_state(state, packet, strategy=strategy, config=local_config)
    return packet


def compare_pragmatist_context_strategies(
    state: PragmatistHybridState,
    labeler: HybridLabeler,
    queries: Sequence[str],
    config: Optional[PipelineConfig] = None,
) -> pd.DataFrame:
    """
    Pragmatist-aware comparison wrapper.

    It preserves the same output columns as the original compare_context_strategies
    function, but applies runtime controls and packet auditing exactly as the bridge
    will do during real turns.
    """
    local_config = _clone_config_with_controls(config=config, state=state)
    rows = []
    for query in queries:
        for strategy in ["relevance_only", "anchor_priority"]:
            temp_state = copy.deepcopy(state)
            packet = build_pragmatist_context_packet(
                state=temp_state,
                labeler=labeler,
                query=query,
                strategy=strategy,
                config=local_config,
            )
            rows.append(
                {
                    "query": query,
                    "strategy": strategy,
                    "issue_count": packet.audit["issue_count"],
                    "issues": ", ".join(packet.audit["issues"]),
                    "estimated_tokens": packet.audit["estimated_tokens"],
                    "anchor_share": packet.audit["anchor_share"],
                    "selected_count": packet.audit["selected_count"],
                }
            )
    return pd.DataFrame(rows)


def _rank_strategy_summary(ab_df: pd.DataFrame) -> pd.DataFrame:
    if ab_df.empty:
        return pd.DataFrame(columns=["strategy", "mean_issue_count", "mean_estimated_tokens", "mean_anchor_share", "mean_selected_count"])
    summary = (
        ab_df.groupby("strategy", dropna=False)
        .agg(
            mean_issue_count=("issue_count", "mean"),
            mean_estimated_tokens=("estimated_tokens", "mean"),
            mean_anchor_share=("anchor_share", "mean"),
            mean_selected_count=("selected_count", "mean"),
        )
        .reset_index()
    )
    summary = summary.sort_values(
        by=["mean_issue_count", "mean_estimated_tokens", "mean_anchor_share"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    return summary


def choose_recommended_context_strategy(ab_df: pd.DataFrame) -> str:
    summary = _rank_strategy_summary(ab_df)
    if summary.empty:
        return "anchor_priority"

    best = summary.iloc[0]
    # When metrics are tied, preserve the anchor-priority bias for safer grounding.
    tied = summary[
        (summary["mean_issue_count"] == best["mean_issue_count"])
        & (summary["mean_estimated_tokens"] == best["mean_estimated_tokens"])
    ]
    if "anchor_priority" in tied["strategy"].tolist():
        return "anchor_priority"
    return str(best["strategy"])


def evaluate_context_strategies(
    state: PragmatistHybridState,
    labeler: HybridLabeler,
    queries: Sequence[str],
    config: Optional[PipelineConfig] = None,
    persist_default: bool = True,
) -> Dict[str, Any]:
    local_config = _clone_config_with_controls(config=config, state=state)
    ab_df = compare_pragmatist_context_strategies(state, labeler, queries=queries, config=local_config)
    summary = _rank_strategy_summary(ab_df)
    recommended = choose_recommended_context_strategy(ab_df)

    report = {
        "recommended_strategy": recommended,
        "summary": summary.to_dict(orient="records"),
        "results": ab_df.to_dict(orient="records"),
    }
    state.eval_memory.setdefault("ab_tests", []).append(report)
    if persist_default:
        state.runtime["default_strategy"] = recommended
    return report


class PragmatistHybridHooks(AgentHooks):
    """
    Drop-in on-start hook that hydrates state with a context packet each turn.
    """

    def __init__(
        self,
        labeler: HybridLabeler,
        config: Optional[PipelineConfig] = None,
        strategy_resolver: Optional[Callable[[PragmatistHybridState, str], str]] = None,
    ):
        self.labeler = labeler
        self.config = config or PipelineConfig()
        self.strategy_resolver = strategy_resolver

    async def on_start(self, ctx: AgentHookContext, agent: Agent) -> None:
        query = _extract_user_text(getattr(ctx, "turn_input", ""))
        state = ctx.context
        if not isinstance(state, PragmatistHybridState):
            raise TypeError("PragmatistHybridHooks expects a PragmatistHybridState context object.")
        strategy = (
            self.strategy_resolver(state, query)
            if self.strategy_resolver is not None
            else state.runtime.get("default_strategy", "anchor_priority")
        )
        build_pragmatist_context_packet(state, self.labeler, query=query, strategy=strategy, config=self.config)


def render_context_diagnostics(packet: Optional[ContextPacket]) -> str:
    if packet is None:
        return ""
    issues = packet.audit.get("issues", [])
    if not issues:
        return "<context_diagnostics>none</context_diagnostics>"
    issue_lines = "\n".join(f"- {issue}" for issue in issues)
    return f"<context_diagnostics>\n{issue_lines}\n</context_diagnostics>"


def make_pragmatist_hybrid_instructions(
    base_instructions: str,
    extra_policy: str = "",
    include_diagnostics: bool = True,
) -> Callable[[RunContextWrapper, Agent], Any]:
    """
    Factory for a dynamic instructions function compatible with the Agents SDK.
    The hook should populate state.system_frontmatter/global_memories_md/session_memories_md
    before this function is called.
    """

    async def instructions(ctx: RunContextWrapper, agent: Agent) -> str:
        state = ctx.context
        if not isinstance(state, PragmatistHybridState):
            raise TypeError("Pragmatist hybrid instructions expect a PragmatistHybridState context object.")

        packet = state.runtime.get("last_context_packet")
        diagnostics_block = ""
        if include_diagnostics:
            diagnostics_block = "\n\n" + render_context_diagnostics(packet)

        session_block = ""
        if state.session_memories_md:
            session_block = (
                "\n\n<session_memory>\n"
                + state.session_memories_md
                + "\n</session_memory>"
            )

        prompt = (
            base_instructions.strip()
            + "\n\n<hybrid_frontmatter>\n"
            + (state.system_frontmatter or "---\nquery: unknown\n---")
            + "\n</hybrid_frontmatter>"
            + "\n\n<global_memory>\n"
            + (state.global_memories_md or "- (none)")
            + "\n</global_memory>"
            + session_block
            + "\n\n"
            + HYBRID_MEMORY_POLICY
            + "\n\n"
            + PRAGMATIST_HYBRID_MEMORY_POLICY
            + diagnostics_block
        )

        if state.runtime.get("require_grounded_followup_on_weak_packets", False):
            prompt += "\n\n" + CONTEXT_ERROR_RESPONSE_POLICY

        if extra_policy.strip():
            prompt += "\n\n" + extra_policy.strip()

        return prompt

    return instructions


def save_runtime_note(
    state: PragmatistHybridState,
    text: str,
    keywords: Optional[Sequence[str]] = None,
    *,
    importance: int = 1,
    note_type: str = "runtime_observation",
    source_mode: str = "observed",
    anchor_persona_id: Optional[str] = None,
    topic_product_category: Optional[str] = None,
    intent_level: Optional[str] = None,
    persona_macro: Optional[str] = None,
    review_status: Optional[str] = None,
) -> HybridMemoryNote:
    note = HybridMemoryNote(
        text=text,
        last_update_date=utc_today_iso(),
        keywords=list(keywords or []),
        importance=importance,
        note_type=note_type,
        source_mode=source_mode,
        anchor_persona_id=anchor_persona_id,
        topic_product_category=topic_product_category,
        intent_level=intent_level,
        persona_macro=persona_macro,
        metadata={"review_status": review_status} if review_status else {},
    )
    notes = state.session_memory.setdefault("notes", [])
    notes.append(note.to_dict())
    state.inject_session_memories_next_turn = True
    return note


def consolidate_pragmatist_hybrid_state(
    state: PragmatistHybridState,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    report = consolidate_state(state, config=config)
    state.eval_memory.setdefault("consolidation_reports", []).append(report)
    return report


def wrap_pipeline_state(
    pipeline: BalancedHybridPipeline,
    *,
    trip_history: Optional[Dict[str, Any]] = None,
    user_controls: Optional[Dict[str, Any]] = None,
    default_strategy: str = "anchor_priority",
) -> PragmatistHybridState:
    base_state = pipeline.state
    state = PragmatistHybridState(
        profile=copy.deepcopy(base_state.profile),
        anchor_registry=copy.deepcopy(base_state.anchor_registry),
        global_memory=copy.deepcopy(base_state.global_memory),
        session_memory=copy.deepcopy(base_state.session_memory),
        eval_memory=copy.deepcopy(base_state.eval_memory),
        system_frontmatter=base_state.system_frontmatter,
        global_memories_md=base_state.global_memories_md,
        session_memories_md=base_state.session_memories_md,
        inject_session_memories_next_turn=base_state.inject_session_memories_next_turn,
        trip_history=copy.deepcopy(trip_history or {"trips": []}),
    )
    state.runtime["default_strategy"] = default_strategy
    if user_controls:
        merged = copy.deepcopy(DEFAULT_RUNTIME_CONTROLS)
        merged.update(user_controls)
        state.user_controls = merged
    sync_context_error_engineering(state, config=pipeline.config)
    return state


class PragmatistHybridBridge:
    """
    Thin orchestration layer that makes the balanced hybrid module feel native
    inside the Pragmatist-style notebook pipeline.
    """

    def __init__(
        self,
        state: PragmatistHybridState,
        labeler: HybridLabeler,
        config: Optional[PipelineConfig] = None,
    ):
        self.state = state
        self.labeler = labeler
        self.config = config or PipelineConfig()

    @classmethod
    def from_pipeline(
        cls,
        pipeline: BalancedHybridPipeline,
        *,
        trip_history: Optional[Dict[str, Any]] = None,
        user_controls: Optional[Dict[str, Any]] = None,
        default_strategy: str = "anchor_priority",
    ) -> "PragmatistHybridBridge":
        state = wrap_pipeline_state(
            pipeline,
            trip_history=trip_history,
            user_controls=user_controls,
            default_strategy=default_strategy,
        )
        return cls(state=state, labeler=pipeline.labeler, config=pipeline.config)

    @classmethod
    def from_codebook(
        cls,
        codebook_path: str | Path,
        *,
        state: Optional[PragmatistHybridState] = None,
        config: Optional[PipelineConfig] = None,
    ) -> "PragmatistHybridBridge":
        codebook_df = load_codebook_csv(codebook_path)
        labeler = HybridLabeler(codebook_df)
        bridge = cls(state=state or PragmatistHybridState(), labeler=labeler, config=config)
        bridge.sync_context_error_layer()
        return bridge

    def hydrate_turn(self, query: str, strategy: Optional[str] = None) -> ContextPacket:
        return build_pragmatist_context_packet(
            state=self.state,
            labeler=self.labeler,
            query=query,
            strategy=strategy,
            config=self.config,
        )

    def evaluate_strategies(self, queries: Sequence[str], persist_default: bool = True) -> Dict[str, Any]:
        return evaluate_context_strategies(
            state=self.state,
            labeler=self.labeler,
            queries=queries,
            config=self.config,
            persist_default=persist_default,
        )

    def make_hooks(
        self,
        strategy_resolver: Optional[Callable[[PragmatistHybridState, str], str]] = None,
    ) -> PragmatistHybridHooks:
        return PragmatistHybridHooks(
            labeler=self.labeler,
            config=self.config,
            strategy_resolver=strategy_resolver,
        )

    def make_instructions(
        self,
        base_instructions: str,
        extra_policy: str = "",
        include_diagnostics: bool = True,
    ) -> Callable[[RunContextWrapper, Agent], Any]:
        return make_pragmatist_hybrid_instructions(
            base_instructions=base_instructions,
            extra_policy=extra_policy,
            include_diagnostics=include_diagnostics,
        )

    def consolidate(self) -> Dict[str, Any]:
        return consolidate_pragmatist_hybrid_state(self.state, config=self.config)

    def sync_context_error_layer(self) -> Dict[str, Any]:
        return sync_context_error_engineering(self.state, config=self.config)

    def context_error_turn_frame(self) -> pd.DataFrame:
        snapshot = self.sync_context_error_layer()
        layer = make_context_error_engineering_layer(self.config)
        return layer.turn_frame_from_layer(snapshot)

    def context_error_event_frame(self) -> pd.DataFrame:
        snapshot = self.sync_context_error_layer()
        layer = make_context_error_engineering_layer(self.config)
        return layer.event_frame_from_layer(snapshot)

    def apply_context_error_actions(
        self,
        *,
        auto_apply: bool = True,
        run_ab_on_error_queries: bool = True,
        max_queries: int = 5,
    ) -> Dict[str, Any]:
        snapshot = self.sync_context_error_layer()
        report = apply_context_error_engineering_actions(
            self.state,
            config=self.config,
            auto_apply=auto_apply,
        )

        if auto_apply and run_ab_on_error_queries:
            top_queries = [row["query"] for row in snapshot.get("summary", {}).get("top_error_queries", [])[:max_queries] if row.get("query")]
            if top_queries:
                strategy_report = evaluate_context_strategies(
                    state=self.state,
                    labeler=self.labeler,
                    queries=top_queries,
                    config=self.config,
                    persist_default=True,
                )
                report["strategy_report"] = {
                    "recommended_strategy": strategy_report["recommended_strategy"],
                    "summary": strategy_report["summary"],
                }

        refreshed = self.sync_context_error_layer()
        report["context_error_summary"] = refreshed.get("summary", {})
        self.state.runtime["last_context_error_action_report"] = report
        return report

    def export_context_error_artifacts(self, output_dir: str | Path) -> Dict[str, str]:
        snapshot = self.sync_context_error_layer()
        layer = make_context_error_engineering_layer(self.config)
        return layer.export_artifacts(snapshot, output_dir)

    def export_context_history(self, path: str | Path) -> Path:
        path = Path(path)
        payload = {
            "runtime": self.state.runtime,
            "eval_memory": self.state.eval_memory,
        }

        def _json_default(obj: Any) -> Any:
            if isinstance(obj, ContextPacket):
                return {
                    "query": obj.query,
                    "predicted_labels": obj.predicted_labels,
                    "audit": obj.audit,
                }
            if isinstance(obj, HybridMemoryNote):
                return obj.to_dict()
            return str(obj)

        path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        return path


def demo_integration(
    codebook_path: str | Path,
    schema_path: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    """
    Small executable demo that starts from the advanced balanced pipeline and then
    wraps it in the Pragmatist bridge.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    codebook_df = _balanced.load_codebook_csv(codebook_path)
    schema = _balanced.load_json_schema(schema_path)
    pipeline = BalancedHybridPipeline(codebook_df, schema)
    pipeline.register_anchors(_balanced.demo_anchor_cards())
    pipeline.ingest_observed_records(_balanced.demo_observed_records(), source_mode="observed", assign_anchor=False)
    candidates = pipeline.generate_candidates()
    selected = pipeline.rebalance(candidates)
    pipeline.consolidate(selected)

    bridge = PragmatistHybridBridge.from_pipeline(pipeline)
    queries = [
        "best zero trust access tools for remote employees with sso",
        "student discount note taking app with offline sync",
        "edr pricing for 100 endpoints",
    ]
    ab_report = bridge.evaluate_strategies(queries)
    packet = bridge.hydrate_turn(queries[0])
    consolidation = bridge.consolidate()
    context_error_snapshot = bridge.sync_context_error_layer()
    context_error_action_report = bridge.apply_context_error_actions(auto_apply=True)

    history_path = bridge.export_context_history(output_dir / "pragmatist_context_history.json")
    pd.DataFrame(ab_report["results"]).to_csv(output_dir / "pragmatist_ab_results.csv", index=False)
    (output_dir / "pragmatist_ab_summary.json").write_text(json.dumps(ab_report, indent=2), encoding="utf-8")
    (output_dir / "pragmatist_packet_audit.json").write_text(json.dumps(packet.audit, indent=2), encoding="utf-8")
    (output_dir / "pragmatist_consolidation.json").write_text(json.dumps(consolidation, indent=2), encoding="utf-8")
    (output_dir / "pragmatist_context_error_snapshot.json").write_text(json.dumps(context_error_snapshot, indent=2, default=str), encoding="utf-8")
    (output_dir / "pragmatist_context_error_actions.json").write_text(json.dumps(context_error_action_report, indent=2, default=str), encoding="utf-8")
    bridge.export_context_error_artifacts(output_dir)

    return {
        "recommended_strategy": ab_report["recommended_strategy"],
        "packet_audit": packet.audit,
        "history_path": str(history_path),
        "context_error_health_score": context_error_snapshot.get("summary", {}).get("health_score"),
    }


__all__ = [
    "PragmatistHybridState",
    "PragmatistHybridBridge",
    "PragmatistHybridHooks",
    "PRAGMATIST_HYBRID_MEMORY_POLICY",
    "CONTEXT_ERROR_RESPONSE_POLICY",
    "ContextErrorEngineeringConfig",
    "ContextErrorEngineeringLayer",
    "build_pragmatist_context_packet",
    "build_context_packet",
    "compare_context_strategies",
    "compare_pragmatist_context_strategies",
    "evaluate_context_strategies",
    "choose_recommended_context_strategy",
    "make_pragmatist_hybrid_instructions",
    "save_runtime_note",
    "consolidate_pragmatist_hybrid_state",
    "wrap_pipeline_state",
    "sync_context_error_engineering",
    "apply_context_error_engineering_actions",
    "make_context_error_engineering_layer",
    "demo_integration",
]
