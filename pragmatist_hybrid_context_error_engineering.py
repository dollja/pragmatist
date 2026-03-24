
from __future__ import annotations

"""
Dedicated context-error engineering layer for the Pragmatist × Hybrid Persona pipeline.

This module consumes:
- runtime["context_history"]: all hydrated context packets
- runtime["context_errors"]: issue-bearing packet summaries

It normalizes these streams into an auditable turn log and issue-event log, computes
dashboard-style summaries, recommends mitigation actions, and can export artifacts
for notebook review.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import json

import pandas as pd


ISSUE_FAMILY = {
    "missing_context": "coverage",
    "missing_topic_support": "coverage",
    "weak_anchor_grounding": "grounding",
    "persona_mismatch": "grounding",
    "topic_leakage": "precision",
    "pending_context_risk": "safety",
    "context_overload": "budget",
}

ISSUE_BASE_SEVERITY = {
    "missing_context": "critical",
    "missing_topic_support": "high",
    "weak_anchor_grounding": "medium",
    "persona_mismatch": "high",
    "topic_leakage": "medium",
    "pending_context_risk": "high",
    "context_overload": "high",
}

SEVERITY_SCORE = {"low": 1, "medium": 2, "high": 3, "critical": 4}

ISSUE_ACTIONS = {
    "missing_context": [
        "queue_coverage_gap",
        "prefer_grounded_followup",
    ],
    "missing_topic_support": [
        "queue_coverage_gap",
        "prefer_grounded_followup",
    ],
    "weak_anchor_grounding": [
        "prefer_anchor_priority",
        "reduce_synthetic_weight",
    ],
    "persona_mismatch": [
        "prefer_anchor_priority",
        "tighten_persona_filtering",
    ],
    "topic_leakage": [
        "shrink_context_window",
        "tighten_topic_filtering",
    ],
    "pending_context_risk": [
        "disable_pending_records",
        "review_pending_before_injection",
    ],
    "context_overload": [
        "shrink_context_window",
        "prune_low_value_notes",
    ],
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _json_key(payload: Any) -> str:
    try:
        return json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        return str(payload)


@dataclass
class ContextErrorEngineeringConfig:
    context_token_budget: int = 1200
    default_context_global_k: int = 8
    default_context_session_k: int = 6
    min_events_for_control_action: int = 2
    error_rate_alert_threshold: float = 0.30
    dominant_issue_threshold: float = 0.35
    critical_anchor_share: float = 0.08
    weak_anchor_share: float = 0.12
    high_overload_ratio: float = 1.20
    critical_overload_ratio: float = 1.40
    max_recent_turns: int = 50
    max_recent_events: int = 100
    max_top_queries: int = 10


@dataclass
class ContextErrorEngineeringLayer:
    config: ContextErrorEngineeringConfig = field(default_factory=ContextErrorEngineeringConfig)

    def turn_signature(self, turn: Dict[str, Any], idx: Optional[int] = None) -> str:
        if turn.get("turn_id"):
            return str(turn["turn_id"])
        query = str(turn.get("query", "")).strip()
        strategy = str(turn.get("strategy", "")).strip()
        timestamp = str(turn.get("timestamp", "")).strip()
        audit = turn.get("audit", {}) or {}
        issues = sorted(_safe_list(audit.get("issues", [])))
        base = {"query": query, "strategy": strategy, "timestamp": timestamp, "issues": issues}
        sig = _json_key(base)
        if idx is not None:
            return f"{idx:05d}:{sig}"
        return sig

    def _error_signature_set(self, context_errors: Sequence[Dict[str, Any]]) -> set[str]:
        error_set = set()
        for idx, item in enumerate(context_errors, start=1):
            error_set.add(self.turn_signature(item, idx=idx))
            if item.get("turn_id"):
                error_set.add(str(item["turn_id"]))
        return error_set

    def build_turn_frame(
        self,
        context_history: Sequence[Dict[str, Any]],
        context_errors: Sequence[Dict[str, Any]],
    ) -> pd.DataFrame:
        error_set = self._error_signature_set(context_errors)
        rows: List[Dict[str, Any]] = []

        for idx, turn in enumerate(context_history, start=1):
            audit = turn.get("audit", {}) or {}
            predicted = turn.get("predicted_labels", {}) or {}
            issues = [str(x) for x in _safe_list(audit.get("issues", [])) if str(x)]
            turn_id = str(turn.get("turn_id") or f"turn_{idx:05d}")
            signature = self.turn_signature(turn, idx=idx)

            rows.append(
                {
                    "turn_index": idx,
                    "turn_id": turn_id,
                    "timestamp": str(turn.get("timestamp", "")),
                    "query": str(turn.get("query", "")),
                    "strategy": str(turn.get("strategy", "")),
                    "issue_count": _safe_int(audit.get("issue_count"), len(issues)),
                    "issues": issues,
                    "estimated_tokens": _safe_int(audit.get("estimated_tokens"), 0),
                    "anchor_share": _safe_float(audit.get("anchor_share"), 0.0),
                    "selected_count": _safe_int(audit.get("selected_count"), 0),
                    "selected_global": _safe_int(turn.get("selected_global"), 0),
                    "selected_session": _safe_int(turn.get("selected_session"), 0),
                    "topics": [str(x) for x in _safe_list(audit.get("topics", [])) if str(x)],
                    "persona_macro": predicted.get("persona_macro"),
                    "intent_level": predicted.get("intent_level"),
                    "topic_product_category": predicted.get("topic_product_category"),
                    "controls_applied": audit.get("controls_applied", {}) or {},
                    "is_error_turn": bool(issues) or (signature in error_set) or (turn_id in error_set),
                    "error_stream_hit": (signature in error_set) or (turn_id in error_set),
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "turn_index", "turn_id", "timestamp", "query", "strategy",
                    "issue_count", "issues", "estimated_tokens", "anchor_share",
                    "selected_count", "selected_global", "selected_session", "topics",
                    "persona_macro", "intent_level", "topic_product_category",
                    "controls_applied", "is_error_turn", "error_stream_hit",
                ]
            )

        return pd.DataFrame(rows)

    def _score_issue_severity(self, issue_code: str, turn_row: Dict[str, Any]) -> str:
        base = ISSUE_BASE_SEVERITY.get(issue_code, "medium")
        anchor_share = _safe_float(turn_row.get("anchor_share"), 0.0)
        selected_count = _safe_int(turn_row.get("selected_count"), 0)
        tokens = _safe_int(turn_row.get("estimated_tokens"), 0)
        overload_ratio = tokens / max(1, self.config.context_token_budget)

        if issue_code == "missing_context":
            return "critical"
        if issue_code == "missing_topic_support":
            if selected_count == 0:
                return "critical"
            return "high"
        if issue_code == "weak_anchor_grounding":
            if anchor_share < self.config.critical_anchor_share:
                return "high"
            if anchor_share < self.config.weak_anchor_share:
                return "medium"
            return "low"
        if issue_code == "context_overload":
            if overload_ratio >= self.config.critical_overload_ratio:
                return "critical"
            if overload_ratio >= self.config.high_overload_ratio:
                return "high"
            return "medium"
        if issue_code == "persona_mismatch":
            if anchor_share < self.config.weak_anchor_share:
                return "critical"
            return "high"
        if issue_code == "pending_context_risk":
            controls = turn_row.get("controls_applied", {}) or {}
            if controls.get("allow_pending_records", True):
                return "high"
            return "medium"
        if issue_code == "topic_leakage":
            topics = _safe_list(turn_row.get("topics", []))
            if len(topics) > 4:
                return "high"
            return "medium"
        return base

    def build_event_frame(self, turn_df: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        if turn_df.empty:
            return pd.DataFrame(
                columns=[
                    "event_index", "turn_index", "turn_id", "timestamp", "query", "strategy",
                    "issue_code", "issue_family", "severity", "severity_score",
                    "recommended_actions", "estimated_tokens", "anchor_share", "selected_count",
                    "persona_macro", "intent_level", "topic_product_category", "topics",
                ]
            )

        event_index = 0
        for row in turn_df.to_dict(orient="records"):
            for issue_code in row.get("issues", []):
                event_index += 1
                severity = self._score_issue_severity(issue_code, row)
                rows.append(
                    {
                        "event_index": event_index,
                        "turn_index": row["turn_index"],
                        "turn_id": row["turn_id"],
                        "timestamp": row["timestamp"],
                        "query": row["query"],
                        "strategy": row["strategy"],
                        "issue_code": issue_code,
                        "issue_family": ISSUE_FAMILY.get(issue_code, "other"),
                        "severity": severity,
                        "severity_score": SEVERITY_SCORE.get(severity, 2),
                        "recommended_actions": ISSUE_ACTIONS.get(issue_code, []),
                        "estimated_tokens": row["estimated_tokens"],
                        "anchor_share": row["anchor_share"],
                        "selected_count": row["selected_count"],
                        "persona_macro": row["persona_macro"],
                        "intent_level": row["intent_level"],
                        "topic_product_category": row["topic_product_category"],
                        "topics": row["topics"],
                    }
                )

        return pd.DataFrame(rows)

    def _group_strategy_summary(self, turn_df: pd.DataFrame) -> List[Dict[str, Any]]:
        if turn_df.empty:
            return []
        rows: List[Dict[str, Any]] = []
        for strategy, group in turn_df.groupby("strategy", dropna=False):
            total_turns = int(len(group))
            error_turns = int(group["is_error_turn"].sum())
            issue_events = int(group["issue_count"].sum())
            top_issues = (
                Counter([issue for issues in group["issues"].tolist() for issue in issues])
                .most_common(5)
            )
            rows.append(
                {
                    "strategy": strategy if pd.notna(strategy) else None,
                    "total_turns": total_turns,
                    "error_turns": error_turns,
                    "error_turn_rate": round(error_turns / max(1, total_turns), 3),
                    "issue_events": issue_events,
                    "mean_estimated_tokens": round(float(group["estimated_tokens"].mean()), 2),
                    "mean_anchor_share": round(float(group["anchor_share"].mean()), 3),
                    "top_issues": [{"issue_code": code, "count": count} for code, count in top_issues],
                }
            )
        return sorted(rows, key=lambda x: (x["error_turn_rate"], x["mean_estimated_tokens"], -x["mean_anchor_share"]))

    def _group_label_hotspots(self, turn_df: pd.DataFrame, column: str) -> List[Dict[str, Any]]:
        error_df = turn_df[turn_df["is_error_turn"] == True]
        if error_df.empty or column not in error_df.columns:
            return []
        grouped = (
            error_df.groupby(column, dropna=False)
            .agg(
                error_turns=("turn_id", "count"),
                issue_events=("issue_count", "sum"),
                mean_anchor_share=("anchor_share", "mean"),
                mean_estimated_tokens=("estimated_tokens", "mean"),
            )
            .reset_index()
        )
        grouped = grouped.sort_values(
            by=["error_turns", "issue_events", "mean_anchor_share"],
            ascending=[False, False, True],
        )
        rows = []
        for row in grouped.to_dict(orient="records"):
            rows.append(
                {
                    "label": row.get(column) if pd.notna(row.get(column)) else None,
                    "error_turns": int(row["error_turns"]),
                    "issue_events": int(row["issue_events"]),
                    "mean_anchor_share": round(float(row["mean_anchor_share"]), 3),
                    "mean_estimated_tokens": round(float(row["mean_estimated_tokens"]), 2),
                }
            )
        return rows

    def _build_coverage_gap_queue(self, turn_df: pd.DataFrame) -> List[Dict[str, Any]]:
        if turn_df.empty:
            return []
        gap_turns = []
        for row in turn_df.to_dict(orient="records"):
            issues = set(row.get("issues", []))
            if "missing_context" not in issues and "missing_topic_support" not in issues:
                continue
            gap_turns.append(
                {
                    "query": row["query"],
                    "predicted_labels": {
                        "persona_macro": row["persona_macro"],
                        "intent_level": row["intent_level"],
                        "topic_product_category": row["topic_product_category"],
                    },
                    "issue_codes": sorted(list(issues & {"missing_context", "missing_topic_support"})),
                }
            )
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for item in gap_turns:
            key = _json_key(item)
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped

    def summarize(
        self,
        turn_df: pd.DataFrame,
        event_df: pd.DataFrame,
        context_history: Sequence[Dict[str, Any]],
        context_errors: Sequence[Dict[str, Any]],
        runtime: Optional[Dict[str, Any]] = None,
        user_controls: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        history_count = len(context_history)
        error_stream_count = len(context_errors)
        issue_turn_count = int(turn_df["is_error_turn"].sum()) if not turn_df.empty else 0
        event_count = int(len(event_df))
        error_turn_rate = round(issue_turn_count / max(1, history_count), 3)

        issue_counts = Counter(event_df["issue_code"].tolist()) if not event_df.empty else Counter()
        family_counts = Counter(event_df["issue_family"].tolist()) if not event_df.empty else Counter()
        severity_counts = Counter(event_df["severity"].tolist()) if not event_df.empty else Counter()

        top_error_queries: List[Dict[str, Any]] = []
        if not turn_df.empty:
            error_turn_df = turn_df[turn_df["is_error_turn"] == True]
            if not error_turn_df.empty:
                grouped = (
                    error_turn_df.groupby("query", dropna=False)
                    .agg(
                        error_turns=("turn_id", "count"),
                        issue_events=("issue_count", "sum"),
                        mean_estimated_tokens=("estimated_tokens", "mean"),
                        mean_anchor_share=("anchor_share", "mean"),
                    )
                    .reset_index()
                )
                grouped = grouped.sort_values(
                    by=["error_turns", "issue_events", "mean_anchor_share"],
                    ascending=[False, False, True],
                )
                for row in grouped.head(self.config.max_top_queries).to_dict(orient="records"):
                    top_error_queries.append(
                        {
                            "query": row["query"],
                            "error_turns": int(row["error_turns"]),
                            "issue_events": int(row["issue_events"]),
                            "mean_estimated_tokens": round(float(row["mean_estimated_tokens"]), 2),
                            "mean_anchor_share": round(float(row["mean_anchor_share"]), 3),
                        }
                    )

        history_ids = set(turn_df["turn_id"].tolist()) if not turn_df.empty else set()
        error_ids = set()
        for idx, item in enumerate(context_errors, start=1):
            if item.get("turn_id"):
                error_ids.add(str(item["turn_id"]))
            else:
                error_ids.add(self.turn_signature(item, idx=idx))
        matched_error_turns = int(turn_df["error_stream_hit"].sum()) if not turn_df.empty else 0
        orphan_error_count = max(0, len(error_ids - history_ids))
        error_stream_alignment = round(matched_error_turns / max(1, issue_turn_count), 3) if issue_turn_count else 1.0

        strategy_summary = self._group_strategy_summary(turn_df)
        persona_hotspots = self._group_label_hotspots(turn_df, "persona_macro")
        intent_hotspots = self._group_label_hotspots(turn_df, "intent_level")
        topic_hotspots = self._group_label_hotspots(turn_df, "topic_product_category")
        coverage_gap_queue = self._build_coverage_gap_queue(turn_df)

        dominant_issue = None
        if event_count:
            code, count = issue_counts.most_common(1)[0]
            if (count / max(1, event_count)) >= self.config.dominant_issue_threshold:
                dominant_issue = {"issue_code": code, "count": int(count), "share": round(count / max(1, event_count), 3)}

        recent_turns = turn_df.tail(self.config.max_recent_turns).to_dict(orient="records") if not turn_df.empty else []
        recent_events = event_df.tail(self.config.max_recent_events).to_dict(orient="records") if not event_df.empty else []

        health_score = max(
            0.0,
            round(
                100.0
                - (error_turn_rate * 45.0)
                - (sum(SEVERITY_SCORE.get(k, 0) * v for k, v in severity_counts.items()) * 2.0),
                2,
            ),
        )

        return {
            "generated_at": _utc_now_iso(),
            "history_count": history_count,
            "error_stream_count": error_stream_count,
            "issue_turn_count": issue_turn_count,
            "event_count": event_count,
            "error_turn_rate": error_turn_rate,
            "health_score": health_score,
            "issue_counts": dict(issue_counts),
            "family_counts": dict(family_counts),
            "severity_counts": dict(severity_counts),
            "strategy_summary": strategy_summary,
            "persona_hotspots": persona_hotspots,
            "intent_hotspots": intent_hotspots,
            "topic_hotspots": topic_hotspots,
            "top_error_queries": top_error_queries,
            "coverage_gap_queue": coverage_gap_queue,
            "dominant_issue": dominant_issue,
            "error_stream_alignment": error_stream_alignment,
            "orphan_error_count": orphan_error_count,
            "current_default_strategy": (runtime or {}).get("default_strategy"),
            "current_controls": user_controls or {},
            "recent_turns": recent_turns,
            "recent_events": recent_events,
        }

    def recommend_actions(
        self,
        summary: Dict[str, Any],
        user_controls: Optional[Dict[str, Any]] = None,
        runtime: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        user_controls = user_controls or {}
        runtime = runtime or {}
        issue_counts = summary.get("issue_counts", {}) or {}
        actions: List[Dict[str, Any]] = []

        def add_action(kind: str, key: str, new_value: Any, reason: str, priority: str, source_issues: Sequence[str], payload: Optional[Dict[str, Any]] = None) -> None:
            actions.append(
                {
                    "kind": kind,
                    "key": key,
                    "new_value": new_value,
                    "reason": reason,
                    "priority": priority,
                    "source_issues": list(source_issues),
                    "payload": payload or {},
                }
            )

        if issue_counts.get("pending_context_risk", 0) >= self.config.min_events_for_control_action and user_controls.get("allow_pending_records", False):
            add_action(
                kind="set_user_control",
                key="allow_pending_records",
                new_value=False,
                reason="Pending synthetic records repeatedly entered live context.",
                priority="high",
                source_issues=["pending_context_risk"],
            )

        grounding_issue_count = issue_counts.get("weak_anchor_grounding", 0) + issue_counts.get("persona_mismatch", 0)
        if grounding_issue_count >= self.config.min_events_for_control_action and runtime.get("default_strategy") != "anchor_priority":
            add_action(
                kind="set_runtime",
                key="default_strategy",
                new_value="anchor_priority",
                reason="Grounding issues suggest stronger anchor-biased retrieval.",
                priority="high",
                source_issues=["weak_anchor_grounding", "persona_mismatch"],
            )

        if issue_counts.get("weak_anchor_grounding", 0) >= self.config.min_events_for_control_action and user_controls.get("allow_synthetic_expansions", True):
            add_action(
                kind="set_user_control",
                key="allow_synthetic_expansions",
                new_value=False,
                reason="Synthetic expansions are dominating packets with weak anchor share.",
                priority="medium",
                source_issues=["weak_anchor_grounding"],
            )

        if issue_counts.get("context_overload", 0) >= self.config.min_events_for_control_action or issue_counts.get("topic_leakage", 0) >= self.config.min_events_for_control_action:
            current_global_k = _safe_int(user_controls.get("max_context_global_k"), self.config.default_context_global_k)
            current_session_k = _safe_int(user_controls.get("max_context_session_k"), self.config.default_context_session_k)
            proposed_global_k = max(4, current_global_k - 2)
            proposed_session_k = max(3, current_session_k - 1)

            if proposed_global_k != current_global_k:
                add_action(
                    kind="set_user_control",
                    key="max_context_global_k",
                    new_value=proposed_global_k,
                    reason="Context packets are too large or too diffuse; trim global retrieval.",
                    priority="medium",
                    source_issues=["context_overload", "topic_leakage"],
                )

            if proposed_session_k != current_session_k:
                add_action(
                    kind="set_user_control",
                    key="max_context_session_k",
                    new_value=proposed_session_k,
                    reason="Context packets are too large or too diffuse; trim session retrieval.",
                    priority="medium",
                    source_issues=["context_overload", "topic_leakage"],
                )

        coverage_gap_queue = summary.get("coverage_gap_queue", []) or []
        if coverage_gap_queue:
            add_action(
                kind="append_runtime_list",
                key="coverage_gap_queue",
                new_value=coverage_gap_queue,
                reason="Coverage-related failures should be queued for anchor backfill or grounded expansion.",
                priority="medium",
                source_issues=["missing_context", "missing_topic_support"],
                payload={"dedupe_key_fields": ["query", "predicted_labels"]},
            )

        if summary.get("error_turn_rate", 0.0) >= self.config.error_rate_alert_threshold:
            add_action(
                kind="set_runtime",
                key="require_grounded_followup_on_weak_packets",
                new_value=True,
                reason="Overall context error rate is high; force more cautious follow-up behavior.",
                priority="medium",
                source_issues=list(issue_counts.keys()),
            )

        # Deduplicate actions by (kind, key), preserving the strictest / smallest context windows.
        deduped: Dict[tuple, Dict[str, Any]] = {}
        for action in actions:
            action_key = (action["kind"], action["key"])
            existing = deduped.get(action_key)
            if existing is None:
                deduped[action_key] = action
                continue

            if action["kind"] == "set_user_control" and action["key"] in {"max_context_global_k", "max_context_session_k"}:
                if _safe_int(action["new_value"]) < _safe_int(existing["new_value"]):
                    deduped[action_key] = action
                continue

            if action["kind"] == "append_runtime_list":
                merged = list(existing.get("new_value", []))
                existing_keys = {_json_key(x) for x in merged}
                for item in _safe_list(action.get("new_value", [])):
                    item_key = _json_key(item)
                    if item_key not in existing_keys:
                        existing_keys.add(item_key)
                        merged.append(item)
                existing["new_value"] = merged
                deduped[action_key] = existing
                continue

        return list(deduped.values())

    def sync_from_runtime(
        self,
        runtime: Dict[str, Any],
        user_controls: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context_history = list(runtime.get("context_history", []) or [])
        context_errors = list(runtime.get("context_errors", []) or [])
        turn_df = self.build_turn_frame(context_history=context_history, context_errors=context_errors)
        event_df = self.build_event_frame(turn_df)
        summary = self.summarize(
            turn_df=turn_df,
            event_df=event_df,
            context_history=context_history,
            context_errors=context_errors,
            runtime=runtime,
            user_controls=user_controls,
        )
        recommendations = self.recommend_actions(summary, user_controls=user_controls, runtime=runtime)

        layer_state = {
            "generated_at": summary["generated_at"],
            "turn_log": turn_df.to_dict(orient="records"),
            "event_log": event_df.to_dict(orient="records"),
            "summary": summary,
            "recommendations": recommendations,
            "history_cursor": len(context_history),
            "error_cursor": len(context_errors),
        }
        runtime["context_error_layer"] = layer_state
        runtime["last_context_error_snapshot"] = summary
        return layer_state

    def apply_recommendations(
        self,
        runtime: Dict[str, Any],
        user_controls: Dict[str, Any],
        recommendations: Optional[Sequence[Dict[str, Any]]] = None,
        auto_apply: bool = True,
    ) -> Dict[str, Any]:
        layer_state = runtime.get("context_error_layer", {}) or {}
        recommendations = list(recommendations or layer_state.get("recommendations", []) or [])

        report = {
            "generated_at": _utc_now_iso(),
            "auto_apply": auto_apply,
            "applied_actions": [],
            "skipped_actions": [],
        }

        if not auto_apply:
            report["skipped_actions"] = list(recommendations)
            runtime["last_context_error_action_report"] = report
            return report

        for action in recommendations:
            kind = action.get("kind")
            key = action.get("key")

            if kind == "set_user_control":
                old_value = user_controls.get(key)
                new_value = action.get("new_value")
                if old_value == new_value:
                    continue
                user_controls[key] = new_value
                report["applied_actions"].append(
                    {
                        **action,
                        "old_value": old_value,
                        "new_value": new_value,
                    }
                )
                continue

            if kind == "set_runtime":
                old_value = runtime.get(key)
                new_value = action.get("new_value")
                if old_value == new_value:
                    continue
                runtime[key] = new_value
                report["applied_actions"].append(
                    {
                        **action,
                        "old_value": old_value,
                        "new_value": new_value,
                    }
                )
                continue

            if kind == "append_runtime_list":
                old_items = list(runtime.get(key, []) or [])
                old_keys = {_json_key(x) for x in old_items}
                appended = []
                for item in _safe_list(action.get("new_value", [])):
                    item_key = _json_key(item)
                    if item_key not in old_keys:
                        old_keys.add(item_key)
                        old_items.append(item)
                        appended.append(item)
                runtime[key] = old_items
                report["applied_actions"].append(
                    {
                        **action,
                        "old_value": None,
                        "new_value": appended,
                    }
                )
                continue

            report["skipped_actions"].append(action)

        runtime["last_context_error_action_report"] = report
        return report

    def turn_frame_from_layer(self, layer_state: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame(layer_state.get("turn_log", []) or [])

    def event_frame_from_layer(self, layer_state: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame(layer_state.get("event_log", []) or [])

    def export_artifacts(self, layer_state: Dict[str, Any], output_dir: str | Path) -> Dict[str, str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        turn_path = output_dir / "context_error_turn_log.csv"
        event_path = output_dir / "context_error_event_log.csv"
        summary_path = output_dir / "context_error_summary.json"
        rec_path = output_dir / "context_error_recommendations.json"

        self.turn_frame_from_layer(layer_state).to_csv(turn_path, index=False)
        self.event_frame_from_layer(layer_state).to_csv(event_path, index=False)
        summary_path.write_text(json.dumps(layer_state.get("summary", {}), indent=2, default=str), encoding="utf-8")
        rec_path.write_text(json.dumps(layer_state.get("recommendations", []), indent=2, default=str), encoding="utf-8")

        return {
            "turn_log_csv": str(turn_path),
            "event_log_csv": str(event_path),
            "summary_json": str(summary_path),
            "recommendations_json": str(rec_path),
        }


__all__ = [
    "ContextErrorEngineeringConfig",
    "ContextErrorEngineeringLayer",
    "ISSUE_FAMILY",
    "ISSUE_BASE_SEVERITY",
    "ISSUE_ACTIONS",
]
