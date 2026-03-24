
# %% [markdown]
"""
Advanced Balanced Hybrid Persona Pipeline
========================================

A notebook-friendly Python module for running the balanced hybrid system:

1. Ground with evidence-backed anchor personas
2. Expand with controlled persona variation
3. Validate with rubric-aware drift checks
4. Rebalance slices for evaluation quality
5. Emit context-ready packets for later integration into a contextual engineering pipeline

This module is designed to pair with:
- hybrid_persona_system_guide.md
- hybrid_persona_schema.json
- hybrid_persona_label_codebook.csv
"""

# %% 
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import Counter, defaultdict
import copy
import datetime as dt
import hashlib
import json
import math
import random
import re
import statistics

import numpy as np
import pandas as pd
import yaml
from jsonschema import Draft202012Validator
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


# %%
# ----------------------------
# Constants and configuration
# ----------------------------

PERSONA_LABELS = ["Enterprise IT Buyer", "Individual User"]
INTENT_LABELS = ["Low", "Medium", "High"]
TOPIC_LABELS = [
    "Identity & Access Management",
    "Endpoint Security & Threat Protection",
    "VPN & Zero Trust Access",
    "Backup, Storage & Continuity",
    "Procurement & Vendor Evaluation",
    "Productivity & Note-Taking",
    "CRM & Marketing Tools",
    "Web & Site Building",
    "Consumer Security & Privacy",
    "End-User Devices & Hardware",
    "Skills & Training",
    "Collaboration & Project Management",
]
OUTPUT_TYPES = ["search_query", "assistant_prompt", "transcript_turn"]
SOURCE_MODES = [
    "observed",
    "script_generated",
    "personahub_zero_shot",
    "personahub_few_shot",
    "hybrid",
]
DRIFT_FLAGS = [
    "persona_drift",
    "intent_drift",
    "topic_drift",
    "vocabulary_drift",
    "constraint_drift",
]

DEFAULT_PERSONA_RULES = {
    "Enterprise IT Buyer": [
        "team", "company", "business", "department", "employees", "employee", "admin",
        "rollout", "deployment", "implementation", "sso", "single sign-on", "soc2",
        "compliance", "procurement", "vendor", "sla", "security review", "approval",
        "manageability", "centralized control", "remote workforce", "organization",
        "staff", "workforce", "it manager", "security analyst", "endpoints", "policy",
        "for our", "for my team", "our team", "for employees", "company-wide",
    ],
    "Individual User": [
        "personal", "family", "class", "classes", "college", "student", "freelance",
        "freelancer", "home", "study", "budget", "my notes", "for me", "my laptop",
        "my family", "solo", "self-serve", "household", "hobby", "hobbyist",
        "my budget", "my classes", "my files", "my website", "my data", "privacy",
    ],
}

DEFAULT_INTENT_RULES = {
    "Low": [
        "what is", "how to", "guide", "tutorial", "benefits", "examples",
        "definition", "learn", "meaning", "basics", "beginner", "overview",
        "why use", "explain", "tips for",
    ],
    "Medium": [
        "best", "top", "vs", "versus", "comparison", "alternatives",
        "checklist", "features", "pros and cons", "compare", "which is better",
        "reviews", "recommend", "framework",
    ],
    "High": [
        "pricing", "demo", "trial", "implementation", "migration", "proof",
        "compliance", "deploy", "deployment", "sign up", "buy", "purchase",
        "quote", "rollout", "roll out", "roi", "security review", "requirements",
        "cost for", "pricing for",
    ],
}

DEFAULT_TOPIC_RULES = {
    "Identity & Access Management": [
        "identity", "access", "sso", "single sign-on", "mfa", "permissions",
        "login", "authentication", "iam", "identity provider", "okta", "entra id",
        "directory", "access control",
    ],
    "Endpoint Security & Threat Protection": [
        "endpoint", "antivirus", "edr", "xdr", "malware", "threat", "device protection",
        "ransomware", "phishing protection", "threat detection",
    ],
    "VPN & Zero Trust Access": [
        "vpn", "remote access", "zero trust", "ztna", "secure access", "remote workforce",
        "remote employees", "remote staff", "access gateway", "private access",
    ],
    "Backup, Storage & Continuity": [
        "backup", "backups", "storage", "recovery", "file retention", "retention",
        "disaster recovery", "continuity", "restore", "cloud storage", "document storage",
        "file backup",
    ],
    "Procurement & Vendor Evaluation": [
        "vendor", "vendors", "shortlist", "rfp", "rfq", "procurement",
        "pricing", "quote", "demo", "trial", "vendor evaluation", "vendor comparison",
        "roi", "purchase", "buying process",
    ],
    "Productivity & Note-Taking": [
        "notes", "note taking", "task app", "organizing", "knowledge base", "pkm",
        "second brain", "meeting notes", "class notes", "study notes", "journal",
        "personal knowledge management",
    ],
    "CRM & Marketing Tools": [
        "crm", "leads", "lead tracking", "sales pipeline", "customer tracking",
        "email marketing", "campaign", "marketing automation", "prospect",
    ],
    "Web & Site Building": [
        "website", "site builder", "cms", "hosting", "landing page", "wordpress",
        "portfolio site", "web design", "blog platform",
    ],
    "Consumer Security & Privacy": [
        "password manager", "privacy", "home security", "identity theft",
        "consumer antivirus", "personal security", "family safety", "secure my accounts",
        "protect my files", "privacy app",
    ],
    "End-User Devices & Hardware": [
        "laptop", "monitor", "hardware", "keyboard", "mouse", "tablet",
        "phone", "device", "pc", "computer", "headset",
    ],
    "Skills & Training": [
        "learn", "course", "tutorial", "training", "certification", "beginner",
        "how to use", "guide", "classes", "lesson", "practice", "skill",
    ],
    "Collaboration & Project Management": [
        "collaboration", "project management", "kanban", "task board", "project board",
        "team docs", "shared docs", "workflow", "asana", "trello", "monday",
        "team coordination", "project tracking",
    ],
}

TOPIC_CANONICAL_TERMS = {
    "Identity & Access Management": ["sso", "mfa", "identity platform", "access management"],
    "Endpoint Security & Threat Protection": ["endpoint protection", "edr", "device security", "threat protection"],
    "VPN & Zero Trust Access": ["vpn", "zero trust access", "remote access", "ztna"],
    "Backup, Storage & Continuity": ["backup", "cloud storage", "file recovery", "storage"],
    "Procurement & Vendor Evaluation": ["vendor evaluation", "tool shortlist", "pricing review", "procurement"],
    "Productivity & Note-Taking": ["note taking app", "pkm tool", "class notes app", "knowledge base"],
    "CRM & Marketing Tools": ["crm", "email marketing platform", "lead tracker", "marketing tool"],
    "Web & Site Building": ["website builder", "cms", "hosting platform", "site builder"],
    "Consumer Security & Privacy": ["password manager", "privacy tool", "personal security app", "home privacy setup"],
    "End-User Devices & Hardware": ["laptop", "device", "hardware", "college laptop"],
    "Skills & Training": ["training", "tutorial", "course", "beginner guide"],
    "Collaboration & Project Management": ["project management tool", "team collaboration app", "task board", "shared docs"],
}

STOPWORDS = set(ENGLISH_STOP_WORDS)
GENERIC_TOKENS = {
    "tool", "tools", "software", "solution", "solutions", "platform", "platforms",
    "system", "systems", "good", "better", "use", "using", "need", "help", "thing",
}
ANCHOR_IMPORTANCE = 5
OBSERVED_IMPORTANCE = 4
HYBRID_IMPORTANCE = 3
SESSION_IMPORTANCE = 1


# %%
# ----------------------------
# Utility helpers
# ----------------------------

def utc_today_iso() -> str:
    return dt.datetime.utcnow().date().isoformat()


def stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return text[:80] if text else "item"


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    text = normalize_text(text).lower()
    tokens = re.findall(r"[a-z0-9]+(?:['\-\+&/\.][a-z0-9]+)*", text)
    return [t for t in tokens if t and len(t) > 1 and t not in STOPWORDS]


def phrase_present(text: str, phrase: str) -> bool:
    text = f" {normalize_text(text).lower()} "
    phrase = normalize_text(phrase).lower()
    if not phrase:
        return False
    if " " in phrase:
        return f" {phrase} " in text or phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def flatten_list(values: Iterable[Iterable[Any]]) -> List[Any]:
    out: List[Any] = []
    for value in values:
        out.extend(list(value))
    return out


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item is None:
            continue
        item = normalize_text(item)
        if not item:
            continue
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def jaccard_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def estimate_tokens(text: str) -> int:
    return max(1, int(len(normalize_text(text).split()) * 1.3))


def safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if v is not None]
    return float(sum(vals) / len(vals)) if vals else default


def parse_semicolon_cues(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    return [normalize_text(part) for part in text.split(";") if normalize_text(part)]


# %%
# ----------------------------
# Data models
# ----------------------------

@dataclass
class PipelineConfig:
    seed: int = 7
    max_search_query_words: int = 18
    default_budget_per_anchor: int = 24
    target_intent_mix: Dict[str, float] = field(
        default_factory=lambda: {"Low": 0.30, "Medium": 0.45, "High": 0.25}
    )
    target_output_mix: Dict[str, float] = field(
        default_factory=lambda: {"search_query": 0.50, "assistant_prompt": 0.30, "transcript_turn": 0.20}
    )
    approve_threshold: float = 0.60
    reject_threshold: float = 0.42
    target_records_per_anchor: int = 12
    min_vocab_overlap: float = 0.10
    max_records_per_micro_persona: int = 6
    context_global_k: int = 8
    context_session_k: int = 6
    context_token_budget: int = 1200
    prune_days_low_importance: int = 365
    vocabulary_top_n: int = 20
    exemplar_per_intent: Dict[str, int] = field(
        default_factory=lambda: {"Low": 1, "Medium": 2, "High": 1}
    )
    mmr_lambda: float = 0.70


@dataclass
class AnchorPersona:
    anchor_persona_id: str
    macro_persona: str
    anchor_name: str
    job_to_be_done: str
    constraints: List[str]
    success_metric: str
    decision_criteria: List[str]
    vocabulary: List[str]
    evidence_sources: List[str]
    confidence_score: float
    priority_topics: List[str]
    allowed_intent_levels: List[str] = field(default_factory=lambda: INTENT_LABELS.copy())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def profile_text(self) -> str:
        parts = [
            self.anchor_name,
            self.macro_persona,
            self.job_to_be_done,
            " ".join(self.constraints),
            self.success_metric,
            " ".join(self.decision_criteria),
            " ".join(self.vocabulary),
            " ".join(self.priority_topics),
        ]
        return " | ".join([normalize_text(p) for p in parts if normalize_text(p)])


@dataclass
class GenerationSpec:
    anchor_persona_id: str
    persona_macro: str
    persona_micro: str
    topic_product_category: str
    intent_level: str
    output_type: str
    mode: str
    source_mode: str = "hybrid"
    exemplars: List[str] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateRecord:
    record_id: str
    source_mode: str
    anchor_persona_id: str
    persona_macro: str
    persona_micro: str
    intent_level: str
    topic_product_category: str
    job_to_be_done: str
    constraints: List[str]
    success_metric: str
    decision_criteria: List[str]
    vocabulary: List[str]
    text: str
    output_type: str
    label_evidence: List[str]
    label_confidence: str
    realism_score: float = 0.0
    drift_flags: List[str] = field(default_factory=list)
    review_status: str = "pending"
    importance: int = HYBRID_IMPORTANCE
    critic_notes: List[str] = field(default_factory=list)
    context_error_flags: List[str] = field(default_factory=list)
    generation_mode: str = "within_anchor"
    planned_persona_macro: Optional[str] = None
    planned_intent_level: Optional[str] = None
    planned_topic_product_category: Optional[str] = None
    created_at: str = field(default_factory=utc_today_iso)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["review_status"] = self.review_status
        return data


@dataclass
class HybridMemoryNote:
    text: str
    last_update_date: str
    keywords: List[str]
    importance: int
    note_type: str
    source_mode: str
    anchor_persona_id: Optional[str] = None
    record_id: Optional[str] = None
    topic_product_category: Optional[str] = None
    intent_level: Optional[str] = None
    persona_macro: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContextPacket:
    query: str
    predicted_labels: Dict[str, Any]
    frontmatter: str
    global_memories_md: str
    session_memories_md: str
    selected_global: List[HybridMemoryNote]
    selected_session: List[HybridMemoryNote]
    audit: Dict[str, Any]


@dataclass
class HybridPersonaState:
    profile: Dict[str, Any] = field(default_factory=dict)
    anchor_registry: Dict[str, AnchorPersona] = field(default_factory=dict)
    global_memory: Dict[str, Any] = field(default_factory=lambda: {"notes": [], "approved_records": []})
    session_memory: Dict[str, Any] = field(default_factory=lambda: {"notes": [], "candidate_records": [], "review_queue": []})
    eval_memory: Dict[str, Any] = field(default_factory=lambda: {"metrics": {}, "ab_tests": []})
    system_frontmatter: str = ""
    global_memories_md: str = ""
    session_memories_md: str = ""
    inject_session_memories_next_turn: bool = False


# %%
# ----------------------------
# Artifact loaders
# ----------------------------

def load_codebook_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_json_schema(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_markdown_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


# %%
# ----------------------------
# Rule compilation and labeling
# ----------------------------

def compile_rules_from_codebook(codebook_df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    persona_rules = copy.deepcopy(DEFAULT_PERSONA_RULES)
    intent_rules = copy.deepcopy(DEFAULT_INTENT_RULES)
    topic_rules = copy.deepcopy(DEFAULT_TOPIC_RULES)

    for _, row in codebook_df.iterrows():
        axis = normalize_text(row.get("Axis"))
        label = normalize_text(row.get("Label"))
        cues = parse_semicolon_cues(row.get("Typical_Cues", ""))
        if axis == "Persona" and label:
            persona_rules.setdefault(label, [])
            persona_rules[label] = dedupe_preserve_order(persona_rules[label] + cues)
        elif axis == "Intent_Level" and label:
            intent_rules.setdefault(label, [])
            intent_rules[label] = dedupe_preserve_order(intent_rules[label] + cues)
        elif axis == "Topic/Product_Category" and label:
            topic_rules.setdefault(label, [])
            topic_rules[label] = dedupe_preserve_order(topic_rules[label] + cues)

    # Ensure the guide's closed list is fully represented even if the CSV omits one row.
    for topic in TOPIC_LABELS:
        topic_rules.setdefault(topic, DEFAULT_TOPIC_RULES.get(topic, []))

    return {"persona": persona_rules, "intent": intent_rules, "topic": topic_rules}


class HybridLabeler:
    """
    Hybrid rule-based + lexical-similarity labeler.

    The labeler is intentionally transparent:
    - direct cue matches become evidence
    - anchor priors can lightly bias ambiguous cases
    - fallback lexical similarity resolves low-signal inputs
    """

    def __init__(self, codebook_df: pd.DataFrame):
        self.codebook_df = codebook_df.copy()
        compiled = compile_rules_from_codebook(codebook_df)
        self.persona_rules = compiled["persona"]
        self.intent_rules = compiled["intent"]
        self.topic_rules = compiled["topic"]

        self.topic_definitions = self._build_axis_definition_text("Topic/Product_Category", TOPIC_LABELS)
        self.persona_definitions = self._build_axis_definition_text("Persona", PERSONA_LABELS)
        self.intent_definitions = self._build_axis_definition_text("Intent_Level", INTENT_LABELS)

        self.topic_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.topic_matrix = self.topic_vectorizer.fit_transform([self.topic_definitions[label] for label in TOPIC_LABELS])

        self.persona_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.persona_matrix = self.persona_vectorizer.fit_transform([self.persona_definitions[label] for label in PERSONA_LABELS])

        self.intent_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.intent_matrix = self.intent_vectorizer.fit_transform([self.intent_definitions[label] for label in INTENT_LABELS])

    def _build_axis_definition_text(self, axis: str, labels: Sequence[str]) -> Dict[str, str]:
        subset = self.codebook_df[self.codebook_df["Axis"] == axis].copy()
        out: Dict[str, str] = {}
        for label in labels:
            rows = subset[subset["Label"] == label]
            if rows.empty:
                fallback_cues = DEFAULT_TOPIC_RULES.get(label, DEFAULT_PERSONA_RULES.get(label, DEFAULT_INTENT_RULES.get(label, [])))
                out[label] = f"{label}. {'; '.join(fallback_cues)}"
            else:
                row = rows.iloc[0]
                out[label] = " ".join(
                    [
                        label,
                        normalize_text(row.get("Definition", "")),
                        normalize_text(row.get("Typical_Cues", "")),
                        normalize_text(row.get("Tie_Breaker", "")),
                    ]
                )
        return out

    def _cue_score(
        self,
        text: str,
        rules: Dict[str, List[str]],
        labels: Sequence[str],
        prior_label: Optional[str] = None,
        prior_boost: float = 0.35,
    ) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        scores: Dict[str, float] = {}
        evidences: Dict[str, List[str]] = {}
        for label in labels:
            score = 0.0
            evidence: List[str] = []
            for cue in rules.get(label, []):
                if phrase_present(text, cue):
                    # Longer multi-word cues are stronger.
                    score += 1.4 if " " in cue else 1.0
                    evidence.append(cue)
            if prior_label and label == prior_label:
                score += prior_boost
            scores[label] = score
            evidences[label] = dedupe_preserve_order(evidence)
        return scores, evidences

    def _similarity_fallback(
        self,
        text: str,
        vectorizer: TfidfVectorizer,
        matrix,
        labels: Sequence[str],
    ) -> Tuple[str, float]:
        vec = vectorizer.transform([text])
        sims = cosine_similarity(vec, matrix)[0]
        idx = int(np.argmax(sims))
        return labels[idx], float(sims[idx])

    @staticmethod
    def _confidence_from_scores(scores: Dict[str, float], fallback_score: float = 0.0) -> str:
        if not scores:
            return "Low"
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top = ordered[0][1]
        gap = top - (ordered[1][1] if len(ordered) > 1 else 0.0)
        if top >= 2.0 and gap >= 1.0:
            return "High"
        if top > 0.0 or fallback_score >= 0.18:
            return "Medium"
        return "Low"

    def label_persona(self, text: str, anchor: Optional[AnchorPersona] = None) -> Dict[str, Any]:
        prior = anchor.macro_persona if anchor else None
        scores, evidences = self._cue_score(text, self.persona_rules, PERSONA_LABELS, prior_label=prior)
        label = max(scores, key=scores.get)
        fallback_label, fallback_score = self._similarity_fallback(text, self.persona_vectorizer, self.persona_matrix, PERSONA_LABELS)

        # Heuristic tie-breaker based on decision consequence.
        if scores["Enterprise IT Buyer"] == scores["Individual User"]:
            if re.search(r"\b(employees|team|company|department|admin|deployment|procurement|soc2|sso)\b", text.lower()):
                label = "Enterprise IT Buyer"
            elif re.search(r"\b(student|college|family|home|freelancer|personal|class)\b", text.lower()):
                label = "Individual User"
            else:
                label = fallback_label

        if scores[label] <= 0.0:
            label = fallback_label

        confidence = self._confidence_from_scores(scores, fallback_score=fallback_score)
        evidence = evidences.get(label, [])
        if not evidence:
            tokens = tokenize(text)
            evidence = tokens[:3]
        return {"label": label, "evidence": evidence, "confidence": confidence, "scores": scores, "fallback_score": fallback_score}

    def label_intent(self, text: str, anchor: Optional[AnchorPersona] = None) -> Dict[str, Any]:
        scores, evidences = self._cue_score(text, self.intent_rules, INTENT_LABELS)
        label = max(scores, key=scores.get)
        fallback_label, fallback_score = self._similarity_fallback(text, self.intent_vectorizer, self.intent_matrix, INTENT_LABELS)

        lower = text.lower()
        if re.search(r"\b(pricing|demo|trial|implementation|migration|deploy|deployment|quote|roi|compliance)\b", lower):
            label = "High"
        elif re.search(r"\b(vs|versus|best|top|compare|comparison|alternatives|features|reviews)\b", lower):
            label = "Medium"
        elif re.search(r"\b(what is|how to|guide|tutorial|learn|beginner|examples|meaning)\b", lower):
            label = "Low"
        elif scores[label] <= 0.0:
            label = fallback_label

        # Anchor intent prior if the chosen intent is disallowed.
        if anchor and label not in anchor.allowed_intent_levels:
            if anchor.allowed_intent_levels:
                label = anchor.allowed_intent_levels[0]

        confidence = self._confidence_from_scores(scores, fallback_score=fallback_score)
        evidence = evidences.get(label, [])
        if not evidence:
            evidence = tokenize(text)[:3]
        return {"label": label, "evidence": evidence, "confidence": confidence, "scores": scores, "fallback_score": fallback_score}

    def label_topic(self, text: str, anchor: Optional[AnchorPersona] = None) -> Dict[str, Any]:
        prior = anchor.priority_topics[0] if anchor and anchor.priority_topics else None
        scores, evidences = self._cue_score(text, self.topic_rules, TOPIC_LABELS, prior_label=prior, prior_boost=0.20)
        label = max(scores, key=scores.get)
        fallback_label, fallback_score = self._similarity_fallback(text, self.topic_vectorizer, self.topic_matrix, TOPIC_LABELS)

        lower = text.lower()
        # Procurement tie-breaker: only choose procurement when procurement language dominates and domain signals are weak.
        procurement_score = scores.get("Procurement & Vendor Evaluation", 0.0)
        non_procurement_scores = {k: v for k, v in scores.items() if k != "Procurement & Vendor Evaluation"}
        strongest_domain_label = max(non_procurement_scores, key=non_procurement_scores.get)
        strongest_domain_score = non_procurement_scores[strongest_domain_label]
        if procurement_score > 0 and strongest_domain_score >= procurement_score:
            label = strongest_domain_label

        # Skills tie-breaker: if an explicit learning query also names a domain, choose Skills only when the domain is not the main object.
        if scores.get("Skills & Training", 0.0) > 0 and strongest_domain_score > 0:
            if re.search(r"\b(best|vs|compare|pricing|demo|trial)\b", lower):
                label = strongest_domain_label
            elif re.search(r"\blearn|tutorial|course|training|beginner|guide\b", lower) and strongest_domain_label not in {"Procurement & Vendor Evaluation"}:
                # if the query is explicitly about learning a tool, keep Skills; otherwise preserve domain label
                if strongest_domain_label in {"Productivity & Note-Taking", "Web & Site Building", "End-User Devices & Hardware"} and re.search(r"\bhow to use|learn|tutorial|course|training|guide\b", lower):
                    label = "Skills & Training"

        if scores[label] <= 0.0:
            label = fallback_label

        confidence = self._confidence_from_scores(scores, fallback_score=fallback_score)
        evidence = evidences.get(label, [])
        if not evidence:
            evidence = tokenize(text)[:3]
        return {"label": label, "evidence": evidence, "confidence": confidence, "scores": scores, "fallback_score": fallback_score}

    def annotate(self, text: str, anchor: Optional[AnchorPersona] = None) -> Dict[str, Any]:
        text = normalize_text(text)
        persona = self.label_persona(text, anchor=anchor)
        intent = self.label_intent(text, anchor=anchor)
        topic = self.label_topic(text, anchor=anchor)

        confidence_order = {"Low": 1, "Medium": 2, "High": 3}
        overall_conf = min(
            [persona["confidence"], intent["confidence"], topic["confidence"]],
            key=lambda x: confidence_order.get(x, 1),
        )
        evidence = dedupe_preserve_order(persona["evidence"] + intent["evidence"] + topic["evidence"])
        return {
            "persona_macro": persona["label"],
            "intent_level": intent["label"],
            "topic_product_category": topic["label"],
            "label_evidence": evidence,
            "label_confidence": overall_conf,
            "axis_details": {"persona": persona, "intent": intent, "topic": topic},
        }


# %%
# ----------------------------
# Vocabulary and exemplar helpers
# ----------------------------

def extract_top_terms(texts: Sequence[str], top_n: int = 20) -> List[str]:
    texts = [normalize_text(t) for t in texts if normalize_text(t)]
    if len(texts) < 2:
        tokens = [t for text in texts for t in tokenize(text) if t not in GENERIC_TOKENS]
        return dedupe_preserve_order(tokens)[:top_n]
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
        matrix = vec.fit_transform(texts)
        weights = np.asarray(matrix.mean(axis=0)).ravel()
        feats = np.array(vec.get_feature_names_out())
        order = np.argsort(weights)[::-1]
        terms = []
        for idx in order:
            term = feats[idx]
            if term in GENERIC_TOKENS:
                continue
            if len(term) < 3:
                continue
            terms.append(term)
            if len(terms) >= top_n:
                break
        return dedupe_preserve_order(terms)
    except Exception:
        tokens = [t for text in texts for t in tokenize(text) if t not in GENERIC_TOKENS]
        return dedupe_preserve_order(tokens)[:top_n]


def cosine_similarity_to_corpus(text: str, corpus: Sequence[str]) -> float:
    corpus = [normalize_text(x) for x in corpus if normalize_text(x)]
    if not corpus:
        return 0.0
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        matrix = vec.fit_transform(corpus + [text])
        sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
        return float(np.max(sims)) if len(sims) else 0.0
    except Exception:
        return 0.0


def mmr_select(
    texts: Sequence[str],
    relevance_scores: Sequence[float],
    k: int,
    lambda_mult: float = 0.70,
) -> List[int]:
    """Simple Maximal Marginal Relevance selector over a small corpus."""
    texts = [normalize_text(t) for t in texts]
    if not texts:
        return []
    k = min(k, len(texts))
    if len(texts) == 1:
        return [0]

    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vec.fit_transform(texts)
    similarity = cosine_similarity(matrix)

    selected: List[int] = []
    candidates = list(range(len(texts)))

    first = int(np.argmax(np.array(relevance_scores)))
    selected.append(first)
    candidates.remove(first)

    while candidates and len(selected) < k:
        best_idx = None
        best_score = -1e9
        for idx in candidates:
            max_sim_to_selected = max(similarity[idx, j] for j in selected) if selected else 0.0
            score = lambda_mult * relevance_scores[idx] - (1 - lambda_mult) * max_sim_to_selected
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        candidates.remove(best_idx)
    return selected


# %%
# ----------------------------
# Anchor construction and assignment
# ----------------------------

def build_anchor_registry(
    anchor_cards: pd.DataFrame | List[Dict[str, Any]],
    observed_records: Optional[pd.DataFrame] = None,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, AnchorPersona]:
    config = config or PipelineConfig()
    if isinstance(anchor_cards, list):
        anchor_df = pd.DataFrame(anchor_cards)
    else:
        anchor_df = anchor_cards.copy()

    required = {
        "anchor_persona_id", "macro_persona", "anchor_name", "job_to_be_done", "constraints",
        "success_metric", "decision_criteria", "vocabulary", "evidence_sources",
        "confidence_score", "priority_topics",
    }
    missing = required - set(anchor_df.columns)
    if missing:
        raise ValueError(f"Missing anchor card columns: {sorted(missing)}")

    out: Dict[str, AnchorPersona] = {}
    for _, row in anchor_df.iterrows():
        constraints = row["constraints"] if isinstance(row["constraints"], list) else parse_semicolon_cues(str(row["constraints"]).replace("|", ";"))
        decision_criteria = row["decision_criteria"] if isinstance(row["decision_criteria"], list) else parse_semicolon_cues(str(row["decision_criteria"]).replace("|", ";"))
        vocabulary = row["vocabulary"] if isinstance(row["vocabulary"], list) else parse_semicolon_cues(str(row["vocabulary"]).replace("|", ";"))
        evidence_sources = row["evidence_sources"] if isinstance(row["evidence_sources"], list) else parse_semicolon_cues(str(row["evidence_sources"]).replace("|", ";"))
        priority_topics = row["priority_topics"] if isinstance(row["priority_topics"], list) else parse_semicolon_cues(str(row["priority_topics"]).replace("|", ";"))
        allowed_intents = row.get("allowed_intent_levels", INTENT_LABELS)
        if not isinstance(allowed_intents, list):
            allowed_intents = parse_semicolon_cues(str(allowed_intents).replace("|", ";")) or INTENT_LABELS.copy()

        anchor = AnchorPersona(
            anchor_persona_id=normalize_text(row["anchor_persona_id"]),
            macro_persona=normalize_text(row["macro_persona"]),
            anchor_name=normalize_text(row["anchor_name"]),
            job_to_be_done=normalize_text(row["job_to_be_done"]),
            constraints=dedupe_preserve_order(constraints),
            success_metric=normalize_text(row["success_metric"]),
            decision_criteria=dedupe_preserve_order(decision_criteria),
            vocabulary=dedupe_preserve_order(vocabulary),
            evidence_sources=dedupe_preserve_order(evidence_sources),
            confidence_score=float(row["confidence_score"]),
            priority_topics=dedupe_preserve_order(priority_topics),
            allowed_intent_levels=dedupe_preserve_order(allowed_intents) or INTENT_LABELS.copy(),
            metadata={"source": row.get("source", "script_card")},
        )

        # Expand anchor vocabulary using aligned observed rows if available.
        if observed_records is not None and not observed_records.empty:
            subset = observed_records[observed_records.get("anchor_persona_id", "") == anchor.anchor_persona_id]
            if not subset.empty and "text" in subset.columns:
                derived = extract_top_terms(subset["text"].astype(str).tolist(), top_n=config.vocabulary_top_n)
                anchor.vocabulary = dedupe_preserve_order(anchor.vocabulary + derived)[: config.vocabulary_top_n]

                top_topics = subset.get("topic_product_category")
                if top_topics is not None and hasattr(top_topics, "mode"):
                    mode_topics = top_topics.dropna().astype(str).tolist()
                    anchor.priority_topics = dedupe_preserve_order(anchor.priority_topics + mode_topics)[:4]

        out[anchor.anchor_persona_id] = anchor
    return out


def assign_best_anchor(
    text: str,
    label_annotation: Dict[str, Any],
    anchor_registry: Dict[str, AnchorPersona],
) -> Tuple[Optional[str], float]:
    if not anchor_registry:
        return None, 0.0

    query = normalize_text(text)
    persona_label = label_annotation.get("persona_macro")
    topic_label = label_annotation.get("topic_product_category")

    anchor_ids = list(anchor_registry.keys())
    anchor_texts = [anchor_registry[a].profile_text() for a in anchor_ids]
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        matrix = vec.fit_transform(anchor_texts + [query])
        sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
    except Exception:
        sims = np.zeros(len(anchor_ids))

    best_anchor = None
    best_score = -1.0
    for idx, anchor_id in enumerate(anchor_ids):
        anchor = anchor_registry[anchor_id]
        score = float(sims[idx])
        if persona_label == anchor.macro_persona:
            score += 0.40
        if topic_label in anchor.priority_topics:
            score += 0.35
        vocab_overlap = jaccard_overlap(tokenize(query), tokenize(" ".join(anchor.vocabulary)))
        score += 0.35 * vocab_overlap
        if score > best_score:
            best_score = score
            best_anchor = anchor_id

    return best_anchor, float(best_score)


def build_anchor_summary_note(anchor: AnchorPersona) -> HybridMemoryNote:
    text = (
        f"Anchor {anchor.anchor_persona_id} | {anchor.macro_persona} | "
        f"JTBD: {anchor.job_to_be_done} | Constraints: {', '.join(anchor.constraints)} | "
        f"Decision criteria: {', '.join(anchor.decision_criteria)} | "
        f"Priority topics: {', '.join(anchor.priority_topics)}"
    )
    keywords = dedupe_preserve_order(anchor.vocabulary + anchor.priority_topics + [anchor.macro_persona])
    return HybridMemoryNote(
        text=text,
        last_update_date=utc_today_iso(),
        keywords=keywords[:12],
        importance=ANCHOR_IMPORTANCE,
        note_type="anchor",
        source_mode="script_generated",
        anchor_persona_id=anchor.anchor_persona_id,
        persona_macro=anchor.macro_persona,
        topic_product_category=anchor.priority_topics[0] if anchor.priority_topics else None,
    )


# %%
# ----------------------------
# Few-shot selection
# ----------------------------

def select_few_shot_exemplars(
    records_df: pd.DataFrame,
    anchor: AnchorPersona,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, List[str]]:
    config = config or PipelineConfig()
    if records_df.empty:
        return {intent: [] for intent in INTENT_LABELS}

    df = records_df.copy()
    if "review_status" in df.columns:
        df = df[df["review_status"].isin(["approved", "pending"])]
    if "anchor_persona_id" in df.columns:
        df = df[df["anchor_persona_id"] == anchor.anchor_persona_id]
    if df.empty:
        return {intent: [] for intent in INTENT_LABELS}

    out: Dict[str, List[str]] = {}
    for intent, k in config.exemplar_per_intent.items():
        subset = df[df["intent_level"] == intent].copy()
        if subset.empty:
            out[intent] = []
            continue

        texts = subset["text"].astype(str).tolist()
        vocab_text = " ".join(anchor.vocabulary + anchor.decision_criteria + anchor.constraints)
        relevances = []
        for text in texts:
            overlap = jaccard_overlap(tokenize(text), tokenize(vocab_text))
            conf = subset.loc[subset["text"] == text, "label_confidence"]
            conf_score = 1.0
            if not conf.empty:
                label_conf = conf.iloc[0]
                conf_score = {"Low": 0.4, "Medium": 0.7, "High": 1.0}.get(label_conf, 0.6)
            relevances.append(0.6 * overlap + 0.4 * conf_score)

        picked_idx = mmr_select(texts, relevances, k=k, lambda_mult=config.mmr_lambda)
        out[intent] = [texts[i] for i in picked_idx]
    return out


# %%
# ----------------------------
# Persona expansion
# ----------------------------

ENTERPRISE_ROLE_LIBRARY = {
    "default": [
        "IT manager at a mid-sized company",
        "security analyst reviewing the stack",
        "systems administrator handling rollout",
        "helpdesk lead onboarding employees",
        "procurement analyst comparing vendors",
        "operations lead standardizing tooling",
    ],
    "security": [
        "security analyst reviewing the stack",
        "IT manager at a mid-sized company",
        "systems administrator handling rollout",
        "procurement analyst comparing vendors",
        "compliance lead preparing for review",
    ],
    "productivity": [
        "operations lead standardizing tooling",
        "IT admin supporting knowledge workflows",
        "department manager improving team organization",
        "procurement analyst comparing vendors",
    ],
}

INDIVIDUAL_ROLE_LIBRARY = {
    "default": [
        "graduate student juggling classes and research",
        "freelancer managing solo work",
        "parent organizing family information",
        "privacy-conscious home user",
        "hobbyist learning a new tool",
    ],
    "security": [
        "privacy-conscious home user",
        "parent protecting family accounts",
        "freelancer securing client files",
        "non-technical home user trying to stay safe",
    ],
    "productivity": [
        "graduate student juggling classes and research",
        "freelancer managing solo work",
        "part-time student keeping notes organized",
        "creator tracking ideas across devices",
    ],
}


def role_library_key(anchor: AnchorPersona) -> str:
    topics = " ".join(anchor.priority_topics).lower()
    if any(k in topics for k in ["vpn", "zero trust", "endpoint", "identity", "security", "privacy"]):
        return "security"
    if any(k in topics for k in ["note", "productivity", "collaboration", "project"]):
        return "productivity"
    return "default"


def propose_micro_personas(anchor: AnchorPersona, mode: str = "within_anchor", n: int = 4) -> List[str]:
    key = role_library_key(anchor)
    if anchor.macro_persona == "Enterprise IT Buyer":
        library = ENTERPRISE_ROLE_LIBRARY.get(key, ENTERPRISE_ROLE_LIBRARY["default"])
    else:
        library = INDIVIDUAL_ROLE_LIBRARY.get(key, INDIVIDUAL_ROLE_LIBRARY["default"])

    if mode == "stakeholder_neighbor":
        if anchor.macro_persona == "Enterprise IT Buyer":
            extras = [
                "economic buyer approving budget",
                "technical evaluator testing fit",
                "administrator responsible for rollout",
                "security reviewer checking controls",
                "end user affected by the decision",
            ]
        else:
            extras = [
                "student comparing options before a semester starts",
                "parent choosing something easy for the household",
                "freelancer balancing value and convenience",
                "privacy-conscious home user avoiding complexity",
                "hobbyist exploring a better workflow",
            ]
        library = dedupe_preserve_order(library + extras)

    seeded = dedupe_preserve_order(library)
    return seeded[:n]


def allocate_generation_budget(
    existing_records: pd.DataFrame,
    anchor: AnchorPersona,
    total_budget: int,
    config: Optional[PipelineConfig] = None,
) -> List[Tuple[str, str]]:
    """
    Returns a list of (intent, output_type) slots to generate.
    Budget is allocated to underrepresented slices for this anchor.
    """
    config = config or PipelineConfig()
    if existing_records.empty:
        existing_records = pd.DataFrame(columns=["anchor_persona_id", "intent_level", "output_type"])

    anchor_df = existing_records.copy()
    if "anchor_persona_id" in anchor_df.columns:
        anchor_df = anchor_df[anchor_df["anchor_persona_id"] == anchor.anchor_persona_id]

    counts = Counter(zip(anchor_df.get("intent_level", pd.Series(dtype=str)), anchor_df.get("output_type", pd.Series(dtype=str))))
    plan: List[Tuple[str, str]] = []

    target_intents = config.target_intent_mix
    target_outputs = config.target_output_mix

    intent_quota = {intent: max(1, round(total_budget * target_intents[intent])) for intent in INTENT_LABELS}
    while sum(intent_quota.values()) > total_budget:
        # subtract from the currently largest quota
        largest = max(intent_quota, key=intent_quota.get)
        intent_quota[largest] -= 1
    while sum(intent_quota.values()) < total_budget:
        smallest = min(intent_quota, key=intent_quota.get)
        intent_quota[smallest] += 1

    for intent, iq in intent_quota.items():
        output_quota = {ot: max(1, round(iq * target_outputs[ot])) for ot in OUTPUT_TYPES}
        while sum(output_quota.values()) > iq:
            largest = max(output_quota, key=output_quota.get)
            output_quota[largest] -= 1
        while sum(output_quota.values()) < iq:
            smallest = min(output_quota, key=output_quota.get)
            output_quota[smallest] += 1

        for ot, oq in output_quota.items():
            existing = counts.get((intent, ot), 0)
            deficit = max(0, oq - min(existing, oq))
            for _ in range(max(1, deficit)):
                plan.append((intent, ot))

    if len(plan) > total_budget:
        plan = plan[:total_budget]
    while len(plan) < total_budget:
        plan.append(("Medium", "search_query"))
    return plan


def build_generation_specs(
    anchor_registry: Dict[str, AnchorPersona],
    records_df: pd.DataFrame,
    config: Optional[PipelineConfig] = None,
) -> List[GenerationSpec]:
    config = config or PipelineConfig()
    specs: List[GenerationSpec] = []

    for anchor in anchor_registry.values():
        exemplars = select_few_shot_exemplars(records_df, anchor, config=config)
        budget_slots = allocate_generation_budget(records_df, anchor, total_budget=config.default_budget_per_anchor, config=config)
        topics = anchor.priority_topics or [TOPIC_LABELS[0]]
        within_roles = propose_micro_personas(anchor, mode="within_anchor", n=4)
        neighbor_roles = propose_micro_personas(anchor, mode="stakeholder_neighbor", n=4)

        role_cycle = within_roles + neighbor_roles
        if not role_cycle:
            role_cycle = [anchor.anchor_name]

        for i, (intent, output_type) in enumerate(budget_slots):
            mode = "within_anchor" if i % 2 == 0 else "stakeholder_neighbor"
            roles = within_roles if mode == "within_anchor" else neighbor_roles
            persona_micro = roles[i % len(roles)]
            topic = topics[i % len(topics)]
            spec = GenerationSpec(
                anchor_persona_id=anchor.anchor_persona_id,
                persona_macro=anchor.macro_persona,
                persona_micro=persona_micro,
                topic_product_category=topic,
                intent_level=intent,
                output_type=output_type,
                mode=mode,
                source_mode="hybrid",
                exemplars=exemplars.get(intent, []),
                notes={"anchor_name": anchor.anchor_name},
            )
            specs.append(spec)

    return specs


# %%
# ----------------------------
# Generation adapters
# ----------------------------

def choose_topic_term(topic: str, anchor: AnchorPersona, rng: random.Random) -> str:
    candidates = TOPIC_CANONICAL_TERMS.get(topic, [])
    anchor_candidates = [v for v in anchor.vocabulary if phrase_present(" ".join(candidates + anchor.vocabulary), v)]
    pool = dedupe_preserve_order(anchor_candidates + candidates + anchor.vocabulary)
    return rng.choice(pool) if pool else topic.lower()


def concise_constraint(anchor: AnchorPersona, rng: random.Random) -> str:
    if not anchor.constraints:
        return ""
    return rng.choice(anchor.constraints)


def concise_criterion(anchor: AnchorPersona, rng: random.Random) -> str:
    if not anchor.decision_criteria:
        return ""
    return rng.choice(anchor.decision_criteria)


def scale_phrase(anchor: AnchorPersona, topic: str, rng: random.Random) -> str:
    if anchor.macro_persona == "Enterprise IT Buyer":
        topic_lower = topic.lower()
        if "endpoint" in topic_lower:
            return rng.choice(["100 endpoints", "250 endpoints", "300 devices"])
        if "vpn" in topic_lower or "zero trust" in topic_lower:
            return rng.choice(["150 remote employees", "300 staff", "a distributed team"])
        if "identity" in topic_lower:
            return rng.choice(["200 employees", "a growing team", "multiple departments"])
        return rng.choice(["a 300-person company", "a growing team", "multiple teams"])
    return rng.choice(["one student budget", "personal use", "my home setup", "solo use"])


def exemplar_lexical_style(exemplars: Sequence[str]) -> Dict[str, Any]:
    if not exemplars:
        return {"avg_words": 12, "leading_terms": []}
    lengths = [len(normalize_text(x).split()) for x in exemplars]
    first_terms = [tokenize(x)[:2] for x in exemplars]
    return {
        "avg_words": int(round(safe_mean(lengths, default=12))),
        "leading_terms": dedupe_preserve_order(flatten_list(first_terms))[:6],
    }


class BaseGeneratorAdapter:
    def generate(self, spec: GenerationSpec, anchor: AnchorPersona, n: int = 1) -> List[str]:
        raise NotImplementedError


class HeuristicPersonaHubAdapter(BaseGeneratorAdapter):
    """
    Deterministic, notebook-safe generator that emulates controlled expansion.

    It is not a replacement for PersonaHub or an LLM. It exists so the notebook
    runs end-to-end without external API dependencies.
    """

    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)

    def _search_query(self, spec: GenerationSpec, anchor: AnchorPersona) -> str:
        rng = self.rng
        term = choose_topic_term(spec.topic_product_category, anchor, rng)
        criterion = concise_criterion(anchor, rng)
        constraint = concise_constraint(anchor, rng)
        scale = scale_phrase(anchor, spec.topic_product_category, rng)

        if spec.intent_level == "Low":
            templates = [
                f"what is {term} for {spec.persona_micro}",
                f"how to {anchor.job_to_be_done.lower()} with {term}",
                f"{term} guide for {scale}",
            ]
        elif spec.intent_level == "Medium":
            templates = [
                f"best {term} for {scale} with {criterion}".strip(),
                f"{term} alternatives for {scale}",
                f"{term} comparison for {anchor.job_to_be_done.lower()}",
            ]
        else:
            if anchor.macro_persona == "Enterprise IT Buyer":
                templates = [
                    f"{term} pricing for {scale}",
                    f"how to deploy {term} with {criterion}".strip(),
                    f"proof {term} meets {constraint}".strip(),
                ]
            else:
                templates = [
                    f"best price {term} for {scale}",
                    f"how to set up {term} with {constraint}".strip(),
                    f"{term} trial or paid for {scale}",
                ]
        query = rng.choice(templates)
        query = re.sub(r"\s+", " ", query).strip()
        # Keep search queries compact.
        words = query.split()
        return " ".join(words[:18])

    def _assistant_prompt(self, spec: GenerationSpec, anchor: AnchorPersona) -> str:
        rng = self.rng
        criterion = concise_criterion(anchor, rng)
        constraint = concise_constraint(anchor, rng)
        term = choose_topic_term(spec.topic_product_category, anchor, rng)
        style = exemplar_lexical_style(spec.exemplars)

        if spec.intent_level == "Low":
            prompt = (
                f"I'm {spec.persona_micro}. I'm trying to understand {term} because I need to "
                f"{anchor.job_to_be_done.lower()}. Explain the basics, common tradeoffs, and what matters most given "
                f"{constraint or 'my limited context'}."
            )
        elif spec.intent_level == "Medium":
            prompt = (
                f"I'm {spec.persona_micro}. I need to shortlist options to {anchor.job_to_be_done.lower()}. "
                f"Compare realistic approaches in {spec.topic_product_category} and focus on {criterion or 'fit, usability, and cost'}. "
                f"Please keep the comparison practical rather than generic."
            )
        else:
            actor = "we" if anchor.macro_persona == "Enterprise IT Buyer" else "I"
            prompt = (
                f"I'm {spec.persona_micro}. {actor.capitalize()} are close to acting on a solution to "
                f"{anchor.job_to_be_done.lower()}. Walk through the implementation or purchase considerations, "
                f"including {constraint or 'constraints'}, success criteria, and how to validate the choice."
            )

        max_words = max(24, style["avg_words"] * 2)
        return " ".join(prompt.split()[:max_words])

    def _transcript_turn(self, spec: GenerationSpec, anchor: AnchorPersona) -> str:
        rng = self.rng
        criterion = concise_criterion(anchor, rng)
        constraint = concise_constraint(anchor, rng)
        term = choose_topic_term(spec.topic_product_category, anchor, rng)

        if spec.intent_level == "Low":
            options = [
                f"I keep seeing {term} come up. What does it actually do for someone trying to {anchor.job_to_be_done.lower()}?",
                f"I'm early in this and want the basics. Where should I start if I need to {anchor.job_to_be_done.lower()}?",
            ]
        elif spec.intent_level == "Medium":
            options = [
                f"I'm comparing options now. What should I look at besides {criterion or 'price'} if the goal is to {anchor.job_to_be_done.lower()}?",
                f"I've narrowed things down a bit. How would you compare the main options for {anchor.job_to_be_done.lower()}?",
            ]
        else:
            options = [
                f"We need to move soon. How do I validate that the option fits {constraint or 'our constraints'} before rollout?",
                f"I'm close to deciding. What would the actual setup or deployment look like if the priority is {criterion or 'fit'}?",
            ] if anchor.macro_persona == "Enterprise IT Buyer" else [
                f"I'm ready to act. What's the easiest way to set this up without running into problems around {constraint or 'my constraints'}?",
                f"I'm close to deciding. What should I double-check before I commit to a tool for {anchor.job_to_be_done.lower()}?",
            ]
        return rng.choice(options)

    def generate(self, spec: GenerationSpec, anchor: AnchorPersona, n: int = 1) -> List[str]:
        outputs = []
        for _ in range(n):
            if spec.output_type == "search_query":
                outputs.append(self._search_query(spec, anchor))
            elif spec.output_type == "assistant_prompt":
                outputs.append(self._assistant_prompt(spec, anchor))
            else:
                outputs.append(self._transcript_turn(spec, anchor))
        return outputs


def build_llm_generation_prompt(spec: GenerationSpec, anchor: AnchorPersona) -> str:
    """
    Optional prompt builder for swapping in a real JSON-capable LLM later.
    """
    exemplars = "\n".join([f"- {x}" for x in spec.exemplars]) or "- (none)"
    return f"""
You are generating ONE synthetic user utterance for an evaluation dataset.

ANCHOR PERSONA
- anchor_persona_id: {anchor.anchor_persona_id}
- macro_persona: {anchor.macro_persona}
- anchor_name: {anchor.anchor_name}
- JTBD: {anchor.job_to_be_done}
- constraints: {', '.join(anchor.constraints)}
- success_metric: {anchor.success_metric}
- decision_criteria: {', '.join(anchor.decision_criteria)}
- vocabulary: {', '.join(anchor.vocabulary)}
- priority_topics: {', '.join(anchor.priority_topics)}

GENERATION TARGET
- persona_micro: {spec.persona_micro}
- topic_product_category: {spec.topic_product_category}
- intent_level: {spec.intent_level}
- output_type: {spec.output_type}
- generation_mode: {spec.mode}

FEW-SHOT EXEMPLARS
{exemplars}

RULES
1. Stay inside the anchor's macro persona, JTBD, constraints, vocabulary, and topic.
2. Do not introduce a different market or different decision context.
3. Preserve the target intent band:
   - Low = learn
   - Medium = compare
   - High = act / validate / buy / deploy
4. Produce ONLY the user text, not labels or explanations.
5. Keep search queries compact. Make assistant prompts and transcript turns natural.
""".strip()


# %%
# ----------------------------
# Validation and realism scoring
# ----------------------------

def output_type_compliance_score(text: str, output_type: str, config: PipelineConfig) -> float:
    words = normalize_text(text).split()
    n_words = len(words)
    if output_type == "search_query":
        if 3 <= n_words <= config.max_search_query_words and not text.endswith("?"):
            return 1.0
        if 2 <= n_words <= config.max_search_query_words + 4:
            return 0.7
        return 0.3
    if output_type == "assistant_prompt":
        first_person = bool(re.search(r"\b(i|i'm|we|we're|our)\b", text.lower()))
        return 1.0 if n_words >= 14 and first_person else 0.7 if n_words >= 10 else 0.4
    if output_type == "transcript_turn":
        conversational = bool(re.search(r"[?]|i\b|we\b|i'm|we're", text.lower()))
        return 1.0 if n_words >= 8 and conversational else 0.6
    return 0.5


def specificity_score(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    unique_ratio = len(set(tokens)) / len(tokens)
    generic_penalty = sum(t in GENERIC_TOKENS for t in tokens) / len(tokens)
    length_bonus = min(1.0, len(tokens) / 12)
    return max(0.0, min(1.0, 0.55 * unique_ratio + 0.45 * length_bonus - 0.35 * generic_penalty))


def constraint_alignment_score(text: str, anchor: AnchorPersona) -> float:
    if not anchor.constraints and not anchor.decision_criteria:
        return 0.5
    lower = text.lower()
    matches = 0
    total = 0
    for phrase in anchor.constraints + anchor.decision_criteria:
        total += 1
        if phrase_present(lower, phrase):
            matches += 1
        else:
            # token overlap fallback
            if jaccard_overlap(tokenize(lower), tokenize(phrase)) > 0.15:
                matches += 1
    return matches / max(1, total)


def detect_constraint_drift(text: str, anchor: AnchorPersona, intent_level: str) -> bool:
    lower = text.lower()

    contradiction_pairs = [
        (["budget", "affordability", "budget ceiling", "student budget"], ["premium", "white-glove", "enterprise platinum"]),
        (["limited admin time", "easy setup", "simple setup"], ["fully custom", "self-hosted", "complex migration"]),
        (["compliance", "security review", "audit readiness"], ["consumer-grade", "no controls", "just trust it"]),
    ]
    for lhs, rhs in contradiction_pairs:
        if any(phrase_present(" ".join(anchor.constraints), x) for x in lhs) and any(phrase_present(lower, y) for y in rhs):
            return True

    # High-intent items should reflect at least some criteria/constraints.
    if intent_level == "High":
        return constraint_alignment_score(text, anchor) < 0.10
    return False


def label_agreement_score(predicted: Dict[str, Any], planned_persona: str, planned_intent: str, planned_topic: str) -> float:
    matches = [
        1.0 if predicted["persona_macro"] == planned_persona else 0.0,
        1.0 if predicted["intent_level"] == planned_intent else 0.0,
        1.0 if predicted["topic_product_category"] == planned_topic else 0.0,
    ]
    return float(sum(matches) / len(matches))


def record_to_memory_note(record: CandidateRecord) -> HybridMemoryNote:
    text = (
        f"{record.source_mode} | {record.persona_macro} | {record.intent_level} | "
        f"{record.topic_product_category} | {record.text}"
    )
    keywords = dedupe_preserve_order(record.vocabulary + record.label_evidence + [record.intent_level, record.topic_product_category])
    note_type = "approved_record" if record.review_status == "approved" else "candidate_record"
    return HybridMemoryNote(
        text=text,
        last_update_date=record.created_at,
        keywords=keywords[:12],
        importance=record.importance if record.review_status == "approved" else SESSION_IMPORTANCE,
        note_type=note_type,
        source_mode=record.source_mode,
        anchor_persona_id=record.anchor_persona_id,
        record_id=record.record_id,
        topic_product_category=record.topic_product_category,
        intent_level=record.intent_level,
        persona_macro=record.persona_macro,
        metadata={"review_status": record.review_status, "realism_score": record.realism_score},
    )


class CandidateValidator:
    def __init__(
        self,
        labeler: HybridLabeler,
        schema: Dict[str, Any],
        config: Optional[PipelineConfig] = None,
    ):
        self.labeler = labeler
        self.validator = Draft202012Validator(schema)
        self.config = config or PipelineConfig()

    def validate_candidate(
        self,
        candidate: CandidateRecord,
        anchor: AnchorPersona,
        corpus_texts: Optional[Sequence[str]] = None,
    ) -> CandidateRecord:
        record = copy.deepcopy(candidate)
        predicted = self.labeler.annotate(record.text, anchor=anchor)

        drift_flags: List[str] = []
        critic_notes: List[str] = []

        if predicted["persona_macro"] != record.persona_macro:
            drift_flags.append("persona_drift")
            critic_notes.append(f"Predicted persona={predicted['persona_macro']} but planned {record.persona_macro}.")
        if predicted["intent_level"] != record.intent_level:
            drift_flags.append("intent_drift")
            critic_notes.append(f"Predicted intent={predicted['intent_level']} but planned {record.intent_level}.")
        if predicted["topic_product_category"] != record.topic_product_category:
            drift_flags.append("topic_drift")
            critic_notes.append(f"Predicted topic={predicted['topic_product_category']} but planned {record.topic_product_category}.")

        vocab_overlap = jaccard_overlap(tokenize(record.text), tokenize(" ".join(anchor.vocabulary)))
        if vocab_overlap < self.config.min_vocab_overlap:
            drift_flags.append("vocabulary_drift")
            critic_notes.append("Low overlap with anchor vocabulary.")

        if detect_constraint_drift(record.text, anchor, record.intent_level):
            drift_flags.append("constraint_drift")
            critic_notes.append("Constraint alignment is weak or contradictory for this anchor.")

        schema_errors = sorted(self.validator.iter_errors(record.to_dict()), key=lambda e: e.path)
        if schema_errors:
            critic_notes.extend([f"Schema error: {e.message}" for e in schema_errors])

        agreement = label_agreement_score(predicted, record.persona_macro, record.intent_level, record.topic_product_category)
        similarity = cosine_similarity_to_corpus(record.text, corpus_texts or [])
        output_ok = output_type_compliance_score(record.text, record.output_type, self.config)
        specificity = specificity_score(record.text)
        constraint_ok = constraint_alignment_score(record.text, anchor)

        realism = (
            0.30 * agreement
            + 0.18 * vocab_overlap
            + 0.16 * similarity
            + 0.14 * output_ok
            + 0.12 * specificity
            + 0.10 * constraint_ok
        )
        realism = float(max(0.0, min(1.0, realism)))

        record.realism_score = realism
        record.label_evidence = dedupe_preserve_order(record.label_evidence + predicted["label_evidence"])
        # take the more conservative confidence
        conf_order = {"Low": 1, "Medium": 2, "High": 3}
        record.label_confidence = min(
            [record.label_confidence, predicted["label_confidence"]],
            key=lambda x: conf_order.get(x, 1),
        )
        record.drift_flags = dedupe_preserve_order(drift_flags)
        record.critic_notes = dedupe_preserve_order(record.critic_notes + critic_notes)

        severe = {"persona_drift", "intent_drift", "topic_drift"} & set(record.drift_flags)
        if schema_errors or realism < self.config.reject_threshold or severe:
            record.review_status = "rejected"
        elif realism >= self.config.approve_threshold and set(record.drift_flags).issubset({"vocabulary_drift"}):
            record.review_status = "approved"
        elif realism >= self.config.approve_threshold and not record.drift_flags:
            record.review_status = "approved"
        else:
            record.review_status = "pending"

        return record


# %%
# ----------------------------
# Balancing and consolidation
# ----------------------------

def rebalance_records(
    records: Sequence[CandidateRecord],
    config: Optional[PipelineConfig] = None,
) -> List[CandidateRecord]:
    config = config or PipelineConfig()
    approved = [r for r in records if r.review_status == "approved"]
    pending = [r for r in records if r.review_status == "pending"]
    pool = approved + pending
    if not pool:
        return []

    # Start from best-scoring records, but enforce intent/output/persona diversity.
    sorted_pool = sorted(pool, key=lambda r: (r.realism_score, -len(r.drift_flags)), reverse=True)

    selected: List[CandidateRecord] = []
    intent_counts = Counter()
    output_counts = Counter()
    persona_counts = Counter()
    micro_counts = Counter()

    num_anchors = max(1, len({r.anchor_persona_id for r in pool}))
    total_target = min(len(pool), max(len(approved), num_anchors * config.target_records_per_anchor))
    target_intents = {k: math.ceil(total_target * v) for k, v in config.target_intent_mix.items()}
    target_outputs = {k: math.ceil(total_target * v) for k, v in config.target_output_mix.items()}
    target_persona = {
        "Enterprise IT Buyer": math.ceil(total_target * 0.50),
        "Individual User": math.ceil(total_target * 0.50),
    }

    for record in sorted_pool:
        if micro_counts[(record.anchor_persona_id, record.persona_micro)] >= config.max_records_per_micro_persona:
            continue
        if intent_counts[record.intent_level] >= target_intents[record.intent_level]:
            continue
        if output_counts[record.output_type] >= target_outputs[record.output_type]:
            continue
        if persona_counts[record.persona_macro] >= target_persona[record.persona_macro]:
            continue

        selected.append(record)
        intent_counts[record.intent_level] += 1
        output_counts[record.output_type] += 1
        persona_counts[record.persona_macro] += 1
        micro_counts[(record.anchor_persona_id, record.persona_micro)] += 1

    # Fill remaining slots from best leftovers if quotas left gaps.
    leftovers = [r for r in sorted_pool if r not in selected]
    for record in leftovers:
        if len(selected) >= total_target:
            break
        if micro_counts[(record.anchor_persona_id, record.persona_micro)] >= config.max_records_per_micro_persona:
            continue
        selected.append(record)
        intent_counts[record.intent_level] += 1
        output_counts[record.output_type] += 1
        persona_counts[record.persona_macro] += 1
        micro_counts[(record.anchor_persona_id, record.persona_micro)] += 1

    # Promote selected pending records if they are needed to satisfy balance and are high enough quality.
    for record in selected:
        if record.review_status == "pending" and record.realism_score >= config.approve_threshold and set(record.drift_flags).issubset({"vocabulary_drift", "constraint_drift"}):
            record.review_status = "approved"

    return selected


def prune_stale_notes(notes: List[HybridMemoryNote], config: Optional[PipelineConfig] = None) -> List[HybridMemoryNote]:
    config = config or PipelineConfig()
    today = dt.datetime.utcnow().date()
    kept: List[HybridMemoryNote] = []
    for note in notes:
        try:
            note_date = dt.date.fromisoformat(note.last_update_date)
        except Exception:
            note_date = today
        age_days = (today - note_date).days
        if note.importance <= 2 and age_days > config.prune_days_low_importance:
            continue
        kept.append(note)
    return kept


def dedupe_memory_notes(notes: List[HybridMemoryNote]) -> List[HybridMemoryNote]:
    seen_text = {}
    out: List[HybridMemoryNote] = []
    for note in sorted(notes, key=lambda n: (n.importance, n.last_update_date), reverse=True):
        key = normalize_text(note.text).lower()
        if key in seen_text:
            continue
        seen_text[key] = True
        out.append(note)
    return out


def consolidate_state(
    state: HybridPersonaState,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    config = config or PipelineConfig()

    approved_records: List[CandidateRecord] = state.global_memory.get("approved_records", [])
    approved_notes = [record_to_memory_note(r) for r in approved_records if r.review_status == "approved"]
    anchor_notes = [build_anchor_summary_note(a) for a in state.anchor_registry.values()]
    session_candidate_notes = [record_to_memory_note(r) for r in state.session_memory.get("candidate_records", [])]

    proposed_global = dedupe_memory_notes(prune_stale_notes(anchor_notes + approved_notes, config=config))
    proposed_session = dedupe_memory_notes(prune_stale_notes(session_candidate_notes, config=config))

    # Critic checks (deterministic)
    critic_checks = {
        "no_pending_in_global": all(n.metadata.get("review_status") in [None, "approved"] for n in proposed_global),
        "anchor_notes_present": any(n.note_type == "anchor" for n in proposed_global),
        "no_duplicate_texts": len({normalize_text(n.text).lower() for n in proposed_global}) == len(proposed_global),
    }
    safe = all(critic_checks.values())

    if safe:
        state.global_memory["notes"] = [n.to_dict() for n in proposed_global]
        state.session_memory["notes"] = [n.to_dict() for n in proposed_session]
        state.inject_session_memories_next_turn = True

    return {"safe": safe, "critic_checks": critic_checks, "global_count": len(proposed_global), "session_count": len(proposed_session)}


# %%
# ----------------------------
# Context packet rendering and auditing
# ----------------------------

HYBRID_MEMORY_POLICY = """
<context_policy>
You may receive two memory lists:
- GLOBAL memory = stable anchor facts and approved synthesis examples.
- SESSION memory = temporary run-specific candidates, review notes, and recent overrides.

Precedence:
1) The current task/query overrides everything.
2) SESSION memory overrides GLOBAL memory for the current run when they conflict.
3) Anchor facts outrank synthetic expansions.
4) Approved records outrank pending records.
5) When two notes conflict, prefer the more recent and more important note.
6) Never use rejected records as evidence.
</context_policy>
""".strip()


def render_frontmatter(payload: Dict[str, Any]) -> str:
    dumped = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{dumped}\n---"


def render_notes_md(notes: Sequence[HybridMemoryNote], k: int) -> str:
    notes = list(notes)[:k]
    if not notes:
        return "- (none)"
    lines = []
    for note in notes:
        meta = []
        if note.persona_macro:
            meta.append(note.persona_macro)
        if note.intent_level:
            meta.append(note.intent_level)
        if note.topic_product_category:
            meta.append(note.topic_product_category)
        header = " | ".join(meta)
        if header:
            lines.append(f"- [{header}] {note.text}")
        else:
            lines.append(f"- {note.text}")
    return "\n".join(lines)


def score_note_relevance(query: str, note: HybridMemoryNote, expected: Dict[str, Any]) -> float:
    q_tokens = tokenize(query)
    note_tokens = tokenize(note.text + " " + " ".join(note.keywords))
    overlap = jaccard_overlap(q_tokens, note_tokens)
    topic_bonus = 0.0
    if expected.get("topic_product_category") and note.topic_product_category == expected.get("topic_product_category"):
        topic_bonus += 0.35
    persona_bonus = 0.0
    if expected.get("persona_macro") and note.persona_macro == expected.get("persona_macro"):
        persona_bonus += 0.25
    intent_bonus = 0.0
    if expected.get("intent_level") and note.intent_level == expected.get("intent_level"):
        intent_bonus += 0.15
    importance_bonus = 0.05 * min(note.importance, 5)
    return overlap + topic_bonus + persona_bonus + intent_bonus + importance_bonus


def select_notes_for_context(
    query: str,
    notes: Sequence[HybridMemoryNote],
    expected: Dict[str, Any],
    strategy: str,
    k: int,
) -> List[HybridMemoryNote]:
    notes = list(notes)
    if not notes:
        return []

    expected_topic = expected.get("topic_product_category")
    expected_persona = expected.get("persona_macro")

    preferred = [
        n for n in notes
        if (
            (
                (not expected_topic or n.topic_product_category == expected_topic)
                or (
                    n.note_type == "anchor"
                    and (
                        (expected_topic and n.topic_product_category == expected_topic)
                        or (expected_persona and n.persona_macro == expected_persona)
                    )
                )
            )
            and (
                (not expected_persona or n.persona_macro == expected_persona)
                or (
                    n.note_type == "anchor"
                    and (
                        (expected_persona and n.persona_macro == expected_persona)
                        or (expected_topic and n.topic_product_category == expected_topic)
                    )
                )
            )
        )
    ]
    backup = [n for n in notes if n not in preferred]

    if strategy == "relevance_only":
        ranked = sorted(notes, key=lambda n: score_note_relevance(query, n, expected), reverse=True)
        return ranked[:k]

    if strategy == "anchor_priority":
        ranked_preferred = sorted(
            preferred,
            key=lambda n: (
                n.note_type == "anchor",
                score_note_relevance(query, n, expected),
                n.importance,
                n.last_update_date,
            ),
            reverse=True,
        )
        if len(ranked_preferred) >= k:
            return ranked_preferred[:k]

        ranked_backup = sorted(
            backup,
            key=lambda n: (
                score_note_relevance(query, n, expected),
                n.importance,
                n.last_update_date,
            ),
            reverse=True,
        )
        return (ranked_preferred + ranked_backup)[:k]

    # default: relevance + recency + importance
    ranked = sorted(
        notes,
        key=lambda n: (
            score_note_relevance(query, n, expected),
            n.importance,
            n.last_update_date,
        ),
        reverse=True,
    )
    return ranked[:k]


def audit_context_packet(packet: ContextPacket, config: Optional[PipelineConfig] = None) -> Dict[str, Any]:
    config = config or PipelineConfig()
    selected = packet.selected_global + packet.selected_session
    issues: List[str] = []
    if not selected:
        issues.append("missing_context")

    topics = {n.topic_product_category for n in selected if n.topic_product_category}
    if packet.predicted_labels.get("topic_product_category") and packet.predicted_labels["topic_product_category"] not in topics:
        issues.append("missing_topic_support")

    if len(topics) > 3:
        issues.append("topic_leakage")

    anchor_share = sum(1 for n in selected if n.note_type == "anchor") / max(1, len(selected))
    if anchor_share < 0.12:
        issues.append("weak_anchor_grounding")

    if any(n.source_mode == "hybrid" and n.metadata.get("review_status") == "pending" for n in packet.selected_session):
        issues.append("pending_context_risk")

    total_tokens = estimate_tokens(packet.frontmatter + "\n" + packet.global_memories_md + "\n" + packet.session_memories_md)
    if total_tokens > config.context_token_budget:
        issues.append("context_overload")

    personas = {n.persona_macro for n in selected if n.persona_macro}
    if packet.predicted_labels.get("persona_macro") and packet.predicted_labels["persona_macro"] not in personas:
        issues.append("persona_mismatch")

    return {
        "issues": issues,
        "issue_count": len(issues),
        "selected_count": len(selected),
        "estimated_tokens": total_tokens,
        "anchor_share": round(anchor_share, 3),
        "topics": sorted([t for t in topics if t]),
    }


def build_context_packet(
    state: HybridPersonaState,
    labeler: HybridLabeler,
    query: str,
    strategy: str = "anchor_priority",
    config: Optional[PipelineConfig] = None,
) -> ContextPacket:
    config = config or PipelineConfig()
    expected = labeler.annotate(query)

    global_notes = []
    for note in state.global_memory.get("notes", []):
        if isinstance(note, HybridMemoryNote):
            global_notes.append(note)
        else:
            global_notes.append(HybridMemoryNote(**note))

    session_notes = []
    for note in state.session_memory.get("notes", []):
        if isinstance(note, HybridMemoryNote):
            session_notes.append(note)
        else:
            session_notes.append(HybridMemoryNote(**note))

    selected_global = select_notes_for_context(query, global_notes, expected, strategy=strategy, k=config.context_global_k)
    selected_session = select_notes_for_context(query, session_notes, expected, strategy=strategy, k=config.context_session_k)

    frontmatter_payload = {
        "query": query,
        "predicted_labels": {
            "persona_macro": expected["persona_macro"],
            "intent_level": expected["intent_level"],
            "topic_product_category": expected["topic_product_category"],
        },
        "strategy": strategy,
        "global_notes_selected": len(selected_global),
        "session_notes_selected": len(selected_session),
    }

    frontmatter = render_frontmatter(frontmatter_payload)
    global_md = render_notes_md(selected_global, k=config.context_global_k)
    session_md = render_notes_md(selected_session, k=config.context_session_k)

    packet = ContextPacket(
        query=query,
        predicted_labels=expected,
        frontmatter=frontmatter,
        global_memories_md=global_md,
        session_memories_md=session_md,
        selected_global=selected_global,
        selected_session=selected_session,
        audit={},
    )
    packet.audit = audit_context_packet(packet, config=config)
    return packet


# %%
# ----------------------------
# Evaluation helpers
# ----------------------------

def summarize_records(records: Sequence[CandidateRecord]) -> Dict[str, Any]:
    records = list(records)
    if not records:
        return {"count": 0}
    status_counts = Counter(r.review_status for r in records)
    drift_counts = Counter(flag for r in records for flag in r.drift_flags)
    intent_counts = Counter(r.intent_level for r in records)
    output_counts = Counter(r.output_type for r in records)
    topic_counts = Counter(r.topic_product_category for r in records)
    persona_counts = Counter(r.persona_macro for r in records)

    return {
        "count": len(records),
        "status_counts": dict(status_counts),
        "mean_realism": round(safe_mean([r.realism_score for r in records]), 3),
        "median_realism": round(float(statistics.median([r.realism_score for r in records])), 3),
        "drift_counts": dict(drift_counts),
        "intent_counts": dict(intent_counts),
        "output_counts": dict(output_counts),
        "topic_counts": dict(topic_counts),
        "persona_counts": dict(persona_counts),
    }


def compare_context_strategies(
    state: HybridPersonaState,
    labeler: HybridLabeler,
    queries: Sequence[str],
    config: Optional[PipelineConfig] = None,
) -> pd.DataFrame:
    config = config or PipelineConfig()
    rows = []
    for query in queries:
        for strategy in ["relevance_only", "anchor_priority"]:
            packet = build_context_packet(state, labeler, query, strategy=strategy, config=config)
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


# %%
# ----------------------------
# Orchestrator
# ----------------------------

class BalancedHybridPipeline:
    def __init__(
        self,
        codebook_df: pd.DataFrame,
        schema: Dict[str, Any],
        config: Optional[PipelineConfig] = None,
    ):
        self.config = config or PipelineConfig()
        self.rng = random.Random(self.config.seed)
        self.labeler = HybridLabeler(codebook_df)
        self.validator = CandidateValidator(self.labeler, schema=schema, config=self.config)
        self.state = HybridPersonaState(
            profile={
                "pipeline_name": "balanced_hybrid_persona_pipeline",
                "created_at": utc_today_iso(),
                "policy": "anchor > approved hybrid > pending hybrid",
            }
        )
        self.records_df = pd.DataFrame()

    def register_anchors(self, anchor_cards: pd.DataFrame | List[Dict[str, Any]]) -> Dict[str, AnchorPersona]:
        observed = self.records_df if not self.records_df.empty else None
        registry = build_anchor_registry(anchor_cards, observed_records=observed, config=self.config)
        self.state.anchor_registry = registry
        return registry

    def ingest_observed_records(
        self,
        records: pd.DataFrame,
        source_mode: str = "observed",
        assign_anchor: bool = True,
    ) -> pd.DataFrame:
        df = records.copy()
        if "text" not in df.columns:
            raise ValueError("Observed records must contain a 'text' column.")

        rows = []
        for _, row in df.iterrows():
            text = normalize_text(row["text"])
            annotation = self.labeler.annotate(text)
            anchor_id = row.get("anchor_persona_id")
            anchor_score = None
            if assign_anchor and self.state.anchor_registry and not anchor_id:
                anchor_id, anchor_score = assign_best_anchor(text, annotation, self.state.anchor_registry)

            row_dict = dict(row)
            row_dict.update(annotation)
            row_dict["source_mode"] = source_mode
            row_dict["anchor_persona_id"] = anchor_id
            row_dict["anchor_assignment_score"] = anchor_score
            row_dict["review_status"] = row_dict.get("review_status", "approved")
            rows.append(row_dict)

        annotated = pd.DataFrame(rows)
        self.records_df = pd.concat([self.records_df, annotated], ignore_index=True) if not self.records_df.empty else annotated
        return annotated

    def plan_generation(self) -> List[GenerationSpec]:
        return build_generation_specs(self.state.anchor_registry, self.records_df, config=self.config)

    def generate_candidates(
        self,
        adapter: Optional[BaseGeneratorAdapter] = None,
        n_per_spec: int = 1,
    ) -> List[CandidateRecord]:
        adapter = adapter or HeuristicPersonaHubAdapter(seed=self.config.seed)
        specs = self.plan_generation()
        candidates: List[CandidateRecord] = []

        # Build anchor-specific corpora for realism scoring.
        corpora: Dict[str, List[str]] = defaultdict(list)
        if not self.records_df.empty and "anchor_persona_id" in self.records_df.columns:
            for _, row in self.records_df.dropna(subset=["text"]).iterrows():
                anchor_id = row.get("anchor_persona_id")
                if anchor_id:
                    corpora[str(anchor_id)].append(str(row["text"]))

        for spec in specs:
            anchor = self.state.anchor_registry[spec.anchor_persona_id]
            generated_texts = adapter.generate(spec, anchor, n=n_per_spec)
            for text in generated_texts:
                base_evidence = dedupe_preserve_order(tokenize(text)[:4] + [spec.intent_level, spec.topic_product_category])
                record = CandidateRecord(
                    record_id=f"rec_{stable_hash('|'.join([spec.anchor_persona_id, spec.persona_micro, spec.intent_level, spec.output_type, text]))}",
                    source_mode=spec.source_mode,
                    anchor_persona_id=spec.anchor_persona_id,
                    persona_macro=spec.persona_macro,
                    persona_micro=spec.persona_micro,
                    intent_level=spec.intent_level,
                    topic_product_category=spec.topic_product_category,
                    job_to_be_done=anchor.job_to_be_done,
                    constraints=anchor.constraints,
                    success_metric=anchor.success_metric,
                    decision_criteria=anchor.decision_criteria,
                    vocabulary=anchor.vocabulary,
                    text=text,
                    output_type=spec.output_type,
                    label_evidence=base_evidence,
                    label_confidence="Medium",
                    review_status="pending",
                    importance=HYBRID_IMPORTANCE,
                    generation_mode=spec.mode,
                    planned_persona_macro=spec.persona_macro,
                    planned_intent_level=spec.intent_level,
                    planned_topic_product_category=spec.topic_product_category,
                )
                validated = self.validator.validate_candidate(record, anchor, corpus_texts=corpora.get(spec.anchor_persona_id, []))
                candidates.append(validated)

        self.state.session_memory["candidate_records"] = candidates
        self.state.session_memory["review_queue"] = [c for c in candidates if c.review_status != "approved"]
        return candidates

    def rebalance(self, candidates: Optional[Sequence[CandidateRecord]] = None) -> List[CandidateRecord]:
        candidates = list(candidates or self.state.session_memory.get("candidate_records", []))
        selected = rebalance_records(candidates, config=self.config)
        return selected

    def consolidate(self, selected_records: Optional[Sequence[CandidateRecord]] = None) -> Dict[str, Any]:
        selected_records = list(selected_records or self.state.session_memory.get("candidate_records", []))
        approved = [r for r in selected_records if r.review_status == "approved"]
        self.state.global_memory["approved_records"] = approved
        return consolidate_state(self.state, config=self.config)

    def build_context_packet(self, query: str, strategy: str = "anchor_priority") -> ContextPacket:
        return build_context_packet(self.state, self.labeler, query, strategy=strategy, config=self.config)

    def evaluate(self, candidates: Optional[Sequence[CandidateRecord]] = None) -> Dict[str, Any]:
        candidates = list(candidates or self.state.session_memory.get("candidate_records", []))
        summary = summarize_records(candidates)
        self.state.eval_memory["metrics"]["generation"] = summary
        return summary


# %%
# ----------------------------
# Demo inputs and exports
# ----------------------------

def demo_anchor_cards() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "anchor_persona_id": "anchor_ent_remote_access",
                "macro_persona": "Enterprise IT Buyer",
                "anchor_name": "Distributed Access IT Manager",
                "job_to_be_done": "secure remote access for distributed staff",
                "constraints": ["limited admin time", "compliance review", "budget ceiling"],
                "success_metric": "reliable remote access with fewer support tickets",
                "decision_criteria": ["easy deployment", "SSO support", "policy control"],
                "vocabulary": ["remote workforce", "access policy", "SSO", "compliance"],
                "evidence_sources": ["gsc_last_28_days", "support_transcripts", "survey_q3"],
                "confidence_score": 0.88,
                "priority_topics": ["VPN & Zero Trust Access", "Identity & Access Management"],
                "allowed_intent_levels": ["Low", "Medium", "High"],
            },
            {
                "anchor_persona_id": "anchor_ent_endpoint",
                "macro_persona": "Enterprise IT Buyer",
                "anchor_name": "Lean Security Operations Lead",
                "job_to_be_done": "standardize endpoint protection for a growing workforce",
                "constraints": ["lean security team", "audit readiness", "device diversity"],
                "success_metric": "reduced incident volume and consistent device coverage",
                "decision_criteria": ["EDR visibility", "ease of rollout", "policy reporting"],
                "vocabulary": ["endpoint", "EDR", "device policy", "security review"],
                "evidence_sources": ["gsc_last_28_days", "win_loss_notes"],
                "confidence_score": 0.84,
                "priority_topics": ["Endpoint Security & Threat Protection", "Procurement & Vendor Evaluation"],
                "allowed_intent_levels": ["Low", "Medium", "High"],
            },
            {
                "anchor_persona_id": "anchor_ind_notes",
                "macro_persona": "Individual User",
                "anchor_name": "Graduate Student Note Organizer",
                "job_to_be_done": "keep class and research notes organized across devices",
                "constraints": ["student budget", "easy setup", "offline access"],
                "success_metric": "faster retrieval and less note sprawl",
                "decision_criteria": ["affordable", "sync", "search"],
                "vocabulary": ["class notes", "organization", "study workflow", "sync"],
                "evidence_sources": ["gsc_last_28_days", "student_surveys"],
                "confidence_score": 0.86,
                "priority_topics": ["Productivity & Note-Taking", "Skills & Training"],
                "allowed_intent_levels": ["Low", "Medium", "High"],
            },
            {
                "anchor_persona_id": "anchor_ind_privacy",
                "macro_persona": "Individual User",
                "anchor_name": "Household Privacy Caretaker",
                "job_to_be_done": "protect family accounts and personal files without admin overhead",
                "constraints": ["non-technical household", "affordability", "minimal maintenance"],
                "success_metric": "fewer lockouts and safer everyday use",
                "decision_criteria": ["simple setup", "password management", "backup"],
                "vocabulary": ["family accounts", "password manager", "backup", "privacy"],
                "evidence_sources": ["gsc_last_28_days", "help_center_searches"],
                "confidence_score": 0.82,
                "priority_topics": ["Consumer Security & Privacy", "Backup, Storage & Continuity"],
                "allowed_intent_levels": ["Low", "Medium", "High"],
            },
        ]
    )


def demo_observed_records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"text": "what is zero trust access for remote employees", "anchor_persona_id": "anchor_ent_remote_access"},
            {"text": "best vpn for remote employees with sso", "anchor_persona_id": "anchor_ent_remote_access"},
            {"text": "zero trust pricing for 200 employees", "anchor_persona_id": "anchor_ent_remote_access"},
            {"text": "sso options for a growing startup", "anchor_persona_id": "anchor_ent_remote_access"},
            {"text": "endpoint protection comparison for 200 employees", "anchor_persona_id": "anchor_ent_endpoint"},
            {"text": "edr pricing for 100 endpoints", "anchor_persona_id": "anchor_ent_endpoint"},
            {"text": "how to evaluate endpoint security vendors", "anchor_persona_id": "anchor_ent_endpoint"},
            {"text": "best edr for lean it team", "anchor_persona_id": "anchor_ent_endpoint"},
            {"text": "best note taking app for college", "anchor_persona_id": "anchor_ind_notes"},
            {"text": "notion vs evernote for class notes", "anchor_persona_id": "anchor_ind_notes"},
            {"text": "how to organize research notes across devices", "anchor_persona_id": "anchor_ind_notes"},
            {"text": "student discount note taking app with offline sync", "anchor_persona_id": "anchor_ind_notes"},
            {"text": "best password manager for families", "anchor_persona_id": "anchor_ind_privacy"},
            {"text": "how to store family documents online safely", "anchor_persona_id": "anchor_ind_privacy"},
            {"text": "backup plan for family photos and documents", "anchor_persona_id": "anchor_ind_privacy"},
            {"text": "simple way to secure family accounts without tech skills", "anchor_persona_id": "anchor_ind_privacy"},
        ]
    )


def export_records_jsonl(records: Sequence[CandidateRecord], path: str | Path) -> Path:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return path


def export_context_packet(packet: ContextPacket, path: str | Path) -> Path:
    payload = {
        "query": packet.query,
        "predicted_labels": packet.predicted_labels,
        "frontmatter": packet.frontmatter,
        "global_memories_md": packet.global_memories_md,
        "session_memories_md": packet.session_memories_md,
        "audit": packet.audit,
        "selected_global": [n.to_dict() for n in packet.selected_global],
        "selected_session": [n.to_dict() for n in packet.selected_session],
        "memory_policy": HYBRID_MEMORY_POLICY,
    }
    path = Path(path)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def demo_run(
    codebook_path: str | Path,
    schema_path: str | Path,
    output_dir: str | Path,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    codebook_df = load_codebook_csv(codebook_path)
    schema = load_json_schema(schema_path)
    pipeline = BalancedHybridPipeline(codebook_df, schema, config=config)

    anchors = demo_anchor_cards()
    pipeline.register_anchors(anchors)
    observed = pipeline.ingest_observed_records(demo_observed_records(), source_mode="observed", assign_anchor=False)

    candidates = pipeline.generate_candidates()
    balanced = pipeline.rebalance(candidates)
    consolidation = pipeline.consolidate(balanced)
    evaluation = pipeline.evaluate(candidates)

    query = "best zero trust access tools for remote employees with sso"
    packet = pipeline.build_context_packet(query, strategy="anchor_priority")
    ab_df = compare_context_strategies(
        pipeline.state,
        pipeline.labeler,
        queries=[
            query,
            "best note taking app for college with offline sync",
            "edr pricing for 100 endpoints",
        ],
        config=pipeline.config,
    )

    candidates_path = export_records_jsonl(candidates, output_dir / "generated_candidates.jsonl")
    balanced_path = export_records_jsonl(balanced, output_dir / "balanced_selected_records.jsonl")
    packet_path = export_context_packet(packet, output_dir / "context_packet.yaml")
    ab_df.to_csv(output_dir / "context_strategy_comparison.csv", index=False)
    (output_dir / "generation_metrics.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    (output_dir / "consolidation_report.json").write_text(json.dumps(consolidation, indent=2), encoding="utf-8")

    return {
        "pipeline": pipeline,
        "observed": observed,
        "candidates": candidates,
        "balanced": balanced,
        "packet": packet,
        "ab_df": ab_df,
        "paths": {
            "generated_candidates": str(candidates_path),
            "balanced_selected_records": str(balanced_path),
            "context_packet": str(packet_path),
            "context_strategy_comparison": str(output_dir / "context_strategy_comparison.csv"),
            "generation_metrics": str(output_dir / "generation_metrics.json"),
            "consolidation_report": str(output_dir / "consolidation_report.json"),
        },
    }
