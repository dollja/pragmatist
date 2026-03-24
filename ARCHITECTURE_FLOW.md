# Architecture Flow

This document explains the essential pieces of the package in plain language.

## One-line view

**Ground with anchor personas, expand carefully, validate aggressively, inject only the best context for each turn, and learn from packet failures.**

## Flow diagram

```mermaid
flowchart TD
    A[Grounded evidence\nGSC, surveys, transcripts] --> B[Anchor persona cards]
    B --> C[BalancedHybridPipeline]
    D[Observed records] --> C
    C --> E[Controlled synthetic expansion]
    E --> F[Validation and drift checks]
    F --> G[Balanced selection]
    G --> H[Consolidation into memory notes]
    H --> I[HybridPersonaState]
    I --> J[build_context_packet(query)]
    J --> K[PragmatistHybridBridge]
    K --> L[Runtime hooks and instructions]
    L --> M[state.runtime.context_history]
    L --> N[state.runtime.context_errors]
    M --> O[ContextErrorEngineeringLayer]
    N --> O
    O --> P[Mitigations, runtime controls, A/B re-evaluation]
```

## The same flow in simple words

### Step 1: Ground the system with evidence
The package assumes you start with **anchor personas** that come from real evidence such as search queries, surveys, or transcripts.

**Why this exists:**  
This prevents the synthetic generator from becoming generic.

**What goes in:**  
Grounded persona cards and observed records.

**What comes out:**  
A trustworthy starting point for controlled expansion.

---

### Step 2: Expand around the anchors
The generation stage creates additional records around the anchors rather than inventing a whole new market.

**Why this exists:**  
You want more coverage and wording variety, but you still want the generated items to sound like the same user family.

**What goes in:**  
Anchor cards, observed records, label taxonomy, and generation settings.

**What comes out:**  
Candidate records across personas, intents, topics, and output types.

---

### Step 3: Filter out bad synthetic records
Every generated record is checked for:
- schema validity
- persona drift
- topic drift
- intent drift
- vocabulary drift
- constraint drift

**Why this exists:**  
Without this stage, the expansion layer can quietly move away from the anchor.

**What goes in:**  
Raw candidate records.

**What comes out:**  
A smaller, cleaner, more defensible candidate set.

---

### Step 4: Rebalance and consolidate
After filtering, the pipeline keeps a balanced slice and converts the selected records into memory notes.

**Why this exists:**  
You do not want your final context memory to overrepresent one topic, one intent band, or one persona.

**What goes in:**  
Validated candidates.

**What comes out:**  
Global memory notes, session memory notes, and a state object ready for runtime use.

---

### Step 5: Build a turn-specific context packet
When a new query arrives, `build_context_packet(...)` predicts labels and chooses the most relevant notes.

**Why this exists:**  
The model should not receive the whole memory store every turn. It needs a focused briefing.

**What goes in:**  
The current query plus the consolidated state.

**What comes out:**  
A context packet with frontmatter, selected notes, and diagnostics.

---

### Step 6: Bridge the packet into the live runtime
`PragmatistHybridBridge` turns the packet builder into something you can use inside a notebook or agent workflow.

**Why this exists:**  
Offline data generation and live turn-time context injection are different problems. The bridge connects them cleanly.

**What goes in:**  
A balanced hybrid pipeline that has already been consolidated.

**What comes out:**  
Hooks, instructions, packet history, and runtime controls.

---

### Step 7: Watch for context failures
Every hydrated turn can produce packet audits. Those audits are stored in:
- `state.runtime["context_history"]`
- `state.runtime["context_errors"]`

**Why this exists:**  
Good context engineering needs observability. You need to see where the packet builder fails.

**What goes in:**  
Packet audit summaries from live turns.

**What comes out:**  
A raw history of what happened and what went wrong.

---

### Step 8: Convert failures into mitigation actions
The `ContextErrorEngineeringLayer` groups repeated issues and recommends actions such as:
- prefer anchor-priority retrieval
- shrink the context window
- disable pending records
- reduce synthetic weight
- queue coverage gaps

**Why this exists:**  
Logging alone does not improve the system. This layer turns recurring failure patterns into operational changes.

**What goes in:**  
Context history and error streams.

**What comes out:**  
Health summaries, recommendations, and optional runtime updates.

## What each core component is doing

## 1) `BalancedHybridPipeline`
**Role:** Offline synthesis and preparation engine  
**Simple explanation:** This is the factory that builds the high-quality records.  
**Main job:** Take grounded personas and observed records, then expand, validate, and consolidate them.

## 2) `HybridPersonaState`
**Role:** Shared state container  
**Simple explanation:** This is the memory cabinet for the whole system.  
**Main job:** Keep the anchor registry, memory notes, candidate records, and run metadata in one place.

## 3) `build_context_packet(...)`
**Role:** Turn-time context selector  
**Simple explanation:** This is the briefing writer for the current query.  
**Main job:** Choose which notes matter right now and package them into a compact context bundle.

## 4) `compare_context_strategies(...)`
**Role:** Strategy evaluator  
**Simple explanation:** This is the referee between different retrieval policies.  
**Main job:** Test whether one packet-selection strategy grounds the system better than another.

## 5) `PragmatistHybridBridge`
**Role:** Runtime adapter  
**Simple explanation:** This is the connector that plugs the offline persona system into a live notebook or agent loop.  
**Main job:** Expose hooks, instructions, packet hydration, and strategy testing in a Pragmatist-style workflow.

## 6) `ContextErrorEngineeringLayer`
**Role:** Error analysis and mitigation engine  
**Simple explanation:** This is the quality-control dashboard plus repair planner.  
**Main job:** Detect repeated context problems and suggest or apply fixes.

## Why the architecture is split this way

The design keeps three concerns separate on purpose:

1. **Generating records**
2. **Choosing turn-time context**
3. **Improving context policy after observing failures**

That split matters because each concern fails differently:

- generation can drift from the anchor
- packet selection can overload or under-cover the model
- runtime can accumulate repeated weak-grounding errors

Keeping them separate makes the system easier to debug, test, and evolve.

## The most important design rule

If you only keep one rule in mind, keep this one:

**The expansion layer is allowed to add breadth, but only the grounded anchor layer is allowed to define the truth boundaries of the persona.**

