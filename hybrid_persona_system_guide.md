# Hybrid Persona-Driven Data Synthesis System Guide

## Purpose

This guide turns the earlier chat, the Synthetic Persona Script workflow, and PersonaHub into a concrete labeling rubric and generation schema for three axes:

1. Persona
2. Intent level
3. Topic / product category

The goal is to combine:
- **Specificity and evidence tracing** from your real GSC / survey / transcript data
- **Scalability and breadth** from PersonaHub’s large persona bank and prompt-generation methods

---

## Core idea

Use a **two-layer system**:

### Layer 1: Anchor layer (grounded)
Use the Synthetic Persona Script on:
- last 28 days of GSC question-type queries
- optional survey CSVs
- optional transcripts

This gives you:
- evidence-backed persona cards
- 5-field persona structure
- confidence scores
- vocabulary grounded in real customer language
- trackable prompts by low / medium / high intent

### Layer 2: Expansion layer (scalable)
Use PersonaHub to:
- sample adjacent or long-tail personas
- expand prompt coverage inside each topic
- generate more prompts and optional conversations
- simulate missing stakeholder viewpoints

Keep the anchor layer as your source of truth. Use the expansion layer only to broaden coverage.

---

## What each component should do

### Synthetic Persona Script should do:
- produce 2–3 grounded seed personas
- define:
  - job-to-be-done
  - constraints
  - success metric
  - decision criteria
  - vocabulary
- generate 15–30 prompts per persona across low / medium / high intent
- trace every field back to source evidence

### PersonaHub should do:
- provide many candidate personas for expansion
- help generate more first-turn prompts
- help simulate subsequent conversations
- let you customize prompt templates
- help cover long-tail variants and adjacent stakeholders

### Rule of thumb
If the question is:
- “What do my real users sound like?” → trust the Script
- “What other plausible users / prompts might I be missing?” → use PersonaHub
- “What should go into the final evaluation set?” → only keep outputs that still fit your labeling rubric and anchor evidence

---

## Step 1: Define your closed label set

Do not leave labels open-ended. Start with a fixed codebook.

### Axis A: Persona
Allowed labels:
- `Enterprise IT Buyer`
- `Individual User`

### Axis B: Intent level
Allowed labels:
- `Low`
- `Medium`
- `High`

### Axis C: Topic / product category
Start with a closed list such as:
- Identity & Access Management
- Endpoint Security & Threat Protection
- VPN & Zero Trust Access
- Backup, Storage & Continuity
- Procurement & Vendor Evaluation
- Productivity & Note-Taking
- CRM & Marketing Tools
- Web & Site Building
- Consumer Security & Privacy
- End-User Devices & Hardware
- Skills & Training
- Collaboration & Project Management

Adjust the topic list only after reviewing at least 50 real rows.

---

## Step 2: Build the labeling rubric

## A. Persona rubric

### Label: Enterprise IT Buyer
Use this label when the query or prompt reflects organizational purchase, deployment, governance, or vendor-selection context.

Typical cues:
- mentions team, company, business, department, employees, admin, organization
- mentions deployment, rollout, implementation, integration, SSO, SOC2, compliance, procurement, RFP, vendor, SLA
- asks about multi-user management, central control, security policy, enterprise support
- implies multiple stakeholders or approval chains

Examples:
- “best vpn for small business remote team”
- “sso options for growing startup”
- “endpoint protection comparison for 200 employees”

### Label: Individual User
Use this label when the query reflects personal use, household use, solo work, education, or self-serve buying.

Typical cues:
- personal goals: learn, organize, budget, study, family, home, freelance
- price or ease-of-use for one person
- no explicit org procurement or governance context
- asks about personal productivity, classes, files, privacy, note taking

Examples:
- “how to learn excel for budgeting”
- “best note taking app for college”
- “how to store family documents online”

### Persona tie-breaker
Ask:
> Is the decision context mainly organizational, or mainly personal?

If organizational consequences are central, choose `Enterprise IT Buyer`.
If personal outcomes are central, choose `Individual User`.

If still unclear:
- keep the initial label
- add `persona_confidence = low`
- add a note with the ambiguous evidence

---

## B. Intent rubric

### Low intent
The user is still learning or framing the problem.

Typical forms:
- what is
- how to
- why
- definition
- meaning
- benefits
- examples
- guide
- tutorial
- learn

Behavioral interpretation:
- exploring the problem space
- not yet comparing vendors or solutions
- no immediate purchase or implementation signal

Examples:
- “what is zero trust access”
- “how to learn excel for budgeting”
- “benefits of password managers”

### Medium intent
The user is evaluating options or narrowing choices.

Typical forms:
- best
- top
- comparison
- vs
- alternatives
- checklist
- framework
- features
- pros and cons

Behavioral interpretation:
- solution-aware
- comparing approaches, tools, or categories
- not yet at final proof-point or purchase stage

Examples:
- “notion vs evernote for class notes”
- “best crm for freelancers”
- “vpn alternatives for remote teams”

### High intent
The user is close to acting, adopting, buying, deploying, or validating a final choice.

Typical forms:
- pricing
- demo
- trial
- implementation
- migration
- proof of concept
- ROI
- security review
- compliance
- approval
- buy
- sign up
- deploy

Behavioral interpretation:
- wants proof points, validation, or rollout details
- next likely step is an action, not just research

Examples:
- “how to migrate to google workspace from office 365”
- “edr pricing for 100 endpoints”
- “proof that vpn solution meets compliance requirements”

### Intent tie-breaker
Ask:
> What is the user’s most likely next action in the next 1–2 weeks?

- learn → `Low`
- compare / shortlist → `Medium`
- buy / deploy / validate / sign up → `High`

---

## C. Topic rubric

Label topic before persona or intent when possible.

### Recommended topic rules
- If the core noun is about identity, SSO, login, MFA, permissions → `Identity & Access Management`
- If the core noun is endpoint, antivirus, EDR, malware → `Endpoint Security & Threat Protection`
- If the core noun is VPN, remote access, zero trust → `VPN & Zero Trust Access`
- If the core noun is backup, storage, recovery, file retention → `Backup, Storage & Continuity`
- If the query is primarily about choosing among vendors → `Procurement & Vendor Evaluation`
- If the core task is notes, task organization, personal knowledge management → `Productivity & Note-Taking`
- If the core task is CRM, email marketing, customer tracking → `CRM & Marketing Tools`
- If the core task is website building, CMS, hosting → `Web & Site Building`
- If the core task is personal privacy, password managers, home security tools → `Consumer Security & Privacy`
- If the core object is a physical device → `End-User Devices & Hardware`
- If the core action is learning a tool or skill → `Skills & Training`
- If the core task is team collaboration, docs, project boards → `Collaboration & Project Management`

### Topic tie-breaker
Ask:
> What is the primary object being chosen, learned, or acted on?

If the query compares two vendors but both belong to one domain, choose:
- the domain topic if the comparison is domain-specific
- `Procurement & Vendor Evaluation` only if the comparison itself is the main purpose

---

## Step 3: Add evidence and confidence columns

For every labeled row, add:
- `Persona`
- `Intent_Level`
- `Topic_Product_Category`
- `Label_Evidence`
- `Label_Confidence`

### Confidence rules
- `High` = direct lexical evidence (“for my team”, “SSO”, “pricing”)
- `Medium` = strong indirect evidence
- `Low` = ambiguous or inferred from weak cues

### Example
Query:
`best vpn for remote employees`

Labels:
- Persona = Enterprise IT Buyer
- Intent = Medium
- Topic = VPN & Zero Trust Access
- Evidence = “remote employees”, “best vpn”
- Confidence = High

---

## Step 4: Convert Script outputs into anchor personas

For each Script-generated persona card, create one normalized anchor record.

## Anchor persona schema
- `anchor_persona_id`
- `macro_persona` (`Enterprise IT Buyer` or `Individual User`)
- `anchor_name`
- `job_to_be_done`
- `constraints`
- `success_metric`
- `decision_criteria`
- `vocabulary`
- `evidence_sources`
- `confidence_score`
- `priority_topics`
- `allowed_intent_levels`

### Why this matters
The Script’s 5-field card becomes your **control structure**.
Do not expand PersonaHub outputs unless they can be mapped back to one of these anchor fields.

---

## Step 5: Use anchor personas as few-shot demonstrations for PersonaHub

This is the key hybrid move.

Instead of using PersonaHub in pure zero-shot mode, use:
- your Script-generated persona cards as anchor personas
- your Script-generated prompts as few-shot exemplars
- PersonaHub as the expansion pool for adjacent personas

### What to do
For each anchor persona:
1. Pick 3–5 representative prompts from the Script
   - 1 low-intent
   - 1–2 medium-intent
   - 1 high-intent
2. Select PersonaHub personas that are:
   - in the same macro persona bucket
   - topically adjacent
   - compatible with the anchor’s JTBD and vocabulary
3. Generate new prompts with those personas
4. Keep only outputs that still fit the anchor’s topic, intent band, and decision context

### Example
Anchor persona:
- `Enterprise IT Buyer`
- JTBD: secure remote access for distributed staff
- Constraints: budget, compliance, manageability
- Vocabulary: SSO, remote workforce, policy, access control

PersonaHub expansion personas:
- IT manager at a mid-sized firm
- security analyst responsible for endpoint policies
- helpdesk lead onboarding remote employees
- procurement analyst comparing vendors

This expands breadth without leaving the real business context.

---

## Step 6: Use two expansion modes

### Mode 1: Within-anchor expansion
Use PersonaHub personas that are clearly compatible with the anchor persona.
Goal: broaden language variety without changing the underlying use case.

Best for:
- evaluation prompts
- long-tail query variants
- coverage expansion

### Mode 2: Stakeholder-neighbor expansion
Use persona-to-persona logic to expand adjacent stakeholders around the same buying journey.

Examples:
For an enterprise anchor:
- economic buyer
- technical evaluator
- security reviewer
- administrator
- end user

For an individual anchor:
- student
- parent
- freelancer
- hobbyist
- privacy-conscious home user

Goal:
Capture variation in language and priorities while still remaining inside the same product/topic area.

---

## Step 7: Use separate schemas for three output types

### Output type 1: Search-style queries
Short, compact, likely to resemble GSC rows.

Fields:
- `prompt_id`
- `persona`
- `intent_level`
- `topic`
- `text`
- `max_words`
- `query_style = search`

### Output type 2: Assistant-style prompts
Longer, more explicit first-turn prompts for LLM testing.

Fields:
- `prompt_id`
- `persona`
- `intent_level`
- `topic`
- `text`
- `query_style = assistant_prompt`

### Output type 3: Transcript turns
Multi-turn synthetic interactions.

Fields:
- `conversation_id`
- `speaker_role`
- `turn_index`
- `persona`
- `intent_level`
- `topic`
- `text`
- `source_anchor_persona`
- `evidence_note`

---

## Step 8: Use this record-level generation schema

```json
{
  "record_id": "uuid",
  "source_mode": "observed|script_generated|personahub_zero_shot|personahub_few_shot|hybrid",
  "anchor_persona_id": "anchor_001",
  "persona_macro": "Enterprise IT Buyer",
  "persona_micro": "IT manager at a 300-person company",
  "intent_level": "Medium",
  "topic_product_category": "VPN & Zero Trust Access",
  "job_to_be_done": "secure remote access for distributed staff",
  "constraints": ["limited admin time", "compliance review", "budget ceiling"],
  "success_metric": "reliable remote access with fewer support tickets",
  "decision_criteria": ["easy deployment", "SSO support", "policy control"],
  "vocabulary": ["remote workforce", "access policy", "SSO", "compliance"],
  "text": "best zero trust access tools for remote employees with sso",
  "output_type": "search_query",
  "label_evidence": ["zero trust", "remote employees", "sso"],
  "label_confidence": "High",
  "realism_score": 0.84,
  "drift_flags": [],
  "review_status": "approved"
}
```

---

## Step 9: Add a drift-control layer

Synthetic expansion gets weaker when it drifts away from the anchor persona.

Add these checks before keeping a generated item:

### Drift check A: Persona drift
Does the generated item still reflect the same decision context?
- organizational vs personal
- same level of authority
- same use case family

### Drift check B: Intent drift
Does the generated wording fit the intended intent band?
- low = learn
- medium = compare
- high = act

### Drift check C: Topic drift
Does the core noun still belong to the intended topic?
If not, reject or relabel.

### Drift check D: Vocabulary drift
Does the item preserve at least some anchor vocabulary or close synonyms?
If not, score realism lower.

### Drift check E: Constraint drift
Does the output contradict anchor constraints?
Examples:
- a budget-sensitive anchor should not suddenly ask about “premium white-glove enterprise migration” unless intentionally expanded

---

## Step 10: Suggested review workflow

### Pass 1: automatic
- label topic with rules or classifier
- label persona with rules
- label intent with rules
- deduplicate near-duplicates
- reject extremely long “query” strings if output type is search query

### Pass 2: human spot check
Review:
- 10% of kept outputs
- all low-confidence labels
- all high-intent enterprise prompts
- all transcript expansions

### Pass 3: rebalance
Make sure you do not overproduce easy medium-intent prompts.
Try to keep a planned distribution.

Example target:
- 30% Low
- 45% Medium
- 25% High

---

## Step 11: A practical starter configuration

For a first balanced pilot:

### Anchors
- 2 Script personas for `Enterprise IT Buyer`
- 2 Script personas for `Individual User`

### Per anchor
- 12 Script-grounded prompts retained
- 24 PersonaHub expansions generated
- keep only the best 10–12 after validation

### Final pilot size
- 4 anchor personas
- around 80–100 approved prompts
- optional 8–12 synthetic transcript snippets

This is large enough to test the system, but small enough to manually audit.

---

## Step 12: Maintenance rules

Refresh the anchor layer when:
- new 28-day GSC export is available
- product positioning changes
- competitors shift the market narrative
- transcript vocabulary changes
- a new buyer objection starts appearing repeatedly

Refresh the expansion layer whenever the anchor layer changes.

---

## Recommended operating principle

**Anchor with real evidence. Expand with synthetic breadth. Validate against the anchor.**

That one rule will keep the hybrid system balanced.

