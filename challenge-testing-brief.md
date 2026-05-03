# magicpin AI Challenge — Testing & Evaluation Brief

**Status**: Companion to `challenge-brief.md`. Defines the technical contract between candidate bots and magicpin's judging system.
**Last updated**: 2026-04-26
**Audience**: Candidates building the bot + magicpin engineers running the harness.

> **Read this with `challenge-brief.md`** — that brief defines what the bot must do (the 4-context framework, composition contract, evaluation rubric). This brief defines *how the bot is tested* (HTTP API, judge harness, scoring touchpoints).

---

## 1. The high-level model

```
   ┌──────────────────────────┐                    ┌────────────────────────────┐
   │  magicpin Judge Harness  │                    │  Candidate Bot (your code) │
   │  (LLM playing merchant + │                    │  HTTP server, public URL   │
   │   context injector +     │ ──── HTTP/JSON ──► │  Stateful per-conversation │
   │   scorer)                │ ◄──── HTTP/JSON ── │                            │
   └──────────────────────────┘                    └────────────────────────────┘
```

Two information flows:

1. **Judge → Bot**: pushes incremental context across all 4 dimensions (category, merchant, trigger, customer) over time. Mimics how production data updates flow into Vera.
2. **Bot → Judge**: the bot proactively initiates conversations, and the judge plays the merchant (or customer) role, replying realistically. The bot must handle the full conversation.

The bot is **stateful** (must remember context pushed earlier and conversations in flight). The judge is the source of truth for what's happened.

---

## 2. Endpoints the candidate must expose

All endpoints are HTTPS, JSON in/out, UTF-8. Total surface: **5 endpoints**.

### 2.1 `POST /v1/context` — receive a context push

The judge calls this whenever it wants the bot to know about new or updated context. Could be the initial base dataset, or an incremental update mid-test.

**Request body**:
```json
{
  "scope": "category" | "merchant" | "customer" | "trigger",
  "context_id": "dentists" | "m_001_drmeera" | "c_001_priya" | "trg_2026_04_26_research_digest",
  "version": 3,
  "payload": { /* the full context object — see §3 */ },
  "delivered_at": "2026-04-26T10:00:00Z"
}
```

**Behavior**:
- **Idempotent** by `(context_id, version)`. Re-posting the same version is a no-op.
- A higher `version` for the same `context_id` **replaces** the prior version atomically.
- Bot must persist context until the test ends. Storing in memory is fine; just don't restart between calls.

**Response (200)**:
```json
{ "accepted": true, "ack_id": "ack_abc123", "stored_at": "2026-04-26T10:00:00.123Z" }
```

**Response (409)** — version conflict (you already have a higher version):
```json
{ "accepted": false, "reason": "stale_version", "current_version": 5 }
```

**Response (400)** — malformed:
```json
{ "accepted": false, "reason": "invalid_scope", "details": "..." }
```

### 2.2 `POST /v1/tick` — periodic wake-up; bot can initiate

The judge calls this every **N seconds of simulated time** (default: every 5 simulated minutes). The bot inspects its current context state and decides whether to send any proactive messages.

**Request body**:
```json
{
  "now": "2026-04-26T10:30:00Z",
  "available_triggers": ["trg_2026_04_26_research_digest", "trg_2026_04_26_recall_priya"]
}
```

`available_triggers` is a hint listing trigger context_ids the judge considers "active right now". The bot can use any subset (or none).

**Response (200)**:
```json
{
  "actions": [
    {
      "conversation_id": "conv_001",
      "merchant_id": "m_001_drmeera",
      "customer_id": null,
      "send_as": "vera",
      "trigger_id": "trg_2026_04_26_research_digest",
      "template_name": "vera_research_digest_v1",
      "template_params": ["Dr. Meera", "JIDA Oct issue", "..."],
      "body": "Dr. Meera, JIDA's Oct issue landed...",
      "cta": "open_ended",
      "suppression_key": "research:dentists:2026-W17",
      "rationale": "External research digest with merchant-relevant clinical anchor; merchant is a dentist with high-risk-adult patient cohort"
    }
  ]
}
```

`actions` MAY be an empty list — the bot is free to decide nothing's worth sending this tick.

`conversation_id`:
- If you want to start a new conversation, generate any unique string.
- Reusing an existing `conversation_id` is invalid here — use `/v1/reply` to continue an existing conversation.

### 2.3 `POST /v1/reply` — receive a reply from the simulated merchant/customer

The judge calls this with the merchant's (or customer's) reply to a previous bot message. The bot must respond synchronously with its next move.

**Request body**:
```json
{
  "conversation_id": "conv_001",
  "merchant_id": "m_001_drmeera",
  "customer_id": null,
  "from_role": "merchant",
  "message": "Yes, send me the abstract",
  "received_at": "2026-04-26T10:45:00Z",
  "turn_number": 2
}
```

**Response (200)** — three valid `action` values:

```json
{ "action": "send",
  "body": "Sending now — also drafted a 90-sec patient-ed WhatsApp...",
  "cta": "open_ended",
  "rationale": "Honoring the merchant's accept; adding the next-best-step (patient-ed) as low-friction follow-on" }
```

```json
{ "action": "wait",
  "wait_seconds": 1800,
  "rationale": "Merchant asked for time; back off 30 min" }
```

```json
{ "action": "end",
  "rationale": "Merchant said not interested; gracefully exiting conversation" }
```

The bot has **30 seconds** to respond. After 30s the judge marks this turn as `timeout` and proceeds.

### 2.4 `GET /v1/healthz` — liveness probe

**Response (200)**:
```json
{ "status": "ok", "uptime_seconds": 3600, "contexts_loaded": { "category": 5, "merchant": 50, "customer": 200, "trigger": 100 } }
```

The judge polls this every 60s during the test window. Three consecutive failures = bot disqualified for that test slot.

### 2.5 `GET /v1/metadata` — bot identity

**Response (200)**:
```json
{
  "team_name": "Team Alpha",
  "team_members": ["Alice", "Bob"],
  "model": "claude-opus-4-7",
  "approach": "single-prompt composer with retrieval over digest items",
  "contact_email": "team@example.com",
  "version": "1.2.0",
  "submitted_at": "2026-04-26T08:00:00Z"
}
```

---

## 3. Context payload schemas (what the judge pushes to `/v1/context`)

Each `scope` has a fixed payload shape. These mirror the dataclasses defined in `challenge-brief.md` §4.

### 3.1 `scope: "category"`
```json
{
  "slug": "dentists",
  "offer_catalog": [{ "title": "Dental Cleaning @ ₹299", "value": "299", "audience": "new_user" }],
  "voice": { "tone": "peer_clinical", "vocab_allowed": ["fluoride varnish", "caries"], "taboos": ["cure", "guaranteed"] },
  "peer_stats": { "avg_rating": 4.4, "avg_reviews": 62, "avg_ctr": 0.030, "scope": "delhi_solo_practices" },
  "digest": [
    { "id": "d_2026W17_jida_fluoride", "kind": "research",
      "title": "3-mo fluoride recall cuts caries 38% better than 6-mo",
      "source": "JIDA Oct 2026, p.14", "trial_n": 2100, "patient_segment": "high_risk_adults",
      "summary": "..." }
  ],
  "patient_content_library": [
    { "id": "pc_001", "title": "3 things your teeth tell you about your heart", "channel": "whatsapp", "body": "..." }
  ],
  "seasonal_beats": [{ "month_range": "Nov-Feb", "note": "exam-stress bruxism spike" }],
  "trend_signals": [{ "query": "clear aligners delhi", "delta_yoy": 0.62, "segment_age": "28-45" }]
}
```

### 3.2 `scope: "merchant"`
```json
{
  "merchant_id": "m_001_drmeera",
  "category_slug": "dentists",
  "identity": { "name": "Dr. Meera's Dental Clinic", "city": "Delhi", "locality": "Lajpat Nagar",
                "place_id": "ChIJ...", "verified": true, "languages": ["en", "hi"] },
  "subscription": { "status": "active", "plan": "Pro", "days_remaining": 82 },
  "performance": {
    "window_days": 30,
    "views": 2410, "calls": 18, "directions": 45, "ctr": 0.021,
    "delta_7d": { "views_pct": 0.18, "calls_pct": -0.05 }
  },
  "offers": [
    { "id": "o_meera_001", "title": "Dental Cleaning @ ₹299", "status": "active" },
    { "id": "o_meera_002", "title": "Deep Cleaning @ ₹499", "status": "expired" }
  ],
  "conversation_history": [
    { "ts": "2026-04-24T10:00:00Z", "from": "vera", "body": "...", "engagement": "merchant_replied" }
  ],
  "customer_aggregate": { "total_unique_ytd": 540, "lapsed_180d_plus": 78, "retention_6mo_pct": 0.38 },
  "signals": ["stale_posts:22d", "ctr_below_peer_median", "high_risk_adult_cohort"]
}
```

### 3.3 `scope: "customer"`
```json
{
  "customer_id": "c_001_priya",
  "merchant_id": "m_001_drmeera",
  "identity": { "name": "Priya", "phone_redacted": "<phone>", "language_pref": "hi-en mix" },
  "relationship": {
    "first_visit": "2025-11-04", "last_visit": "2026-05-12", "visits_total": 4,
    "services_received": ["cleaning", "cleaning", "whitening", "cleaning"]
  },
  "state": "lapsed_soft",
  "preferences": { "preferred_slots": "weekday_evening", "channel": "whatsapp" },
  "consent": { "opted_in_at": "2025-11-04", "scope": ["recall_reminders", "appointment_reminders"] }
}
```

### 3.4 `scope: "trigger"`
```json
{
  "id": "trg_2026_04_26_research_digest_dentists",
  "scope": "merchant",
  "kind": "research_digest",
  "source": "external",
  "merchant_id": "m_001_drmeera",
  "customer_id": null,
  "payload": {
    "category": "dentists",
    "top_item_id": "d_2026W17_jida_fluoride"
  },
  "urgency": 2,
  "suppression_key": "research:dentists:2026-W17",
  "expires_at": "2026-05-03T00:00:00Z"
}
```

For `scope: "customer"` triggers (e.g., `recall_due`), `customer_id` is populated.

---

## 4. The judge harness behavior — full lifecycle

### Phase 1 — Warmup (T-15 min before scoring window opens)

1. Judge calls `GET /v1/healthz` and `GET /v1/metadata` to verify the bot is reachable.
2. Judge POSTs the **base dataset** to `/v1/context`:
    - 5 category contexts
    - 50 merchant contexts
    - 200 customer contexts
    - 0 triggers (triggers come during the test window)
3. Judge waits 60s for the bot to settle, then re-checks `/healthz`.
4. If `contexts_loaded` reflects all 255 base contexts, warmup passes.

### Phase 2 — Test window (T0 to T0 + 60 simulated minutes)

The judge advances simulated time in **5-minute ticks**. At each tick:

1. Judge POSTs any new/updated contexts that "happened" during this tick (incremental updates).
2. Judge calls `POST /v1/tick` with current simulated time + currently-active triggers.
3. Bot returns `actions[]` — zero or more proactive sends.
4. For each action:
    - Judge logs it (with rationale).
    - Judge invokes a sub-LLM playing the merchant/customer role.
    - The sub-LLM replies with a plausible response (sometimes engaged, sometimes auto-reply, sometimes hard "no", sometimes a question).
    - Judge POSTs that reply to `/v1/reply`.
    - Bot returns next action — `send`, `wait`, or `end`.
    - Repeat up to 5 turns or until bot ends.

### Phase 3 — Adaptive context injection (interleaved during Phase 2)

Between ticks, the judge injects **post-submission context** the bot didn't see during development:

- New `digest` items (5 new research/compliance items per category context, posted as a new `version`)
- Updated `performance` snapshots (10 merchants get new perf numbers — some spikes, some dips)
- New `triggers` (15 new triggers spread across the test window)
- For 5 specific merchants: a new `customer` context is pushed mid-test with a `recall_due` trigger 2 minutes later

Bots that incorporate the new context in subsequent sends score higher. Bots that ignore it (sending stale composition) score lower. Bots that hallucinate (invent context that wasn't pushed) score lowest.

### Phase 4 — Replay test (top 10 only)

For the 10 highest-scoring bots from Phase 2, the judge runs **3 deep-dive scenarios** as standalone replays:

1. **Auto-reply hell** — judge plays a merchant whose phone is on WA Business with a canned auto-reply; sends the same canned text 4 times in a row. Bot must detect and exit gracefully.
2. **Intent transition** — judge plays an engaged merchant who, after 2 turns of qualification, says "ok let's do it". Bot must switch from qualifying to action immediately (not ask another qualifying question).
3. **Hostile / off-topic** — judge plays a merchant who replies with abuse, then asks an unrelated question ("can you also help me file my GST?"). Bot must stay on-mission politely.

Each replay: 5 turns, scored on conversation flow only.

### Phase 5 — Scoring + report (T0 + 90 min)

Judge aggregates:
- Phase 2 scores (per the 5-dimension rubric in §8 of the main brief)
- Phase 3 adaptation bonus (max +5 per dimension)
- Phase 4 replay scores (top 10 only; max +30)
- Operational penalties (timeouts, healthz failures, malformed responses; max -20)

Sends each team a per-message scorecard with judge rationale.

---

## 5. Rate limits + timeouts

| Limit | Value |
|---|---|
| Max requests/sec from judge to bot | 10 |
| Per-call timeout (judge waits this long) | 30 s |
| `/v1/context` payload size cap | 500 KB |
| `/v1/tick` action count cap | 20 actions per tick |
| Healthz failures before disqualification | 3 consecutive |
| Total test window | 60 simulated minutes (real-time ~30-45 min) |

If your bot needs more than 30s for `/v1/tick`, return an empty `actions: []` immediately and process work asynchronously — but you can't catch up later, so design for the budget.

---

## 6. Where to deploy

Deploy your bot anywhere that gives you a **public URL**:
- Any cloud (AWS, GCP, Azure, Render, Fly, Railway, Replit, …)
- ngrok tunnel to localhost
- Any hosting that exposes HTTP endpoints

Requirements:
- Must respond at the URL pattern `https://<your-host>/v1/*` (or `http://` for local testing)
- Submit your public URL via the submission portal

---

## 7. Reference implementation skeleton

A minimal-viable bot in ~80 lines of Python (FastAPI). Save as `bot.py`:

```python
import os, time
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Any

app = FastAPI()
START = time.time()

# In-memory stores (use Redis/SQLite for production-grade)
contexts: dict[tuple[str, str], dict] = {}    # (scope, context_id) -> {version, payload}
conversations: dict[str, list] = {}           # conversation_id -> [turns]


@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in contexts.items():
        counts[scope] = counts.get(scope, 0) + 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    return {"team_name": "Team Alpha", "team_members": ["Alice"], "model": "gpt-4o-mini",
            "approach": "single-prompt composer", "contact_email": "alice@example.com",
            "version": "0.1.0", "submitted_at": "2026-04-26T08:00:00Z"}


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str

@app.post("/v1/context")
async def push_context(body: CtxBody):
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": datetime.utcnow().isoformat() + "Z"}


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []

@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    for trg_id in body.available_triggers:
        trg = contexts.get(("trigger", trg_id), {}).get("payload")
        if not trg: continue
        merchant_id = trg.get("merchant_id")
        merchant = contexts.get(("merchant", merchant_id), {}).get("payload")
        category = contexts.get(("category", merchant.get("category_slug")), {}).get("payload") if merchant else None
        if not (merchant and category): continue
        # YOUR COMPOSER GOES HERE — call your LLM with the 4 contexts
        body_text = f"Hi {merchant['identity']['name']}, ..."  # replace with real composition
        actions.append({
            "conversation_id": f"conv_{merchant_id}_{trg_id}",
            "merchant_id": merchant_id, "customer_id": None,
            "send_as": "vera", "trigger_id": trg_id,
            "template_name": "vera_generic_v1",
            "template_params": [merchant['identity']['name'], "...", "..."],
            "body": body_text, "cta": "open_ended",
            "suppression_key": trg.get("suppression_key", ""),
            "rationale": "Composed from category+merchant+trigger"
        })
    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conversations.setdefault(body.conversation_id, []).append({"from": body.from_role, "msg": body.message})
    # YOUR REPLY-COMPOSER GOES HERE
    return {"action": "send", "body": "Got it, here's what's next...", "cta": "open_ended",
            "rationale": "acknowledged + advanced"}
```

Run: `uvicorn bot:app --host 0.0.0.0 --port 8080`

This is a working skeleton. The composer logic is stubbed — replace the `# YOUR COMPOSER GOES HERE` blocks with your LLM call.

---

## 9. Local self-test before submitting

Magicpin provides a `judge_simulator.py` that runs a mini version of the harness against your endpoint. Use it during development:

```bash
export BOT_URL=http://localhost:8080
python judge_simulator.py
```

Each scenario prints the judge's prompts + your bot's responses + a mock score. Iterate until you're happy, then submit your URL.

---

## 10. Failure modes the judge handles

| Failure | Judge behavior | Penalty |
|---|---|---|
| `/v1/healthz` returns non-200 (3× in a row) | Mark bot offline; skip remaining ticks | -10 (operational) |
| `/v1/tick` times out (>30s) | Skip this tick's actions; continue | -1 per timeout |
| `/v1/reply` times out | Mark turn as `bot_silent`; judge plays next merchant turn after 30s | -1 per timeout |
| Bot returns malformed JSON | Logged, scored as 0 for that action | -2 per malformed |
| Bot returns `action: send` with empty body | Treated as malformed | -2 |
| Bot returns the same body verbatim it sent before in the same conversation | Anti-repetition flag | -2 per repeat |

---

## 11. Security + privacy

- All payloads are synthetic — no real PII.
- Bots **must not** transmit any payload data outside the test environment (no calls to non-LLM external APIs with merchant/customer fields).
- Bots **may** use commercial LLM APIs (OpenAI, Anthropic, Google, DeepSeek, etc.) — those are necessary for composition.
- Bots **must not** persist context data after the test ends. magicpin will issue a `POST /v1/teardown` (optional) at end of test; on receiving it, wipe state.

---

## 12. Pre-flight checklist for candidates

Before submitting:

- [ ] Endpoint reachable from the public internet (HTTPS or HTTP)
- [ ] All 5 endpoints implemented and returning correct schemas
- [ ] `/v1/context` is idempotent on `(scope, context_id, version)`
- [ ] `/v1/tick` returns within 30s even if it has nothing to send (returns `{"actions": []}`)
- [ ] `/v1/reply` returns within 30s for any conversation
- [ ] Bot persists context across calls (in-memory is fine; no restarts during test)
- [ ] `judge_simulator.py` passes locally with non-zero scores
- [ ] Submitted URL via submission portal
- [ ] Compute budget set (rate limits, LLM API quota, etc.) so the bot survives 60-min test

---

## 13. What the judge logs (for transparency)

Every test produces a per-team artifact:

```
results/<team_name>/
├── conversations.jsonl      # all turns, both sides, with timestamps
├── context_pushes.jsonl     # every context push, with bot's ack
├── scoring.json             # 5-dimension scores per action + per conversation
├── timeline.html            # visual timeline of the test window
├── replay_*.jsonl           # phase 4 replay transcripts (top 10 only)
└── final_report.md          # aggregated score + judge's qualitative feedback
```

Candidates receive their own artifact bundle within 48h of the test. Top scorers' bundles (with consent) become reference material for the next cohort.

---

## 14. FAQ

**Q: Can the bot use external tools / function calling during composition?**
Yes. Your LLM can call any tool you implement internally. You can't call out to non-LLM external APIs that receive merchant/customer payloads (privacy rule §11).

**Q: What if my bot needs more than 30s to compose a really good message?**
Two options: (a) speed it up; (b) at `/v1/tick`, return immediately with `{"actions": []}` and skip the cycle. Don't try to background-process and return late — late responses are dropped.

**Q: Can I send multiple messages in one tick to the same merchant?**
Yes, but only one `action` per `(merchant_id, conversation_id)` pair per tick. Use a follow-up tick to send more.

**Q: Does the judge see my bot's `rationale` field?**
Yes — it's included in the scoring rubric ("did the rationale match the actual output?"). High-quality rationales help the judge interpret edge cases generously.

**Q: What language do replies have to be in?**
Match the merchant's `identity.languages` field. Default is English. Hindi-English code-mix is encouraged where the language pref says `hi`.

**Q: Can my bot refuse to send when nothing's worth saying?**
Yes — return `{"actions": []}` from `/v1/tick`. Restraint is rewarded; spam is penalized.

**Q: What if the judge pushes a context for a merchant I've never seen before, mid-conversation?**
Treat it as a normal new merchant. The bot should be ready for any context to arrive at any time.

**Q: Is there a way to query the judge for clarification mid-test?**
No. The bot has only the contexts it's been pushed. This is intentional — production Vera doesn't get clarifications either.

---

## End of testing brief

The two briefs together (`challenge-brief.md` for *what to build* and this one for *how it's tested*) are the complete spec. A team should be able to read both end-to-end in 30 minutes and start coding.
