# API Call Examples — Judge ↔ Candidate Bot

This file shows the exact HTTP calls the judge will make during testing, and what the bot is expected to return. Read this together with `challenge-testing-brief.md` (which defines the contract) and the dataset (which provides the payloads).

Every example uses Dr. Meera's Dental Clinic (`m_001_drmeera_dentist_delhi`) as the running merchant.

---

## Phase 1 — Warmup (T-15 min)

### Example 1.1 — `GET /v1/healthz`

**Request**
```http
GET /v1/healthz HTTP/1.1
Host: bot.candidate-team-alpha.example.com
Accept: application/json
```

**Expected response (200)**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "status": "ok",
  "uptime_seconds": 124,
  "contexts_loaded": { "category": 0, "merchant": 0, "customer": 0, "trigger": 0 }
}
```

The judge calls this before pushing context. `contexts_loaded` should be all zeros at this point (bot just started).

### Example 1.2 — `GET /v1/metadata`

**Request**
```http
GET /v1/metadata HTTP/1.1
Host: bot.candidate-team-alpha.example.com
```

**Expected response (200)**
```json
{
  "team_name": "Team Alpha",
  "team_members": ["Alice", "Bob"],
  "model": "claude-opus-4-7",
  "approach": "single-prompt composer with retrieval over digest items + dispatch by trigger.kind",
  "contact_email": "team@example.com",
  "version": "1.2.0",
  "submitted_at": "2026-04-26T08:00:00Z"
}
```

### Example 1.3 — `POST /v1/context` (push CategoryContext)

**Request**
```http
POST /v1/context HTTP/1.1
Host: bot.candidate-team-alpha.example.com
Content-Type: application/json

{
  "scope": "category",
  "context_id": "dentists",
  "version": 1,
  "delivered_at": "2026-04-26T09:45:00Z",
  "payload": {
    "slug": "dentists",
    "voice": { "tone": "peer_clinical", "vocab_taboo": ["guaranteed", "100% safe"] },
    "offer_catalog": [
      { "id": "den_001", "title": "Dental Cleaning @ ₹299", "value": "299", "audience": "new_user", "type": "service_at_price" }
    ],
    "peer_stats": { "avg_rating": 4.4, "avg_ctr": 0.030 },
    "digest": [{ "id": "d_2026W17_jida_fluoride", "kind": "research", "title": "3-month fluoride recall cuts caries 38% better", "source": "JIDA Oct 2026, p.14" }],
    "patient_content_library": [],
    "seasonal_beats": [{ "month_range": "Nov-Feb", "note": "exam-stress bruxism spike" }],
    "trend_signals": [{ "query": "clear aligners delhi", "delta_yoy": 0.62 }]
  }
}
```

**Expected response (200)**
```json
{ "accepted": true, "ack_id": "ack_dentists_v1", "stored_at": "2026-04-26T09:45:00.123Z" }
```

> **Note**: For the actual test the full category JSON (`dataset/categories/dentists.json`) goes in `payload`, not the abbreviated form above.

### Example 1.4 — `POST /v1/context` (push MerchantContext)

**Request**
```http
POST /v1/context HTTP/1.1
Content-Type: application/json

{
  "scope": "merchant",
  "context_id": "m_001_drmeera_dentist_delhi",
  "version": 1,
  "delivered_at": "2026-04-26T09:45:30Z",
  "payload": {
    "merchant_id": "m_001_drmeera_dentist_delhi",
    "category_slug": "dentists",
    "identity": { "name": "Dr. Meera's Dental Clinic", "city": "Delhi", "locality": "Lajpat Nagar",
                  "verified": true, "languages": ["en", "hi"], "owner_first_name": "Meera" },
    "subscription": { "status": "active", "plan": "Pro", "days_remaining": 82 },
    "performance": { "window_days": 30, "views": 2410, "calls": 18, "directions": 45,
                     "ctr": 0.021, "delta_7d": { "views_pct": 0.18, "calls_pct": -0.05 } },
    "offers": [{ "id": "o_meera_001", "title": "Dental Cleaning @ ₹299", "status": "active" }],
    "conversation_history": [],
    "customer_aggregate": { "total_unique_ytd": 540, "lapsed_180d_plus": 78,
                            "retention_6mo_pct": 0.38, "high_risk_adult_count": 124 },
    "signals": ["stale_posts:22d", "ctr_below_peer_median", "high_risk_adult_cohort"]
  }
}
```

**Expected response (200)**
```json
{ "accepted": true, "ack_id": "ack_m_001_drmeera_v1", "stored_at": "2026-04-26T09:45:30.456Z" }
```

### Example 1.5 — `POST /v1/context` (idempotency check — same version re-pushed)

**Request** (same body as 1.4 — version 1 again)

**Expected response (409)**
```json
{ "accepted": false, "reason": "stale_version", "current_version": 1 }
```

### Example 1.6 — `POST /v1/context` (version bump replaces)

**Request**: same as 1.4 but `version: 2` and `performance.views: 2580` (updated).

**Expected response (200)**
```json
{ "accepted": true, "ack_id": "ack_m_001_drmeera_v2", "stored_at": "2026-04-26T10:30:00.789Z" }
```

The bot must now use the new version when composing for `m_001_drmeera_dentist_delhi`.

### Example 1.7 — `GET /v1/healthz` after warmup complete

**Expected response (200)**
```json
{
  "status": "ok",
  "uptime_seconds": 1024,
  "contexts_loaded": { "category": 5, "merchant": 50, "customer": 200, "trigger": 0 }
}
```

If counts don't match what the judge pushed, warmup fails and the bot is disqualified for that test slot.

---

## Phase 2 — Test window (T0 → T0 + 60 min)

### Example 2.1 — `POST /v1/context` (incremental trigger push)

The judge now starts pushing triggers as simulated time advances.

**Request**
```http
POST /v1/context HTTP/1.1
Content-Type: application/json

{
  "scope": "trigger",
  "context_id": "trg_001_research_digest_dentists",
  "version": 1,
  "delivered_at": "2026-04-26T10:32:00Z",
  "payload": {
    "id": "trg_001_research_digest_dentists",
    "scope": "merchant",
    "kind": "research_digest",
    "source": "external",
    "merchant_id": "m_001_drmeera_dentist_delhi",
    "customer_id": null,
    "payload": {
      "category": "dentists",
      "top_item_id": "d_2026W17_jida_fluoride"
    },
    "urgency": 2,
    "suppression_key": "research:dentists:2026-W17",
    "expires_at": "2026-05-03T00:00:00Z"
  }
}
```

**Expected response (200)**
```json
{ "accepted": true, "ack_id": "ack_trg_001_v1", "stored_at": "2026-04-26T10:32:00.150Z" }
```

### Example 2.2 — `POST /v1/tick` (bot decides to send)

**Request**
```http
POST /v1/tick HTTP/1.1
Content-Type: application/json

{
  "now": "2026-04-26T10:35:00Z",
  "available_triggers": ["trg_001_research_digest_dentists"]
}
```

**Expected response (200) — bot chose to send**
```json
{
  "actions": [
    {
      "conversation_id": "conv_m_001_drmeera_research_W17",
      "merchant_id": "m_001_drmeera_dentist_delhi",
      "customer_id": null,
      "send_as": "vera",
      "trigger_id": "trg_001_research_digest_dentists",
      "template_name": "vera_research_digest_v1",
      "template_params": [
        "Dr. Meera",
        "JIDA Oct issue landed. One item relevant to your high-risk adult patients — 2,100-patient trial showed 3-month fluoride recall cuts caries recurrence 38% better than 6-month",
        "Worth a look (2-min abstract). Want me to pull it + draft a patient-ed WhatsApp you can share?"
      ],
      "body": "Dr. Meera, JIDA's Oct issue landed. One item relevant to your high-risk adult patients — 2,100-patient trial showed 3-month fluoride recall cuts caries recurrence 38% better than 6-month. Worth a look (2-min abstract). Want me to pull it + draft a patient-ed WhatsApp you can share? — JIDA Oct 2026 p.14",
      "cta": "open_ended",
      "suppression_key": "research:dentists:2026-W17",
      "rationale": "External research digest with merchant-relevant clinical anchor (high-risk-adult cohort matches signal). Source citation at end maintains credibility. Open-ended CTA invites continuation without forcing a binary choice."
    }
  ]
}
```

### Example 2.3 — `POST /v1/tick` (bot decides NOT to send)

If no triggers are worth acting on this tick (or all have been suppressed):

**Expected response (200)**
```json
{ "actions": [] }
```

Empty `actions` is valid. Restraint is rewarded; spam is penalized.

### Example 2.4 — `POST /v1/reply` (judge plays merchant — engaged response)

The judge's sub-LLM, playing Dr. Meera, replies to the message from 2.2:

**Request**
```http
POST /v1/reply HTTP/1.1
Content-Type: application/json

{
  "conversation_id": "conv_m_001_drmeera_research_W17",
  "merchant_id": "m_001_drmeera_dentist_delhi",
  "customer_id": null,
  "from_role": "merchant",
  "message": "Yes please send the abstract. Also draft the patient WhatsApp.",
  "received_at": "2026-04-26T10:42:00Z",
  "turn_number": 2
}
```

**Expected response (200) — bot follows up with action**
```json
{
  "action": "send",
  "body": "Sending the abstract now (PDF, 2 pages). Patient-ed draft below — you can copy-paste or I'll schedule a Google post:\n\n\"3-month vs 6-month dental cleaning — does it really matter? New research shows yes, especially if you've had cavities recently. Drop us a note for a quick check.\"\n\nWant me to schedule the post for tomorrow 10am?",
  "cta": "binary_yes_no",
  "rationale": "Honoring both asks (abstract + draft) in one turn. Draft is at patient-reading level. Final question is a binary yes/no to lower friction."
}
```

### Example 2.5 — `POST /v1/reply` (judge plays merchant — auto-reply)

Some test scenarios simulate WhatsApp Business auto-reply:

**Request**
```json
{
  "conversation_id": "conv_m_001_drmeera_research_W17",
  "from_role": "merchant",
  "message": "Thank you for contacting Dr. Meera's Dental Clinic! Our team will respond shortly.",
  "received_at": "2026-04-26T10:42:00Z",
  "turn_number": 2
}
```

**Expected response (200) — bot detects auto-reply, exits gracefully**
```json
{
  "action": "wait",
  "wait_seconds": 14400,
  "rationale": "Detected merchant auto-reply (canned 'Thank you for contacting' phrasing). Backing off 4 hours to wait for owner."
}
```

After the wait expires, the judge will call `/v1/tick` again and the bot can re-engage.

### Example 2.6 — `POST /v1/reply` (judge plays merchant — hard "no")

**Request**
```json
{
  "conversation_id": "conv_m_001_drmeera_research_W17",
  "from_role": "merchant",
  "message": "Not interested. Stop messaging me.",
  "received_at": "2026-04-26T10:42:00Z",
  "turn_number": 2
}
```

**Expected response (200) — bot ends gracefully**
```json
{
  "action": "end",
  "rationale": "Merchant explicitly opted out. Closing conversation; suppressing this conversation_id for future ticks."
}
```

The bot must not send any further messages on this `conversation_id`.

### Example 2.7 — `POST /v1/reply` (judge plays merchant — curveball)

**Request**
```json
{
  "conversation_id": "conv_m_001_drmeera_research_W17",
  "from_role": "merchant",
  "message": "Btw can you also help me with my GST filing this month?",
  "received_at": "2026-04-26T10:42:00Z",
  "turn_number": 2
}
```

**Expected response (200) — bot stays on mission politely**
```json
{
  "action": "send",
  "body": "I'll have to leave GST filing to your CA — that's outside what I can help with directly. Coming back to the JIDA piece — want me to draft the patient post first, or send the abstract?",
  "cta": "open_ended",
  "rationale": "Out-of-scope ask politely declined; redirects back to the original trigger without losing thread."
}
```

### Example 2.8 — `POST /v1/context` (mid-test adaptive injection)

While the conversation is live, the judge pushes a new digest item. A good bot will incorporate it in subsequent sends.

**Request**
```json
{
  "scope": "category",
  "context_id": "dentists",
  "version": 2,
  "delivered_at": "2026-04-26T10:50:00Z",
  "payload": {
    "slug": "dentists",
    "voice": { "tone": "peer_clinical" },
    "digest": [
      { "id": "d_2026W17_jida_fluoride", "kind": "research", "title": "3-month fluoride recall cuts caries 38% better", "source": "JIDA Oct 2026, p.14" },
      { "id": "d_2026W17_dci_radiograph_NEW", "kind": "compliance", "title": "DCI revised radiograph dose limits effective 2026-12-15",
        "source": "DCI circular 2026-11-04", "summary": "Max dose drops 1.5→1.0 mSv per IOPA. E-speed film passes; D-speed does not." }
    ],
    "// other fields": "..."
  }
}
```

**Expected response (200)**
```json
{ "accepted": true, "ack_id": "ack_dentists_v2", "stored_at": "2026-04-26T10:50:00.110Z" }
```

The bot must replace the old version atomically and use the new digest item if relevant in the next send.

### Example 2.9 — `POST /v1/tick` (customer-scoped trigger emerges)

A `recall_due` trigger fires for one of Dr. Meera's patients:

**Context push first**
```json
{
  "scope": "customer",
  "context_id": "c_001_priya_for_m001",
  "version": 1,
  "payload": { /* Priya's CustomerContext from dataset/customers_seed.json */ }
}
```

```json
{
  "scope": "trigger",
  "context_id": "trg_003_recall_due_priya",
  "version": 1,
  "payload": { /* the recall trigger from dataset/triggers_seed.json */ }
}
```

**Then `/v1/tick`**
```json
{
  "now": "2026-04-26T11:00:00Z",
  "available_triggers": ["trg_003_recall_due_priya"]
}
```

**Expected response (200)**
```json
{
  "actions": [
    {
      "conversation_id": "conv_priya_recall_2026_11",
      "merchant_id": "m_001_drmeera_dentist_delhi",
      "customer_id": "c_001_priya_for_m001",
      "send_as": "merchant_on_behalf",
      "trigger_id": "trg_003_recall_due_priya",
      "template_name": "merchant_recall_reminder_v1",
      "template_params": [
        "Priya",
        "Dr. Meera's clinic",
        "It's been 5 months since your last visit",
        "Wed 5 Nov, 6pm or Thu 6 Nov, 5pm",
        "₹299 cleaning + complimentary fluoride"
      ],
      "body": "Hi Priya, Dr. Meera's clinic here 🦷 It's been 5 months since your last visit — your 6-month cleaning recall is due. Apke liye 2 slots ready hain: **Wed 5 Nov, 6pm** ya **Thu 6 Nov, 5pm**. ₹299 cleaning + complimentary fluoride. Reply 1 for Wed, 2 for Thu, or tell us a time that works.",
      "cta": "multi_choice_slot",
      "suppression_key": "recall:c_001_priya_for_m001:6mo",
      "rationale": "Customer-scoped recall, sending via merchant's number (send_as=merchant_on_behalf). Honoring Priya's hi-en mix language pref + weekday-evening preference (both slots offered are weekday evenings). Multi-choice slot CTA is appropriate for booking flows."
    }
  ]
}
```

---

## Phase 4 — Replay test (top 10 only)

The judge runs 3 standalone scenarios. Each is a fresh conversation with a controlled merchant persona.

### Example 4.1 — Auto-reply hell scenario

The judge sends 4 turns of identical canned auto-replies.

**Turn 1 — bot initiates**
```json
POST /v1/tick { "now": "...", "available_triggers": ["trg_022_cde_webinar_dentists"] }
→ { "actions": [{ /* CDE webinar invite */ }] }
```

**Turn 2 — judge plays auto-reply**
```json
POST /v1/reply { "from_role": "merchant",
                 "message": "Thank you for contacting Dr. Meera's Dental Clinic! Our team will respond shortly.",
                 "turn_number": 2 }
```

**Good bot response**
```json
{ "action": "send",
  "body": "Looks like an auto-reply 😊 When the owner sees this, just reply 'Yes' for the webinar invite.",
  "cta": "binary_yes_no",
  "rationale": "Detected auto-reply; one explicit prompt to flag it for the owner." }
```

**Turn 3 — judge plays same auto-reply again**
```json
POST /v1/reply { "from_role": "merchant",
                 "message": "Thank you for contacting Dr. Meera's Dental Clinic! Our team will respond shortly.",
                 "turn_number": 3 }
```

**Good bot response**
```json
{ "action": "wait", "wait_seconds": 86400,
  "rationale": "Same auto-reply twice in a row → owner not at phone. Wait 24h before retry." }
```

**Turn 4 — judge plays same auto-reply yet again**

**Good bot response**
```json
{ "action": "end",
  "rationale": "Auto-reply 3x in a row, no real reply. Conversation has zero engagement signal; closing." }
```

### Example 4.2 — Intent transition scenario

After 2 turns of qualification, judge plays an explicit "let's do it":

**Turn 3 — judge**
```json
POST /v1/reply { "from_role": "merchant",
                 "message": "Ok, let's do it. What's next?",
                 "turn_number": 3 }
```

**Good bot response — switches from qualifying to action**
```json
{ "action": "send",
  "body": "Great. Drafting your patient WhatsApp now — 90 seconds. I'll also pre-fill the GBP post for tomorrow 10am. Reply CONFIRM to send the WhatsApp draft to your patient list (40 high-risk adult patients).",
  "cta": "binary_confirm_cancel",
  "rationale": "Merchant explicitly committed; switching from question-asking to action-execution. Concrete next step + measurable scope (40 patients)." }
```

**Bad bot response — keeps qualifying (this loses points)**
```json
{ "action": "send",
  "body": "Excellent! Just to plan well — would you say most of your high-risk patients are diabetic, or is there another common factor?",
  "rationale": "Continuing qualification..."  // ← penalty: ignored explicit intent transition
}
```

### Example 4.3 — Hostile / off-topic scenario

**Turn 2 — judge plays hostile**
```json
POST /v1/reply { "from_role": "merchant",
                 "message": "Why are you bothering me. This is useless. Stop sending these.",
                 "turn_number": 2 }
```

**Good bot response — graceful exit**
```json
{ "action": "end",
  "rationale": "Merchant frustration explicit; closing without further engagement. Suppressing all triggers for this merchant for 30 days." }
```

**Acceptable alternative — short apology + exit**
```json
{ "action": "send",
  "body": "Apologies — I won't message again. If anything changes, you can always restart with 'Hi Vera'. 🙏",
  "cta": "none",
  "rationale": "One-line acknowledgment + opt-out path; conversation will close after this send." }
```

---

## Failure-mode examples

### Example F.1 — Bot times out

If `/v1/tick` doesn't respond within 30s, the judge logs a timeout and continues. No retries.

### Example F.2 — Malformed response

```json
{ "actions": [{ "merchant_id": "m_001", "body": "..." }] }
```

Missing required fields (`conversation_id`, `send_as`, `trigger_id`, `cta`, `suppression_key`, `rationale`) → action scored as 0, -2 penalty.

### Example F.3 — Body too long

```json
{ "body": "...500 chars..." }
```

No hard body-length cap. Messages are judged on quality, specificity, and relevance.

### Example F.4 — URL in body

```json
{ "body": "Read more: https://magicpin.com/blog" }
```

Hard fail for that action — Meta would reject. Penalty: -3 per URL.

### Example F.5 — Repetition

Same `body` text sent twice in the same `conversation_id` → -2 anti-repetition penalty per repeat.

---

## Curl examples (for local testing)

```bash
# Set your bot URL
export BOT_URL=http://localhost:8080

# Healthz
curl $BOT_URL/v1/healthz

# Push a category context
curl -X POST -H "Content-Type: application/json" \
  -d @dataset/categories/dentists.json \
  $BOT_URL/v1/context

# Trigger a tick
curl -X POST -H "Content-Type: application/json" \
  -d '{"now": "2026-04-26T10:35:00Z", "available_triggers": ["trg_001_research_digest_dentists"]}' \
  $BOT_URL/v1/tick

# Send a reply
curl -X POST -H "Content-Type: application/json" \
  -d '{"conversation_id": "conv_001", "merchant_id": "m_001_drmeera_dentist_delhi", "from_role": "merchant", "message": "Yes please send the abstract", "received_at": "2026-04-26T10:42:00Z", "turn_number": 2}' \
  $BOT_URL/v1/reply
```

---

## Summary table — request shapes at a glance

| Endpoint | Method | Body | Latency budget | Retried? |
|---|---|---|---|---|
| `/v1/healthz` | GET | none | 2 s | yes (×3) |
| `/v1/metadata` | GET | none | 2 s | no |
| `/v1/context` | POST | full payload | 5 s | no |
| `/v1/tick` | POST | `{now, available_triggers}` | 10 s | no |
| `/v1/reply` | POST | reply turn | 10 s | no |

That's the full surface. If your bot handles every example here correctly, it'll pass the warmup, the test window, and the replay scenarios with no operational issues — leaving the score entirely to the quality of your composition.
