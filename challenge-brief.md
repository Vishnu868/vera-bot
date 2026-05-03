# magicpin AI Challenge — Build a Merchant AI Assistant ("Vera")

**Status**: Brief — designed to be loaded as standalone context in a fresh AI session.
**Last updated**: 2026-04-26
**Audience**: Challenge participants + the AI judge that will evaluate submissions.

> **Read me first** — this document is fully self-contained. A reader (human or LLM) starting cold should be able to understand the company, the product, the challenge, the framework, the dataset, and the evaluation rubric without needing any other reference.

---

## 1. The challenge in one sentence

> Build an AI chatbot that engages and assists merchants on WhatsApp the way **Vera** (magicpin's merchant-AI assistant) does — but better. Same base dataset for every participant. AI judges the outcome.

---

## 2. About magicpin (background, ~30 sec read)

magicpin is one of India's largest local-commerce platforms — a network of ~100,000 merchant partners across 50+ Indian cities (restaurants, salons, gyms, dentists, retailers, car dealers, etc.). Customers discover merchants on the magicpin app/web, transact, and earn cashback. Merchants benefit from visibility, walk-ins, and online orders.

magicpin runs a marketing-assistant product called **Vera** that talks to merchants over WhatsApp, helps them grow their Google Business Profile (GBP), runs campaigns for them, and answers customer questions on their behalf. Vera is the product you're rebuilding.

---

## 3. About Vera (the product you're rebuilding)

### What Vera does today
- Talks to ~6,000-10,000 merchants per day over WhatsApp
- Helps them improve their Google Business Profile (photos, hours, posts, reviews)
- Runs marketing campaigns for them (festival, news, local events)
- Recommends pricing, offers, and content
- Answers customers' questions about merchants on their behalf

### Live engagement shape (real numbers — Apr 23-25, 2026)

| Metric | Apr 23 | Apr 24 | Apr 25 |
|---|---:|---:|---:|
| Merchants who engaged with Vera | 6,917 | 10,158 | 5,181 |
| Total messages mx ↔ Vera | 34,022 | 47,651 | 23,712 |
| Avg messages per engaged merchant | 4.9 | 4.7 | 4.6 |
| Customers who engaged with Vera | 720 | 707 | 681 |
| Unique merchants customers asked about | 580 | 514 | 466 |
| Total messages cx ↔ Vera | 3,713 | 4,839 | 4,695 |
| Avg messages per customer | 5.2 | 6.8 | 6.9 |

### Today's biggest pain points (your opportunities to outperform Vera)

1. **Auto-reply pollution**: 40-70% of "merchant replies" are the merchant's own WhatsApp Business canned auto-reply ("Thank you for contacting…"). Production Vera detects this but burns 2-3 turns each time. **Better detection + faster routing wins**.
2. **Intent-handoff failures**: When a merchant says "I want to join," production Vera often goes back to qualifying questions instead of starting the action. **Better intent detection wins**.
3. **Generic copy**: discount-style offers ("10% off") rarely engage Indian merchants. Service+price ("Haircut @ ₹99", "Dental Cleaning @ ₹299") works better. **Compelling, category-correct copy wins**.
4. **Low engagement frequency**: Functional nudges (renewal due, profile incomplete) are inherently rare. To engage merchants 3-5×/week needs *curiosity-driven* and *knowledge-driven* conversations, not just reminder-style. **Diversified conversation portfolio wins**.

---

## 4. The 4-context framework (what your bot must work with)

Every message Vera sends — whether to the merchant or the merchant's customer — is composed from **four context layers**. Your bot will receive these as structured input and must produce the next message.

```
your_bot.compose(category, merchant, trigger, customer?) → message
```

### 4.1 CategoryContext

Slow-changing knowledge pack about *the kind of business this is*. Shared across all merchants in the vertical.

```python
@dataclass
class CategoryContext:
    slug: str                          # "dentists", "salons", "restaurants", ...
    offer_catalog: list[OfferTemplate] # canonical service+price patterns for this vertical
    voice: VoiceProfile                # tone, allowed vocabulary, taboos
    peer_stats: PeerStats              # benchmarks: avg rating, avg reviews, typical CTR
    digest: list[DigestItem]           # weekly research / compliance / trend items, source-cited
    patient_content_library: list[ContentItem]  # content the merchant can reshare with customers
    seasonal_beats: list[SeasonalBeat] # e.g., "exam-stress bruxism spikes Nov-Feb"
    trend_signals: list[TrendSignal]   # e.g., "clear aligners Delhi searches +62% YoY"
```

**Example for `dentists`**:
- voice: technical terms welcome ("fluoride varnish at 3-month recall"), legal taboos ("cure", "guaranteed"), peer tone not hype
- offer_catalog: `["Dental Cleaning @ ₹299", "Free Consultation", "Teeth Whitening @ ₹1,499"]`
- peer_stats: `{avg_rating: 4.4, avg_reviews: 62, avg_ctr: 0.030}` (South Delhi solo practices)
- digest: this week's items from JIDA / DCI / Dental Tribune India / IDA Delhi calendar
- seasonal_beats: `[{month: "Nov-Feb", note: "exam-stress bruxism spike"}, {month: "Oct-Dec", note: "wedding whitening peak"}]`

### 4.2 MerchantContext

The specific business's current state. Refreshed daily for performance, real-time for conversation history.

```python
@dataclass
class MerchantContext:
    merchant_id: str
    identity: Identity                # name, place_id, locality, city, verified, languages
    subscription: Subscription        # status, days_remaining, plan
    performance: PerformanceSnapshot  # views/calls/CTR/leads/directions, 30d + 7d deltas
    offers: list[MerchantOffer]       # active + paused offers from their catalog
    conversation_history: ConversationHistory  # last N turns w/ Vera, w/ engagement tags
    customer_aggregate: CustomerAggregate  # active count, lapsed count, retention %
    signals: list[DerivedSignal]      # "stale_posts", "ctr_below_peer", "dormant", ...
```

**Running example — Dr. Meera's Dental Clinic**:
- identity: name="Dr. Meera's Dental Clinic", city="Delhi", locality="Lajpat Nagar", verified=true
- subscription: status="active", days_remaining=82, plan="Pro"
- performance (30d): views=2410, calls=18, directions=45, ctr=0.021
- offers: `["Dental Cleaning @ ₹299"]` active; `["Deep Cleaning @ ₹499"]` expired 2mo ago
- customer_aggregate: 540 unique patients YTD, 78 lapsed >180 days, 38% 6mo retention
- signals: `["stale_posts" (last post 22d ago), "ctr_below_peer_median" (peer median 0.030)]`

### 4.3 TriggerContext

The *event* that prompts this message right now. Every message must have one.

```python
@dataclass
class TriggerContext:
    id: str
    scope: Literal["merchant", "customer"]
    kind: str                # "research_digest", "recall_due", "perf_spike", ...
    source: Literal["external", "internal"]
    payload: dict
    urgency: int             # 1-5
    suppression_key: str     # for dedup
    expires_at: datetime
```

**External triggers** (happen outside the merchant's account):
- `festival_upcoming` (Diwali in 4 days)
- `weather_heatwave` (42°C today in Delhi)
- `local_news_event` (Mumbai-Pune expressway closed 3h)
- `category_research_digest_release` (JIDA Oct issue dropped)
- `regulation_change` (DCI radiograph dose limit revised)
- `competitor_opened` (new dentist 1.3km away on GBP)
- `category_trend_movement` ("clear aligners Delhi" searches +62%)

**Internal triggers** (happen within the merchant's data):
- `perf_spike` (yesterday's views +28% vs avg)
- `perf_dip` (calls dropped 40% week-over-week)
- `milestone_reached` (crossed 100 reviews)
- `dormant_with_vera` (no merchant message in 14 days)
- `customer_lapsed_soft` (a customer's 6mo recall window opens)
- `appointment_tomorrow` (booking exists for next day)
- `review_theme_emerged` (3 reviews this week mention "wait time")
- `scheduled_recurring` (weekly Friday curious-ask cadence)

**Example trigger for Dr. Meera**:
```json
{
  "id": "trg_2026_04_26_research_digest_dentists",
  "scope": "merchant",
  "kind": "research_digest",
  "source": "external",
  "payload": {
    "category": "dentists",
    "top_item": {
      "title": "3-mo fluoride recall cuts caries recurrence 38% better than 6-mo",
      "source": "JIDA Oct 2026, p.14",
      "trial_n": 2100,
      "patient_segment": "high-risk adults"
    }
  },
  "urgency": 2,
  "suppression_key": "research:dentists:2026-W17",
  "expires_at": "2026-05-03T00:00:00Z"
}
```

### 4.4 CustomerContext (optional — only for customer-facing messages)

When the bot sends a message *on behalf of the merchant* to one of the merchant's own customers, this is populated.

```python
@dataclass
class CustomerContext:
    customer_id: str
    merchant_id: str           # which merchant this customer belongs to
    identity: CustomerIdentity # name, phone, language preference
    relationship: Relationship # first_visit, last_visit, visits_total, services_received
    state: Literal["new", "active", "lapsed_soft", "lapsed_hard", "churned"]
    preferences: Preferences   # preferred_time, channel, opt-in scope
    consent: Consent           # when + how they opted in to merchant outreach
```

**Example — Priya, a patient of Dr. Meera**:
- identity: name="Priya", phone="<redacted>", language_pref="hi-en mix"
- relationship: first_visit="2025-11-04", last_visit="2026-05-12", visits_total=4, services=["cleaning", "cleaning", "whitening", "cleaning"]
- state: "lapsed_soft" (5mo since last visit)
- preferences: weekday evening slots (derived from booking history)
- consent: opted in via merchant on 2025-11-04, scope=["recall_reminders", "appointment_reminders"]

---

## 5. The composition you must implement

```
def compose(
    category: CategoryContext,
    merchant: MerchantContext,
    trigger: TriggerContext,
    customer: Optional[CustomerContext] = None
) -> ComposedMessage:
    """
    Returns:
        body              — the WhatsApp message body
        cta               — the call-to-action (binary YES/STOP, open-ended, none)
        send_as           — "vera" (merchant-facing) or "merchant_on_behalf" (customer-facing)
        suppression_key   — for dedup
        rationale         — short explanation of why this message, what it should achieve
    """
```

### Constraints your bot must respect

1. **WhatsApp 24h session window** — the *first* outbound to a merchant or customer must use a pre-approved template (use any sensible template structure with `{{1}}/{{2}}/…` parameters; we won't actually call Meta). Subsequent messages within 24h of a merchant reply can be free-form.
2. **Body length** — no hard cap; keep it concise and context-appropriate for readability.
3. **Single primary CTA** — binary choice (YES/STOP) for action triggers; no CTA acceptable for pure-information triggers.
4. **URLs** — allowed when they add clear value to the merchant.
5. **Specificity wins** — anchor on a verifiable fact the merchant can check (number, date, headline). "X% off" is generic; "Haircut @ ₹99" is specific.
6. **Voice match** — peer/colleague tone, not promotional; technical vocabulary OK if category allows.
7. **Hindi-English code-mix is fine** — and often preferred for Indian merchant audiences. Match the merchant's language.
8. **Don't fabricate** — if data isn't in the contexts, don't invent it. No fake offers, no fake research citations, no fake competitor names.

---

## 6. The base dataset (provided to every participant)

Every participant gets the same JSON dataset:

```
dataset/
├── categories/
│   ├── dentists.json
│   ├── salons.json
│   ├── restaurants.json
│   ├── gyms.json
│   └── pharmacies.json        # 5 fully-populated CategoryContexts
├── merchants/
│   ├── m_001_drmeera_dentist_delhi.json
│   ├── m_002_studio11_salon_hyderabad.json
│   ├── m_003_pizzajunction_restaurant_delhi.json
│   ├── ...                    # 50 fully-populated MerchantContexts (10 per category)
│   └── m_050_*.json
├── customers/
│   ├── c_001_priya_for_m001.json
│   ├── c_002_rohit_for_m001.json
│   ├── ...                    # 200 customer profiles, distributed across merchants
│   └── c_200_*.json
└── triggers/
    ├── trg_external_001_diwali.json
    ├── trg_external_002_heatwave_delhi.json
    ├── trg_internal_001_recall_due_priya.json
    ├── trg_internal_002_perf_spike_m003.json
    ├── ...                    # 100 sample triggers covering all kinds
    └── trg_100_*.json
```

Each trigger references one merchant (via `payload.merchant_id`) and optionally one customer. Participants can compose any (category, merchant, trigger) or (category, merchant, trigger, customer) combination from this set.

A canonical "submission test set" is **30 specific (merchant, trigger) pairs** that all participants must produce a message for. This gives the AI judge a like-for-like comparison.

---

## 7. What participants submit

A single Python module + a JSONL file:

### 7.1 `bot.py` (your code)

```python
def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None) -> dict:
    """
    Inputs are the dicts loaded from the dataset JSON.
    Return a dict with keys: body, cta, send_as, suppression_key, rationale.
    Free to use any LLM, any prompting strategy, any retrieval.
    Must be deterministic given the same inputs (set temperature=0 if using LLMs).
    Must complete in < 30s per call.
    """
    ...
```

### 7.2 `submission.jsonl` (30 lines, one per test pair)

```json
{"test_id": "T01", "body": "Hi Dr. Meera, JIDA's Oct issue ...", "cta": "open_ended", "send_as": "vera", "suppression_key": "...", "rationale": "..."}
```

### 7.3 `README.md` (1 page max)

Briefly: your approach, what tradeoffs you made, what additional context would have helped you most.

### 7.4 Optional: `conversation_handlers.py`

If you want to demonstrate multi-turn handling (replying to merchant responses), implement:

```python
def respond(state: ConversationState, merchant_message: str) -> dict:
    """Given the conversation so far + the merchant's latest message, produce the reply."""
    ...
```

Multi-turn capability is a tiebreaker, not a requirement.

---

## 8. Evaluation methodology (how AI judges)

Submissions are scored by an LLM judge across **5 dimensions**, each 0-10. Total = 50.

| Dimension | What the judge looks for |
|---|---|
| **Specificity** | Does the message anchor on a concrete, verifiable fact from the contexts (number, date, headline, peer stat)? Penalty for generic "10% off" / "increase your sales" framings. |
| **Category fit** | Does the voice, vocabulary, and offer format match the category? Dentist messages should sound clinical-peer, not retail-promo. |
| **Merchant fit** | Is the message personalized to *this specific* merchant's state (their numbers, their offers, their conversation history)? Is the language preference honored? |
| **Trigger relevance** | Does the message clearly communicate *why now* — the specific trigger that prompted it? Not "you should improve your profile" generically. |
| **Engagement compulsion** | Would a real merchant want to reply? Does it use one or more compulsion levers (curiosity, social proof, loss aversion, effort externalization, single-binary CTA)? |

### The twist — post-submission context injection

After submissions close, the judge injects **additional context that participants didn't see**:
- New `digest` items added to category contexts (recent research/news/regulations)
- Updated `performance` snapshots on merchant contexts (their numbers shifted)
- New `triggers` (events that happened "after" submission)
- For 5 of the 30 test pairs: a populated `CustomerContext` is added (testing customer-facing capability)

The judge then asks the bot to produce **follow-up messages** for these enriched scenarios. Bots that adapt well to new context (without hallucinating, without going off-voice) score higher.

### Replay test

For the 10 highest-scoring submissions, the judge runs a **simulated multi-turn replay**:
- Judge plays the merchant.
- Bot composes the first message.
- Judge generates a plausible merchant reply (sometimes engaged, sometimes auto-reply, sometimes a hard "not interested", sometimes a curveball question).
- Bot must respond.
- 3-5 turns deep.
- Judge scores how the conversation flows: did the bot detect auto-reply? Did it route action requests correctly? Did it know when to gracefully exit?

---

## 9. Reference: real Vera conversation patterns (anonymized excerpts)

These are real conversations from production Vera — provided as reference, not as the bar to clear (you should aim higher).

### Pattern A — Real intent, real value exchange (the gold standard)

```
[USER]  Hi Vera. Please update my google profile.
[VERA]  Hi! Priya, Vera se bol rahi hoon. Aapka Google profile abhi 62.5% complete hai
        — description, website aur business hours missing hain. Kya aap yeh details
        share kar sakte hain taaki main sab update kar sakun?
[USER]  Ok please check & update the profile.
[VERA]  Done! Maine aapka Google profile update kar diya hai:
        - Business hours — daily 9 AM se 10 PM
        - Description — restaurant description add kar diya
        - Google Post — ek welcome post bhi publish kar diya
[USER]  I can't see my profile is updated yet. It is showing temporarily closed.
[VERA]  Google ke changes mein 24-48 ghante lagte hain. Kyunki aapka profile abhi
        unverified hai, Google har update review karta hai. Tab tak main aur
        cheezein add kar sakti hoon...
```
**What's right**: Hindi-English mix, specific numbers (62.5%, 24-48 ghante), Vera does work + reports back, handles uncertainty honestly.

### Pattern B — Auto-reply detection + graceful exit

```
[VERA]     Studio11 Family Salon ka Google profile check kar rahi thi — aapki 4.9★
           rating hai 298 reviews ke saath... business description missing hai.
[MERCHANT] Aapki jaankari ke liye bahut-bahut shukriya. Main aapki yeh sabhi baatein
           aur sujhaav hamari team tak pahuncha deti hoon.   ← merchant's auto-reply
[VERA]     Samajh gayi. Team tak pahunchane se pehle, kya aap khud dekhna chahingi
           ki exact kya missing hai Google pe? 2 minute ka kaam hai. Chalega?
[MERCHANT] Aapki madad ke liye shukriya, lekin main ek automated assistant hoon...
[VERA]     Koi baat nahi, samajh gayi. Main owner/manager se directly connect kar
           lungi. Aapka salon accha chal raha hai — best wishes! 🙂
```
**What's right**: Vera tried once after detecting auto-reply, then stopped wasting turns. Polite exit.

### Pattern C — Strong specificity hook

```
[VERA] Quick nudge: your dashboard shows 6,777 missed searches in Sector 14
       for makeup services — people are looking but not finding you. Want me
       to show how your listing would appear?
```
**What's right**: "6,777" is verifiable; "Sector 14" is local; "missed searches" frames as loss aversion.

### Pattern D — Intent-handoff failure (DON'T do this)

```
[MERCHANT] Mujhe magicpin judrna hai.   ← "I want to join magicpin" — clear intent
[VERA]     Badhiya! Aapke liye simple rakhti hoon: agar aapko 10-15 naye customers
           regular milne lagen to helpful hoga na?   ← went back to qualifying
```
**What's wrong**: Merchant said yes; Vera asked another qualification question. Lost momentum. Your bot should detect explicit intent and route to action immediately.

---

## 10. Compulsion levers — what makes merchants reply

These are the levers that drive engagement. Use one or more per message:

1. **Specificity / verifiability** — concrete number, date, headline, source citation
2. **Loss aversion** — "you're missing X" / "before this window closes"
3. **Social proof** — "3 dentists in your locality did Y this month"
4. **Effort externalization** — "I've drafted X — just say go" / "5-min setup"
5. **Curiosity** — "want to see who?" / "want the full list?"
6. **Reciprocity** — "I noticed Y about your account, thought you'd want to know"
7. **Asking the merchant** — "what's your most-asked treatment this week?"
8. **Single binary commitment** — Reply YES / STOP, not multi-choice

Production Vera's biggest miss is **#3 social proof** and **#7 asking the merchant** — these families barely fire today and would unlock a lot of engagement.

---

## 11. Anti-patterns the judge will penalize

- Generic offers ("Flat 30% off") when service+price is available ("Haircut @ ₹99")
- Multiple CTAs in one message ("Reply YES for X, NO for Y, MAYBE for Z")
- Buried call-to-action (the "what to do" should land in the last sentence)
- Promotional tone ("AMAZING DEAL!") for categories that need clinical/peer voice (dentists, doctors, lawyers)
- Hallucinated data (citing a "JIDA paper" not in the context, naming a competitor not in the context)
- Long preambles ("I hope you're doing well. I'm reaching out today to…")
- Re-introducing yourself after the first message
- Ignoring the language preference ("hi-en mix" merchant getting pure English)
- Sending the same message verbatim that was sent before (anti-repetition)

---

## 12. Open challenges (these are intentional — extra credit)

1. **Detect auto-replies** vs. real merchant replies and route differently. (Hint: same message verbatim 3+ times = auto-reply.)
2. **Handle intent transitions** — when the merchant says "yes I want to join" / "let's do it" / "go ahead", switch from pitch mode to action mode immediately.
3. **Multi-turn cadence planning** — within a 24h session window, what's the optimal sequence of nudges?
4. **Language detection per turn** — merchant might switch from English to Hindi mid-conversation.
5. **Knowing when to stop** — gracefully exit when the merchant signals not-interested or after 3 unanswered nudges.

---

## 13. Suggested approach (one of many — you decide)

The challenge is LLM-agnostic. Most participants will:

1. **Build a prompt template** that takes the 4 contexts as structured input and asks an LLM to compose. Likely best with a frontier model (Claude / GPT / Gemini / DeepSeek).
2. **Add a routing layer** — different trigger kinds may want different prompt variants (research-digest framing vs recall-reminder framing vs perf-dip framing).
3. **Add post-LLM validation** — check the output for CTA shape and language match. Re-prompt if it fails.
4. **For multi-turn**: maintain a small conversation state with what was already said, what the merchant signaled, what the next best move is.

You can also use retrieval (e.g., embed the category digest items, retrieve the most relevant for the trigger), tool use (e.g., let the LLM call a "lookup peer stat" function), or any other architecture.

---

## 14. Logistics (placeholders — fill in for your run)

- **Eligibility**: <e.g., undergrad/grad CS or related, India-based or remote>
- **Team size**: solo or pairs
- **Submission window**: <e.g., 14 days from launch>
- **Deliverables**: as in §7
- **Compute reimbursement**: <e.g., ₹X for API costs>
- **Prizes**: <top 1 / top 3 / honorable mentions>
- **Selection outcomes**: <e.g., top N invited to final on-site round>

---

## 15. Dataset privacy + ethics

- The base dataset is **synthetic but realistic** — generated from anonymized templates, no real PII.
- Merchant names in samples are altered; phone numbers redacted.
- Participants must **not scrape** real magicpin / Google data; must not impersonate magicpin in any external test.
- All bot output is for the challenge only; not for real merchant outreach.

---

## 16. What gets shared with the judge in the new session

When a fresh AI session starts to evaluate submissions, the judge gets:

1. **This brief** (the entire document you're reading) — full context.
2. **The base dataset** (categories/, merchants/, customers/, triggers/).
3. **The post-submission context injections** (new digest items, updated performance, new triggers, optional customer contexts).
4. **The submitted bot artifacts** (`bot.py`, `submission.jsonl`, optional `conversation_handlers.py`, `README.md`).
5. **The 5-dimension rubric** (§8) and the replay-test instructions.

That's everything the judge needs. No prior context from this session is required.

---

## Appendix A — Composition example (so the judge knows what good looks like)

**Inputs**:
- Category: `dentists` (voice=peer/clinical, peer_stats.avg_ctr=0.030, digest top item="JIDA Oct trial: 3-mo fluoride recall cuts caries 38% better")
- Merchant: Dr. Meera, Lajpat Nagar Delhi, CTR 2.1% (below peer), high-risk-adult patient cohort, last Vera touch 2 days ago (engaged)
- Trigger: `research_digest_release` (urgency 2, external)
- Customer: none

**Good composed message** (the bar):

```
Dr. Meera, JIDA's Oct issue landed. One item relevant to your high-risk adult
patients — 2,100-patient trial showed 3-month fluoride recall cuts caries
recurrence 38% better than 6-month. Worth a look (2-min abstract). Want me to
pull it + draft a patient-ed WhatsApp you can share?  — JIDA Oct 2026 p.14
```

**Why it scores well**:
- Specificity: "2,100-patient", "38% better", "JIDA Oct 2026 p.14"
- Category fit: clinical vocabulary, source citation, peer tone
- Merchant fit: "your high-risk adult patients" (derived from customer aggregate)
- Trigger relevance: explicitly references the digest as the reason for messaging
- Engagement compulsion: curiosity ("Worth a look"), reciprocity ("I'll pull it for you"), low-friction CTA

---

## Appendix B — Customer-facing composition example

**Inputs**:
- Category: `dentists` (customer-facing voice rules: no "guaranteed", no medical claims)
- Merchant: Dr. Meera (active offer "Dental Cleaning @ ₹299", available slots Wed 6pm + Thu 5pm)
- Trigger: `recall_due` (scope=customer, urgency=3, internal)
- Customer: Priya (lapsed_soft, weekday-evening preference, hi-en mix language)

**Good composed message** (sent from merchant's WA number, drafted by bot):

```
Hi Priya, Dr. Meera's clinic here 🦷 It's been 5 months since your last visit —
your 6-month cleaning recall is due. Apke liye 2 slots ready hain: Wed 6 Nov,
6pm ya Thu 7 Nov, 5pm. ₹299 cleaning + complimentary fluoride. Reply 1 for
Wed, 2 for Thu, or tell us a time that works.
```

**Why it scores well**:
- send_as = `merchant_on_behalf` (correctly attributed)
- Category fit: clinical, no overclaims
- Merchant fit: real catalog price, real open slots
- Trigger relevance: explicitly names the 6-month recall
- Customer fit: name, language mix, evening preference honored, multi-choice slot offer (allowed for booking flows)

---

## End of brief

This document is everything the next session needs. Save it as `challenge-brief.md` and load it into the judge session along with the dataset and the submitted bot artifacts.

Good luck. Build something better than today's Vera.
