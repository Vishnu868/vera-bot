# Vera Engagement Framework — Design

**Status**: Draft proposal — not implemented.
**Last updated**: 2026-04-26
**Scope**: How Vera composes every outbound conversation, both merchant-facing and (on-behalf-of-merchant) customer-facing.

---

## Why this exists

The current engagement loop (`agents/vera/followup/`) is a collection of ad-hoc nudges — campaign templates, performance reminders, conversation requeues. Each was built point-to-point, with hardcoded assumptions baked in. Adding a new nudge type means re-writing prompts, finding new data, and re-implementing send/dedup logic.

Two problems this creates:

1. **Functional nudges are inherently low-frequency.** A merchant only has a handful of "broken things" or "events" per month. To engage them 3-5×/week, we need a framework that supports curiosity-driven, knowledge-driven, and customer-driven conversations — not just "fix-this" reminders.

2. **Same engine should drive merchant-facing AND customer-facing messages.** When Dr. Meera's clinic sends a recall reminder to her patient Priya, Vera does the work — but the framework for composing that message should be the same one that produces the research digest Vera sends to Dr. Meera herself.

This doc proposes a **4-context composition framework** that separates the slow-changing (category knowledge) from the fast-changing (per-merchant, per-customer state) and the event-driven (triggers).

---

## The 4 contexts

Every Vera message = `compose(category, merchant, trigger, customer?)`.

| Context | Question it answers | Refresh cadence | Owner | Shared across |
|---|---|---|---|---|
| **Category** | How do we talk to *this type* of business? What do they know, care about, offer, get regulated on? | Weekly (digest), monthly (voice/catalog) | Domain curator | All merchants in the vertical |
| **Merchant** | Who is this specific business, how are they doing, what's in their history with us? | Daily (perf), real-time (conversation) | Snapshot + conversation sync | Just this merchant |
| **Trigger** | Why am I messaging *right now*? What specific event prompts this? | Per-event | Event detectors | This one send |
| **Customer** | Who is the merchant's customer, and what's their state with this merchant? | Per-visit / per-interaction | Merchant CRM sync | Just this customer |

The composer takes these 4 (3 if no customer scope) and produces the message body, template parameters, CTA, and suppression key. Nothing is hardcoded into the composer — all variation comes from the contexts.

```
                  ┌─────────────────┐
   Category   ───►│                 │
   Merchant   ───►│  Composer (LLM) │───► message {body, params, cta, send_as}
   Trigger    ───►│                 │
   Customer?  ───►│                 │
                  └─────────────────┘
```

---

## Layer details

### CategoryContext

Slow-changing knowledge pack per vertical. One per category (`dentists`, `salons`, `gyms`, `restaurants`, `car_service`, ...). Bootstrap is labor-intensive (needs domain expertise); ongoing maintenance is mostly the weekly digest.

Fields:

- `slug` — `"dentists"`
- `offer_catalog` — canonical service+price patterns from vera-mcp + category-specific extensions. Prefer `"Dental Cleaning @ ₹299"` over `"Flat 20% OFF"`. Service+price is more compelling than discount.
- `voice` — tone, vocabulary, taboos. For dentists: technical terms welcome (`"fluoride varnish at 3-month recall"`), legal taboos (`"cure"`, `"guaranteed"`), peer tone not hype.
- `peer_stats` — city-scoped benchmarks: avg rating, avg reviews, typical CTR, typical patient volume. Anchors comparative messages.
- `digest` — this week's curated research / compliance / CDE / tech / peer-practice items, with source citations. Sourced from a per-category source list (PubMed dental RSS, JIDA, IDA Delhi calendar, DCI circulars, Dental Tribune India, Google Trends for dental queries, vendor press releases).
- `patient_content_library` — items written at patient-reading level that the merchant can reshare with their own customers (powers the `PRO_PATIENT_CONTENT` family).
- `seasonal_beats` — cycles like "exam-stress bruxism spikes Nov-Feb" that cue category-specific timing.
- `trend_signals` — Google Trends + Practo-style query data showing what patients in this vertical are searching for.

### MerchantContext

Per-merchant state. Refreshed daily for performance; real-time for conversation history.

Fields:

- `merchant_id`
- `identity` — name, place_id, locality, city, verified, languages
- `subscription` — status, days remaining, plan
- `performance` — views/calls/CTR/leads/directions, 30d + 7d deltas
- `offers` — active + paused, sourced from vera-mcp's offers collection (and eventually the as-yet-undefined "real" offer source-of-truth)
- `conversation_history` — last N turns w/ Vera, with engagement tags (replied, ignored, unsubscribed-from-topic)
- `customer_aggregate` — derived stats over the merchant's customer roster (active count, lapsed count, retention rate). Not individual customers — aggregates only.
- `signals` — derived flags: `stale_posts`, `ctr_below_peer_median`, `customer_lapse_rate_high`, `dormant_with_vera`, ...

### TriggerContext

The event that prompts this specific message. Two families:

- **External** — happens outside the merchant's account. News, weather, festival, regulation change, category-trend movement, competitor opens nearby, weekly research digest release.
- **Internal** — happens within the merchant's account or customer roster. Performance dip/spike, milestone hit, dormancy threshold crossed, customer lapse threshold crossed, appointment due, review pattern emerged, scheduled-recurring nudge.

Fields:

- `id` — unique
- `scope` — `merchant` | `customer`
- `kind` — `research_digest`, `recall_due`, `perf_spike`, `competitor_opened`, `festival`, ...
- `source` — `external` | `internal`
- `payload` — kind-specific data (e.g., for `recall_due`: `{patient_id, last_visit, due_date}`)
- `urgency` — 1-5; ranks against other queued triggers
- `suppression_key` — used by Redis dedup to prevent re-sends
- `expires_at` — after which the trigger is stale

### CustomerContext

Only populated when `scope=customer`. Per-customer state with this specific merchant.

Fields:

- `customer_id`
- `merchant_id`
- `identity` — name, phone, language preference
- `relationship` — first_visit, last_visit, visits_total, services received, lifetime value
- `state` — `new` | `active` | `lapsed_soft` (3-6mo) | `lapsed_hard` (6mo+) | `churned` (12mo+)
- `preferences` — preferred slot times (derived from booking history), preferred channel, opt-in status
- `consent` — when did they opt in, via what mechanism, scope of consent

---

## Composer

Single LLM-prompted module. Takes the 4 contexts as input. Produces:

- `body` — the WhatsApp message body
- `template_params` — params to fill an approved Kaleyra template (used only for the first touch in a session window)
- `cta` — the binary or open-ended ask
- `suppression_key` — for the trigger-level dedup
- `send_as` — `"vera"` for merchant-facing, `"merchant_on_behalf"` for customer-facing

The composer prompt is the single point of failure. It must be:

- Versioned (`composer_v1`, `composer_v2`, ...)
- A/B-testable
- Auditable (we can replay any past message and see all 4 input contexts)

Different `kind` values may use different prompt variants — e.g., `research_digest` needs source-citation framing, `recall_due` needs slot-offering framing, `competitor_opened` needs voyeur-curiosity framing. The composer dispatches by `kind`.

---

## Worked example 1: merchant-facing

**Merchant**: Dr. Meera, Lajpat Nagar, Delhi
**Trigger**: weekly dentistry research digest just landed

**Inputs:**

| Context | Key values used |
|---|---|
| Category (dentists) | voice=peer/technical; digest_top_item="JIDA Oct trial: 3-mo fluoride recall cuts caries 38% better"; peer_stat="South-Delhi solo CTR median 3.0%" |
| Merchant (Dr. Meera) | CTR 2.1% (below peer); ran "Deep Cleaning ₹499" 2mo ago; 78 lapsed patients; last Vera touch 2d ago (engaged) |
| Trigger | kind=`research_digest_release`, scope=merchant, urgency=2, source=external, suppression_key=`research:dentists:2026-W17` |
| Customer | (not populated) |

**Composed message:**

> Dr. Meera, JIDA's Oct issue landed. One item relevant to your high-risk adult patients — 2,100-patient trial showed 3-month fluoride recall cuts caries recurrence 38% better than 6-month. Worth a look (2-min abstract). Want me to pull it + draft a patient-ed WhatsApp you can share?  *— JIDA Oct 2026 p.14*

Why it works:
- **Category** drives voice (technical, source-cited, peer tone)
- **Merchant** drives specificity ("your high-risk adult patients" — derived from her customer aggregate)
- **Trigger** drives the hook (this week's digest, not a promo ask)
- No customer context needed; this is merchant-to-Vera

---

## Worked example 2: customer-facing (same framework)

**Merchant**: Dr. Meera (same)
**Customer**: Priya — patient since 2025-11, last visit 2026-05 (cleaning + whitening), prefers weekday evenings, opted-in to reminders
**Trigger**: 6-month recall window opens

**Inputs:**

| Context | Key values used |
|---|---|
| Category (dentists, customer-facing) | voice=warm-clinical; taboos=no medical claims, no "guaranteed"; recall framing pattern |
| Merchant (Dr. Meera) | active offer `Dental Cleaning @ ₹299`; available slots Wed 6pm + Thu 5pm (next 7d); WhatsApp Business number |
| Trigger | kind=`recall_due`, scope=customer, urgency=3, source=internal, payload={patient_id: priya, last_visit: 2026-05, due_date: 2026-11} |
| Customer (Priya) | name + phone; lapsed_soft state; preferred=weekday evening; consent active; language=Hindi-English mix |

**Composed message** (sent from Dr. Meera's WhatsApp number, drafted by Vera):

> Hi Priya, Dr. Meera's clinic here 🦷 It's been 5 months since your last visit — your 6-month cleaning recall is due. Apke liye 2 slots ready hain: **Wed 6 Nov, 6pm** ya **Thu 7 Nov, 5pm**. ₹299 cleaning + complimentary fluoride. Reply 1 for Wed, 2 for Thu, or tell us a time that works.

Why it works:
- **Category** sets the legal/clinical voice constraints
- **Merchant** provides the actual catalog price + actual open slots from the schedule
- **Trigger** provides the recall payload (last_visit, due_date)
- **Customer** drives personalization (name, language mix, evening preference)

Same composer. Different context inputs. Two completely different conversations.

---

## Engagement loops this enables

Once the framework exists, every loop is just a small cron that emits `TriggerContext` instances. The composer handles the rest.

| Loop | Emits trigger kinds | Scope |
|---|---|---|
| News/weather scanner *(already built — see `agents/vera/followup/event_sources.py`)* | `external` (festival, heatwave, fuel, IPL, monsoon, news) | merchant |
| Weekly research digest per category | `research_digest_release` | merchant |
| Performance monitor | `perf_spike`, `perf_dip`, `milestone_reached` | merchant |
| Review-pattern detector | `review_theme_emerged` | merchant |
| Conversation curiosity-ask scheduler | `curious_ask_due` | merchant |
| Recall scheduler (from merchant CRM) | `recall_due` | customer |
| Lapse detector | `customer_lapsed_soft`, `customer_lapsed_hard` | customer |
| Appointment reminder | `appointment_tomorrow` | customer |
| Capacity optimizer | `unplanned_slot_open` (offered to likely-to-book lapsed customers) | customer |

Adding a loop = define one new `kind`, implement the detector, add a composer prompt variant. No change to merchant/category/customer code.

---

## Implementation shape

```python
# agents/vera/engagement/contexts.py

@dataclass
class CategoryContext:
    slug: str                                # "dentists"
    offer_catalog: list[OfferTemplate]
    voice: VoiceProfile
    peer_stats: PeerStats
    digest: list[DigestItem]
    patient_content_library: list[ContentItem]
    seasonal_beats: list[SeasonalBeat]
    trend_signals: list[TrendSignal]

@dataclass
class MerchantContext:
    merchant_id: str
    identity: Identity
    subscription: Subscription
    performance: PerformanceSnapshot
    offers: list[MerchantOffer]
    conversation_history: ConversationHistory
    customer_aggregate: CustomerAggregate
    signals: list[DerivedSignal]

@dataclass
class TriggerContext:
    id: str
    scope: Literal["merchant", "customer"]
    kind: str
    source: Literal["external", "internal"]
    payload: dict
    urgency: int                              # 1-5
    suppression_key: str
    expires_at: datetime

@dataclass
class CustomerContext:
    customer_id: str
    merchant_id: str
    identity: CustomerIdentity
    relationship: Relationship
    state: Literal["new", "active", "lapsed_soft", "lapsed_hard", "churned"]
    preferences: Preferences
    consent: Consent
```

```python
# agents/vera/engagement/composer.py

class EngagementComposer:
    def compose(self,
                category: CategoryContext,
                merchant: MerchantContext,
                trigger: TriggerContext,
                customer: CustomerContext | None = None) -> ComposedMessage:
        """Returns ComposedMessage(body, template_params, cta,
        suppression_key, send_as)."""
```

Both engagement surfaces (merchant-facing, customer-on-behalf-of-merchant) call the same composer. The only thing that changes is whether `customer` is populated.

---

## Phased rollout

### Phase 1 — framework skeleton + dentistry vertical (≈ 2 weeks)

1. Define the 4 dataclasses in `agents/vera/engagement/contexts.py`.
2. Build the `CategoryContext` for dentistry — offer catalog, voice profile, peer stats, one weekly research digest pipeline, patient-content seed.
3. Build `MerchantContext` loader from the existing `merchant_snapshot_data` collection.
4. Build the first `EngagementComposer` with a prompt that handles 2 trigger kinds (`research_digest_release` and one merchant-facing perf trigger).
5. Render (no send) the Dr. Meera research-digest message end-to-end from the 4 contexts. Inspect the output before any send happens.

### Phase 2 — customer-on-behalf sends (≈ 2 weeks)

6. **Resolve the customer-data source-of-truth.** This is the biggest unknown. Options: clinic SaaS integration (Practo, Dentcubate), merchant CSV upload, BOTOPS chat-derived patient list. Without this, customer engagement is theoretical.
7. Define the consent model: customer opted in via merchant, not via Vera directly. Capture timestamp + scope.
8. Stand up a send-as-merchant channel: WhatsApp Business API under the merchant's number, or Vera's shared number with attribution `"Dr. Meera's clinic via Vera"`.
9. Ship the first customer-facing trigger in production: `recall_due`. Lowest abuse risk, highest merchant intent.

### Phase 3 — multiply verticals (≈ 1 week per vertical)

10. Replicate `CategoryContext` for 4-5 more verticals (salons, gyms, pharmacies, restaurants, car service). Mostly data filling, not code.
11. Add 3 more triggers per scope. Merchant: `perf_dip`, `milestone_reached`, `review_theme_emerged`. Customer: `customer_lapsed_soft`, `appointment_tomorrow`, `unplanned_slot_open`.

---

## Open questions

These need answers before Phase 2 can ship:

1. **Where does the merchant's customer list live?** No clean answer yet. Most likely: per-merchant clinic software with no standard integration. May need a self-serve CSV upload or a per-vertical SaaS adapter.
2. **Consent architecture.** Can Vera message a patient directly, or must every outbound require merchant approval before send? Recommendation: templated auto-sends with merchant override available, switching to fully-auto after the merchant has approved N consecutive sends.
3. **Attribution.** Does the patient see "Dr. Meera's clinic" or "Vera on behalf of Dr. Meera's clinic"? Trust + legal implications either way. Probably category-dependent (regulated verticals need clearer attribution).
4. **Composer prompt versioning.** Single point of failure. Versioned + A/B tested from day 1 — every send records the prompt version that produced it.
5. **Offer source-of-truth.** Per the parallel discussion, the canonical merchant offer catalog likely lives outside vera-mcp (aryan `catalogoffer`, merchant-portal-api, or magicpin_jobs output). MerchantContext needs to read from that source — pending identification.
6. **Composer model choice.** Azure OpenAI primary, Deepseek fallback (matching `template_generator._call_llm`)? Or is there a case for a smaller faster model for high-volume per-customer sends?

---

## Why this is worth building

- **Engagement frequency goes from "few times a month" to "few times a week"** — by adding curiosity-driven, knowledge-driven, and customer-driven loops on top of the existing functional ones.
- **One framework, two products** — the same composition engine drives Vera-to-merchant *and* merchant-to-customer messaging. Build once, ship twice.
- **Vertical scaling is data work, not code work** — adding a new category becomes "fill in a CategoryContext", not "write a new agent".
- **Auditable + versioned** — every message has explicit inputs and a versioned composer; we can replay, A/B test, and answer "why did Vera send this?" for any past send.

---

## Appendix: relationship to existing code

- `agents/vera/followup/event_sources.py` and `agents/vera/followup/event_extractor.py` *(branch `feature/vera-campaign-engagement`)* already produce external `TriggerContext`-shaped objects for the news/weather scanner. They become the first concrete trigger source feeding the new composer.
- `agents/vera/followup/template_registry.py` will continue to host the Kaleyra-approved template names (used for the first-touch send before the 24h session window opens). The composer fills the template parameters.
- `agents/vera/followup/snapshot_data.py` already provides most of the `MerchantContext` fields. Customer aggregate fields would be added as new sections on `MerchantSnapshotData`.
- `services/vera-mcp/src/services/offer_suggester.py` is the leading candidate for `CategoryContext.offer_catalog` (pending the open offer source-of-truth question).
