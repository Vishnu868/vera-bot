# Vera Engagement Framework — Research: Current Merchant Data Access

**Status**: Research notes — companion to `engagement-design.md`.
**Last updated**: 2026-04-26
**Scope**: How the existing system loads merchant + customer data at conversation init and during a turn. Used to inform what the new framework must build vs. adapt.

---

## TL;DR

Two distinct paths exist today — **merchant-facing** (Vera ↔ Dr. Meera) and **customer-facing** (a customer asks Vera *about* Dr. Meera). They share infrastructure (vera-mcp + merchant-support-mcp + Redis) but compose context very differently.

Most of what the proposed `MerchantContext` needs **already exists in scattered form** (`_merchant_snapshot`, `_behavioral_profile`, `_session_scenario`). The genuinely new pieces are:

- `CategoryContext` (no existing equivalent)
- `TriggerContext` (no normalized abstraction today — every cron emits ad-hoc payloads)
- `CustomerContext.relationship` (visit history aggregation doesn't exist)
- `EngagementComposer` (no shared composer — each agent has its own prompt-builder)

The aryan dependency for `category` and `locality` is the soft underbelly — every customer-info-pack call hits aryan synchronously.

---

## Path 1 — Merchant-facing (`VeraMerchantAgent`)

### Init (per session, instance-level)

`agents/vera/merchant_agent.py:402` — `__init__`:

- Spawns its own `vera_mcp_client` (`HTTPMCPClient` → `VERA_MCP_SERVER_URL`, default `vera-mcp:8000`).
- Inherits `BaseAgent._merchant_mcp_client` — class-level shared singleton pointing to `merchant-support-mcp` (`https://search5.magicpin.com/staging/mcp-server/mcp`).
- **No merchant data is loaded at init** — agent doesn't know who it's talking to yet.

### Per-turn (every inbound message)

`agents/vera/merchant_agent.py:2171` — `handle_input(agent_input)`:

1. Extract `merchant_id` from `metadata.context_data.merchant_id`, falling back to regex on the message body (`\d{7,}`).
2. Fire 2 parallel tasks:
    - `_detect_state(merchant_id)` — GBP connection + subscription status checks.
    - `_prefetch_merchant_context(merchant_id)` — full snapshot + behavioral profile.

### `_prefetch_merchant_context` — `merchant_agent.py:740`

Cache-first design:

1. **Redis check**: `vera:merchant_ctx:{merchant_id}` — TTL **30 min**. Hit → return immediately, skip everything below.
2. Parallel via `asyncio.gather`:
    - `vera_merchant_snapshot` (vera-mcp tool)
    - `vera_get_merchant_profile` (vera-mcp tool)
3. If snapshot is empty: fall back to `get_aggregated_unassociated_merchant_data` (merchant-support-mcp) — basic GBP health, no commercial data. Wrap it in a snapshot-shaped envelope.
4. Sequential enrichments:
    - `_prefetch_product_context` — fans out to `vera_get_subscription_context`, `vera_get_performance_summary`, and (only if DA subscribed) `da_get_campaign_context`. ~2KB total, embedded into snapshot.
    - `_enrich_snapshot_with_pricing` — direct HTTP to `https://vera.magicpin.com/api/v1/merchant/pricing/get?mid=...` (vera-mcp's pricing path is unreachable locally). Mounted into snapshot as `pricing_recommendation`.
5. Stuff result into Redis at the same key, TTL 30 min.

Final state: `self._merchant_snapshot` and `self._behavioral_profile` are populated. `_get_system_prompt()` reads from these and serializes the snapshot directly into the LLM system prompt (`merchant_agent.py:996`).

### What `vera_merchant_snapshot` actually fetches

`vera-mcp/src/services/merchant_snapshot.py:51` — `build_merchant_snapshot()`:

1. **Resolve identity** via `gbp_resolve_merchant` — gets `place_id`, `location_name`, `merchant_title`.
2. **One async fan-out** of up to 13 tool calls in parallel:

| Bucket | Tools |
|---|---|
| Merchant-level (no GBP needed) | `vera_get_subscription_status`, `vera_get_pricing_by_merchant`, `vera_get_merchant_pain_points`, `vera_get_merchant_offer`, `vera_get_merchant_config`, `vera_get_onboarding_status`, `vera_get_enhancement_suggestions` |
| Location-level (skipped if no place_id) | `gbp_get_location`, `gbp_get_profile_completeness`, `gbp_get_performance_summary`, `gbp_get_search_keywords`, `gbp_get_review_stats`, `gbp_list_posts` |

3. Composes 7 sections: `identity`, `profile`, `reputation`, `growth`, `commercial`, `conversation_hooks`, `issues`.

> **Note**: `category` and `locality` are not first-class fields on the snapshot — they're buried inside `profile.business_info` (from `gbp_get_location`) and surfaced via `aryan_client.get_merchant_v2()` only when `vera_get_customer_info_pack` is called.

### Mid-conversation tool calls

The LLM gets the full toolset from `MERCHANT_MCP_ALLOWED_TOOLS` (declared on the agent class) merged with `_vera_mcp_tools_cache`. Mid-turn tool calls flow through:

- `self.vera_mcp_client.call_tool(name, args)` — for vera-mcp tools (HTTP)
- `BaseAgent._merchant_mcp_client.call_tool(name, args)` — for merchant-support-mcp tools (HTTP)

No re-prefetch of the snapshot mid-turn. The agent relies on the prefetched snapshot + LLM-initiated lookups when it needs fresh data.

---

## Path 2 — Customer-facing (`CustomerIncomingAgent`)

`agents/vera/customer_incoming_agent.py:91` — different agent, different state model.

### Init + per-turn

The instance carries:

- `_merchant_id` — the merchant the customer is asking about
- `_merchant_data`, `_merchant_name`, `_category`, `_offers`, `_jd_info` — about the **business**
- `_customer_phone`, `_customer_profile` — about the **customer** (the one chatting)
- `_chat_history`, `_session_state` — conversation continuity

### Merchant data load — `_load_merchant_data` (`customer_incoming_agent.py:221`)

Single shape, no Redis cache:

1. **Primary call**: `vera_get_customer_info_pack(merchant_id)` — one MCP call returns `business_info` + `reviews` + `photos` + `offers` + `metadata` in one shot. The aggregated customer-facing endpoint.
2. **Fallback chain** if `info_pack` is empty:
    - `get_unassociated_merchant_data` — basic merchant info
    - `vera_list_merchant_offers` — separate offer list
3. **Supplementary** (non-blocking): `vera_get_merchant_jd_info` — JustDial crawl data for additional name / address / phone.

Caching is at the request level (in-memory on the agent instance) — re-using the same `merchant_id` in a session avoids re-fetching.

### What `vera_get_customer_info_pack` does

`vera-mcp/src/tools/merchant_info.py:188`:

1. **Resolve place_id**: `_resolve_place_id(merchant_id)` — chain of MongoDB (`gbp_status`) → local JSON → `aryan_client.get_mapping()`.
2. **Parallel fetch**:
    - GBP data via `get_or_fetch(place_id)` — 24h cache in MongoDB, falls through to Google Places API
    - `_get_active_offers(merchant_id)` — direct MongoDB read of `offers` collection where `status=active`
    - `_get_merchant_metadata(merchant_id)` — `aryan_client.get_merchant_v2()` → returns `name`, `category`, `locality`
3. **Last-resort fallback**: `_fetch_embed_data(place_id)` — vera REST `/embed` endpoint if everything else came back empty.

### Customer profile (the *caller*)

`_customer_profile` is loaded separately — populated in `_init_customer_data` from past conversation tickets keyed off `_customer_phone`. There's no rich CRM behind it today; it's mostly conversation continuity (last visit, last topic).

---

## Cross-cutting infrastructure

| Concern | Implementation |
|---|---|
| MCP transport | `HTTPMCPClient` (vera client) — keeps a session, calls `/mcp/tools/{name}` HTTP POST |
| Auth | Not required for challenge bot endpoints |
| Cache key for merchant context | `vera:merchant_ctx:{merchant_id}` — Redis, TTL 30 min, written by `_prefetch_merchant_context` |
| Cache key for GBP data | `gbp_health_report:{place_id}` — MongoDB, TTL 24h, in vera-mcp |
| Snapshot freshness for sends | Whatever's in Redis — not refreshed on send unless agent is in active conversation |
| Source of truth for `category` | aryan `get_merchant_v2` API (via `aryan_client`) — used in `_get_merchant_metadata` |
| Source of truth for `name` | aryan first, GBP `business_info` second |

---

## Observations relevant to the engagement framework

### What already exists

1. **Most of `MerchantContext` already exists** — spread across `_merchant_snapshot`, `_behavioral_profile`, `_session_scenario`, `_jd_info`. A `MerchantContext.from_existing(agent_state)` adapter could load ~80% of the fields without any new fetching.
2. **The customer agent already has half a `CustomerContext`** — `_customer_phone` + `_customer_profile` give us identity + conversation continuity. Missing: visit history, services received, lapse state.
3. **Two MCP servers, one orchestration** — vera-mcp (instance-level) for vera tools, merchant-support-mcp (class-level shared) for fallback merchant tools. The composer can just consume whatever `MerchantContext` already collected — no new MCP wiring required.
4. **Cache TTL of 30 min is fine** for engagement nudges that fire daily/weekly. Redis hits during conversation are plenty fresh for composition.

### What does NOT exist

1. **No `CategoryContext`.** Category is just a string buried in metadata. Voice rules, peer benchmarks, knowledge digests — none of it exists. Biggest greenfield area, but also the most leveraged (one CategoryContext serves all merchants in the vertical).
2. **No `customer_aggregate` field on the merchant snapshot.** No pipeline today aggregates per-merchant customer roster stats (active count, lapsed count, retention rate).
3. **No `TriggerContext` abstraction.** Every nudge type today has its own cron + its own fetch logic + its own send code. The proposed `TriggerContext` is the genuinely new architectural primitive; everything else is reorganization.
4. **No visit-history aggregation per (merchant, customer_phone).** Required for `CustomerContext.relationship`. BOTOPS chat history has the raw data; no derived view exists.
5. **No shared composer.** Each agent has its own prompt builder embedded in `_get_system_prompt()`. The proposed `EngagementComposer` is net new.

### Operational risks to mitigate

1. **Aryan is the synchronous bottleneck.** `aryan_client.get_merchant_v2()` is the only path to category and locality, and it's a remote HTTP call. If aryan is slow, every customer-info-pack call is slow. Worth caching aryan responses per merchant for ~24h before scaling engagement frequency.
2. **The 30-min Redis cache is keyed per-merchant** — fine for in-conversation reuse, but a daily engagement cron will miss this cache 100% of the time and pay the full snapshot-build cost per send. Consider a longer-TTL background-warmed cache for the engagement loop specifically.
3. **No version tracking on prompts today.** Every send loses the prompt-version provenance. The composer should record prompt version + context hash on every send so we can replay and A/B.

---

## Concrete recommendation for Phase 1 of the framework

Phase 1 is mostly an **adapter layer**, not new infrastructure:

| Layer | Effort | What it actually does |
|---|---|---|
| `CategoryContext` | Net new | Build for dentistry first (offer catalog, voice, peer stats, weekly digest, patient-content seed) |
| `MerchantContext` | Adapter | Wrap existing `_merchant_snapshot` + `_behavioral_profile` + a new `_customer_aggregate` derived from BOTOPS chat history |
| `TriggerContext` | Net new abstraction | Normalize the payloads existing crons emit into a single shape; new triggers (research_digest, recall_due) emit it natively |
| `CustomerContext` | Partial adapter | Wrap existing `_customer_profile`, plus a new visit-history aggregator |
| `EngagementComposer` | Net new | Single LLM-prompted module with versioned prompt; consumes the above |

Phase 1 should NOT require modifying any existing agent. The adapter reads from existing state; the composer is a new module that runs in parallel; the new triggers (research digest, recall) are new crons that don't touch the existing matrix-followup or campaign-engagement loops.

---

## Appendix: file pointers for follow-up implementation

- `agents/vera/merchant_agent.py:402` — VeraMerchantAgent init
- `agents/vera/merchant_agent.py:740` — `_prefetch_merchant_context` (the main load)
- `agents/vera/merchant_agent.py:899` — `_prefetch_product_context` (subscription + DA campaign + perf summary)
- `agents/vera/merchant_agent.py:996` — system-prompt builder reads `_merchant_snapshot`
- `agents/vera/merchant_agent.py:2171` — `handle_input` per-turn entry
- `agents/vera/customer_incoming_agent.py:91` — CustomerIncomingAgent class
- `agents/vera/customer_incoming_agent.py:221` — `_load_merchant_data` (single info-pack call + fallbacks)
- `agents/base_agent.py:80-82` — MERCHANT_MCP_SERVER_URL config
- `agents/base_agent.py:237-244` — class-level shared `_merchant_mcp_client`
- `vera-mcp/src/tools/merchant_snapshot.py` — `vera_merchant_snapshot` tool
- `vera-mcp/src/services/merchant_snapshot.py:51` — `build_merchant_snapshot` (the 13-call fan-out)
- `vera-mcp/src/tools/merchant_info.py:188` — `vera_get_customer_info_pack` tool
- `vera-mcp/src/tools/merchant_info.py:30` — `_resolve_place_id` chain (MongoDB → JSON → aryan)
- `vera-mcp/src/services/aryan_client.py` — aryan HTTP client (`get_merchant_v2`, `get_mapping`)
