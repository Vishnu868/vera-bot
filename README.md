# Vera-Bot — magicpin AI Challenge Submission

## Approach

**Single-prompt LLM composer with structured routing**, built on Claude Sonnet.

### Architecture

```
Judge Harness
    │
    ├── POST /v1/context  → In-memory context store (scope, id) → versioned payload
    ├── GET  /v1/healthz  → Live context count report
    ├── GET  /v1/metadata → Bot identity
    ├── POST /v1/tick     → Trigger prioritization → async parallel composition → actions[]
    └── POST /v1/reply    → Intent detection → auto-reply detection → contextual follow-up
```

### Key decisions

**1. Trigger-kind routing via system prompt**
Instead of separate prompt templates per trigger kind, a single rich system prompt encodes routing rules for each kind (research_digest = clinical hook + study + ask; recall_due = patient name + slots; etc.). This keeps the system flexible for unseen trigger kinds the judge might inject.

**2. Concurrency on /v1/tick**
All trigger compositions run as `asyncio.gather()` in parallel — each trigger gets its own LLM call simultaneously, staying well within the 30s timeout.

**3. Auto-reply detection**
Pattern matching on known WhatsApp Business auto-reply phrases + verbatim repeat detection. Three-tier response: flag → wait 24h → end conversation.

**4. Intent transition**
`detect_intent()` classifies every reply as accept/decline/neutral using Hindi + English signal words. On "accept", the composer switches immediately to action mode with no re-qualifying questions.

**5. Suppression log**
Every `suppression_key` is tracked in-memory. Fired triggers are not re-used on subsequent ticks.

**6. Anti-repetition**
Before returning a reply, the bot checks if the composed body matches any previous Vera turn in the same conversation.

### Model choice
`claude-sonnet-4-20250514` — fast enough for parallel composition within tick budget, quality sufficient for nuanced category voice (clinical for dentists, warm for salons, etc.).

### Compulsion levers used
- **Specificity**: real peer stats, digest study numbers, date-stamped sources
- **Effort externalization**: "I've already drafted it — say Go"
- **Loss aversion**: renewal windows, competitor opened nearby
- **Single binary CTA**: binary_yes_no or multi_choice_slot for booking flows

## Running locally

```bash
pip install fastapi uvicorn httpx anthropic
export ANTHROPIC_API_KEY=your_key
uvicorn bot:app --host 0.0.0.0 --port 8080
```

Then run the judge simulator:
```bash
export BOT_URL=http://localhost:8080
python judge_simulator.py
```

## Deployment

Deploy to any cloud with HTTPS. The bot is stateless across restarts except for in-memory context — ensure no restarts during the judge test window.

Recommended: Railway, Render, or a simple VPS with nginx + uvicorn.
