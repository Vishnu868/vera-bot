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
pip install fastapi uvicorn httpx

# Pick ONE of these — Groq is recommended (free, fast):
export LLM_PROVIDER=groq
export GROQ_API_KEY=gsk_xxxxxxxxxxxx    # free at console.groq.com

uvicorn bot:app --host 0.0.0.0 --port 8080
```

### All supported providers

| Provider | Env vars to set | Free? | Notes |
|---|---|---|---|
| `groq` (default) | `GROQ_API_KEY` | ✅ Free | Fastest. Get key at console.groq.com |
| `gemini` | `GEMINI_API_KEY` | ✅ Free tier | aistudio.google.com |
| `openrouter` | `OPENROUTER_API_KEY` | ✅ Free models | openrouter.ai — pick `meta-llama/llama-3.3-70b-instruct:free` |
| `ollama` | `OLLAMA_URL` (optional) | ✅ Local | Run `ollama serve` + `ollama pull llama3.2` |
| `openai` | `OPENAI_API_KEY` | 💰 Paid | |
| `anthropic` | `ANTHROPIC_API_KEY` | 💰 Paid | |

To override the default model for any provider:
```bash
export GROQ_MODEL=mixtral-8x7b-32768      # or any Groq model
export GEMINI_MODEL=gemini-1.5-pro        # if you have pro access
export OPENROUTER_MODEL=mistralai/mistral-7b-instruct:free
```

## Deployment

Deploy to any cloud with HTTPS. The bot is stateless across restarts except for in-memory context — ensure no restarts during the judge test window.

Recommended: Railway, Render, or a simple VPS with nginx + uvicorn.
