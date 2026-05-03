"""
Vera-Bot — magicpin AI Challenge submission
Multi-provider LLM support: Groq (free), OpenAI, Anthropic, Gemini, OpenRouter, Ollama.
Set LLM_PROVIDER + the matching API key env var. Groq is default (free tier).
"""

import os
import time
import json
import asyncio
from groq import Groq
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ─── Provider config ─────────────────────────────────────────────────────────
# Set LLM_PROVIDER to one of: groq | openai | anthropic | gemini | openrouter | ollama
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()

# Free options:
#   groq        → GROQ_API_KEY       (free at console.groq.com)
#   gemini      → GEMINI_API_KEY     (free tier at aistudio.google.com)
#   openrouter  → OPENROUTER_API_KEY (free models at openrouter.ai)
#   ollama      → no key needed      (local, free — set OLLAMA_URL)
# Paid options:
#   openai      → OPENAI_API_KEY
#   anthropic   → ANTHROPIC_API_KEY

PROVIDER_CONFIGS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "model_default": "llama-3.3-70b-versatile",   # free, fast, 70B
        "model_env": "GROQ_MODEL",
        "format": "openai",
    },
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "key_env": "OPENAI_API_KEY",
        "model_default": "gpt-4o-mini",
        "model_env": "OPENAI_MODEL",
        "format": "openai",
    },
    "anthropic": {
        "url": "https://api.anthropic.com/v1/messages",
        "key_env": "ANTHROPIC_API_KEY",
        "model_default": "claude-3-5-haiku-20241022",  # cheapest claude
        "model_env": "ANTHROPIC_MODEL",
        "format": "anthropic",
    },
    "gemini": {
        # model is inserted into URL
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "key_env": "GEMINI_API_KEY",
        "model_default": "gemini-1.5-flash",           # free tier
        "model_env": "GEMINI_MODEL",
        "format": "gemini",
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "model_default": "meta-llama/llama-3.3-70b-instruct:free",  # free on openrouter
        "model_env": "OPENROUTER_MODEL",
        "format": "openai",
    },
    "ollama": {
        "url": os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/chat",
        "key_env": None,
        "model_default": "llama3.2",
        "model_env": "OLLAMA_MODEL",
        "format": "ollama",
    },
}

cfg = PROVIDER_CONFIGS.get(LLM_PROVIDER, PROVIDER_CONFIGS["groq"])
LLM_API_KEY = os.environ.get(cfg["key_env"], "") if cfg["key_env"] else ""
LLM_MODEL   = os.environ.get(cfg.get("model_env", ""), "") or cfg["model_default"]
LLM_URL     = cfg["url"]
LLM_FORMAT  = cfg["format"]

START_TIME = time.time()

app = FastAPI(title="Vera-Bot")

# ─── In-memory state ──────────────────────────────────────────────────────────
# (scope, context_id) → {version, payload}
contexts: dict[tuple[str, str], dict] = {}
# conversation_id → {merchant_id, customer_id, turns:[{from, body}], trigger_id, suppressed}
conversations: dict[str, dict] = {}
# suppression_key → bool (True = used)
suppression_log: set[str] = set()
# merchant_id → set of conversation_ids (to avoid re-opening same conversation)
merchant_active_convs: dict[str, set] = {}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_ctx(scope: str, context_id: str) -> dict | None:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def all_contexts_of_scope(scope: str) -> list[dict]:
    return [v["payload"] for (s, _), v in contexts.items() if s == scope]


def is_auto_reply(message: str, conv_turns: list[dict]) -> bool:
    """Detect WhatsApp Business canned auto-replies."""
    msg_lower = message.lower()
    auto_signals = [
        "thank you for contacting",
        "we will respond shortly",
        "our team will get back",
        "automated response",
        "auto-reply",
    ]
    if any(sig in msg_lower for sig in auto_signals):
        return True
    # Same message verbatim sent before
    recent_merchant_msgs = [t["body"] for t in conv_turns if t["from"] == "merchant"]
    if recent_merchant_msgs.count(message) >= 1:
        return True
    return False


def detect_intent(message: str) -> str:
    """Classify merchant's intent from a reply."""
    msg_lower = message.lower()
    affirmative = ["yes", "ok", "sure", "go ahead", "let's do", "do it", "sounds good",
                   "great idea", "confirm", "haan", "ha ", "bilkul", "theek", "karo"]
    negative = ["no", "stop", "not interested", "don't send", "nahin", "nahi", "band karo",
                "bother", "useless", "annoying", "spam"]
    if any(w in msg_lower for w in negative):
        return "decline"
    if any(w in msg_lower for w in affirmative):
        return "accept"
    return "neutral"


def count_auto_replies(conv_turns: list[dict]) -> int:
    msgs = [t["body"] for t in conv_turns if t["from"] == "merchant"]
    if not msgs:
        return 0
    last = msgs[-1]
    return sum(1 for m in msgs if m == last)


# ─── LLM composer ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Vera — magicpin's AI merchant assistant. You compose WhatsApp messages to help Indian merchants grow their businesses.

COMPOSITION RULES:
1. ONE clear CTA per message. Never list multiple actions.
2. Use REAL numbers, dates, and facts from the provided context. Never hallucinate.
3. Match the merchant's language preference (hi-en mix = Hindi-English naturally mixed, not forced).
4. NO URLs in messages (WhatsApp blocks them, -3 penalty per URL).
5. NO preambles like "I hope you're doing well". Get to the point.
6. Don't re-introduce yourself after the first message.
7. Match category voice: dentists = peer_clinical, salons = warm_practical, restaurants = warm_practical, gyms = energetic_coach, pharmacies = trustworthy_precise.
8. End with the CTA — it's the last thing they read.
9. Never use taboo words for the category (guaranteed, miracle, best in city, etc.).
10. For customer-facing messages (send_as=merchant_on_behalf): speak as the merchant's clinic/salon/etc, not as Vera.

COMPULSION LEVERS to use (pick 1-2 per message):
- Specificity: concrete numbers, dates, sources
- Loss aversion: "before this closes", "only N slots left"
- Social proof: "3 clinics in your area did this"
- Effort externalization: "I've already drafted it — just say Go"
- Curiosity: short hook question
- Single binary ask: YES/NO or 1/2 choice

TRIGGER-KIND ROUTING:
- research_digest: clinical hook + specific study finding + low-friction ask
- recall_due: personal patient name + time since last visit + 2 specific slots
- perf_dip: specific metric + comparison + single diagnostic question  
- perf_spike: celebrate + attribute cause + one growth next-step
- renewal_due: urgency + what they lose + single action
- competitor_opened: neutral framing + differentiation hook
- festival_upcoming: early mover advantage + simple content offer
- review_theme_emerged: specific theme + count + one fix suggestion
- customer_lapsed_hard: empathy + reactivation offer + low bar to return
- trial_followup: warmth + next step + low friction
- winback_eligible: honest framing + specific value add
- curious_ask_due: one curious question about their business
- active_planning_intent: immediately continue the merchant's stated intent — no re-qualifying!
- chronic_refill_due: urgency (stock runs out date) + auto-delivery pitch
- supply_alert: urgency level 5 = lead with the alert fact directly

OUTPUT FORMAT (JSON only, no markdown):
{
  "body": "the message text",
  "cta": "open_ended | binary_yes_no | binary_confirm_cancel | multi_choice_slot | none",
  "send_as": "vera | merchant_on_behalf",
  "rationale": "1-2 sentences explaining decision + context used"
}"""


async def call_llm(system: str, user: str) -> str:
    """Universal LLM caller — routes to the configured provider."""

    if LLM_PROVIDER == "groq":
        client = Groq(api_key=LLM_API_KEY)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_completion_tokens=600,
        )
        return resp.choices[0].message.content.strip()

    async with httpx.AsyncClient(timeout=25.0) as client:

        if LLM_FORMAT == "openai":
            headers = {"Content-Type": "application/json"}
            if LLM_API_KEY:
                headers["Authorization"] = f"Bearer {LLM_API_KEY}"
            if LLM_PROVIDER == "openrouter":
                headers["HTTP-Referer"] = "https://magicpin-vera-bot.app"

            resp = await client.post(LLM_URL, headers=headers, json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.3,
                "max_tokens": 600,
            })
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

        elif LLM_FORMAT == "anthropic":
            resp = await client.post(LLM_URL, headers={
                "x-api-key": LLM_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }, json={
                "model": LLM_MODEL,
                "max_tokens": 600,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            })
            resp.raise_for_status()
            return resp.json()["content"][0]["text"].strip()

        elif LLM_FORMAT == "gemini":
            url = LLM_URL.format(model=LLM_MODEL) + f"?key={LLM_API_KEY}"
            resp = await client.post(url, json={
                "contents": [{"parts": [{"text": f"{system}\n\n{user}"}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 600},
            })
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

        elif LLM_FORMAT == "ollama":
            resp = await client.post(LLM_URL, json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
            })
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()

    return ""

async def compose_message(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None,
    conversation_history: list[dict],
    is_reply: bool = False,
    merchant_reply: str = "",
) -> dict:
    """Call Claude to compose the next message."""
    
    trigger_kind = trigger.get("kind", "unknown")
    merchant_name = merchant.get("identity", {}).get("name", "merchant")
    owner_first = merchant.get("identity", {}).get("owner_first_name", "")
    lang_pref = merchant.get("identity", {}).get("languages", ["en"])
    
    # Resolve digest item if trigger references one
    digest_item = None
    if trigger.get("payload", {}).get("top_item_id"):
        top_id = trigger["payload"]["top_item_id"]
        for item in category.get("digest", []):
            if item.get("id") == top_id:
                digest_item = item
                break
    
    context_block = {
        "category": {
            "slug": category.get("slug"),
            "voice": category.get("voice", {}),
            "peer_stats": category.get("peer_stats", {}),
            "digest_item": digest_item,
            "offer_catalog": category.get("offer_catalog", [])[:4],
            "seasonal_beats": category.get("seasonal_beats", []),
            "patient_content_library": category.get("patient_content_library", [])[:2],
        },
        "merchant": {
            "identity": merchant.get("identity", {}),
            "subscription": merchant.get("subscription", {}),
            "performance": merchant.get("performance", {}),
            "offers": merchant.get("offers", []),
            "customer_aggregate": merchant.get("customer_aggregate", {}),
            "signals": merchant.get("signals", []),
            "review_themes": merchant.get("review_themes", []),
            "conversation_history": merchant.get("conversation_history", [])[-3:],
        },
        "trigger": {
            "kind": trigger_kind,
            "source": trigger.get("source"),
            "urgency": trigger.get("urgency"),
            "payload": trigger.get("payload", {}),
        },
        "customer": customer,
        "current_conversation": conversation_history[-6:] if conversation_history else [],
    }
    
    if is_reply:
        user_msg = f"""Merchant replied: "{merchant_reply}"

Context:
{json.dumps(context_block, ensure_ascii=False, indent=2)}

Compose the bot's NEXT message responding to this reply. Follow the trigger kind '{trigger_kind}' and intent.
If merchant said YES/accept/go → move to action, don't re-qualify.
If merchant declined → graceful end.
Output JSON only."""
    else:
        user_msg = f"""Compose a proactive Vera message for this merchant.

Context:
{json.dumps(context_block, ensure_ascii=False, indent=2)}

Trigger kind: {trigger_kind} (urgency: {trigger.get('urgency', 2)}/5)
Language pref: {lang_pref}

Output JSON only."""

    text = await call_llm(SYSTEM_PROMPT, user_msg)
    
    # Parse JSON output
    # Strip markdown fences if present
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = text[:-3].strip()
    
    try:
        result = json.loads(text)
    except Exception:
        # Fallback
        result = {
            "body": text[:400],
            "cta": "open_ended",
            "send_as": "vera",
            "rationale": "Parsed fallback",
        }
    
    return result


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vishnu",
        "team_members": ["Vishnu"],
        "model": f"{LLM_PROVIDER}/{LLM_MODEL}",
        "approach": "Multi-provider LLM composer with trigger-kind routing, auto-reply detection, intent transition handling, suppression log, and language-aware generation",
        "contact_email": "vishnu@example.com",
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in ("category", "merchant", "customer", "trigger"):
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope", "details": f"Unknown scope: {body.scope}"},
        )
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return JSONResponse(
            status_code=409,
            content={"accepted": False, "reason": "stale_version", "current_version": cur["version"]},
        )
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    
    # Process triggers — prioritize by urgency descending, cap at 20
    trigger_data = []
    for trg_id in body.available_triggers:
        trg = get_ctx("trigger", trg_id)
        if not trg:
            continue
        # Skip suppressed
        sup_key = trg.get("suppression_key", "")
        if sup_key and sup_key in suppression_log:
            continue
        # Skip expired
        exp = trg.get("expires_at", "")
        if exp and exp < body.now:
            continue
        trigger_data.append((trg.get("urgency", 1), trg_id, trg))
    
    trigger_data.sort(key=lambda x: -x[0])
    trigger_data = trigger_data[:20]
    
    # Compose messages concurrently
    tasks = []
    task_meta = []
    
    for urgency, trg_id, trg in trigger_data:
        merchant_id = trg.get("merchant_id")
        customer_id = trg.get("customer_id")
        merchant = get_ctx("merchant", merchant_id) if merchant_id else None
        if not merchant:
            continue
        category_slug = merchant.get("category_slug", "")
        category = get_ctx("category", category_slug)
        if not category:
            continue
        customer = get_ctx("customer", customer_id) if customer_id else None
        
        conv_id = f"conv_{merchant_id}_{trg_id}"
        # Don't re-send if conversation already active
        if conv_id in conversations:
            continue
        
        tasks.append(compose_message(category, merchant, trg, customer, []))
        task_meta.append((conv_id, merchant_id, customer_id, trg_id, trg))
    
    if not tasks:
        return {"actions": []}
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            actions.append({
                "conversation_id": f"debug_{i}",
                "merchant_id": "",
                "customer_id": "",
                "send_as": "vera",
                "trigger_id": "",
                "template_name": "debug_error",
                "template_params": [],
                "body": f"LLM compose failed: {str(result)}",
                "cta": "none",
                "suppression_key": "",
                "rationale": "debug exception"
            })
            continue
        conv_id, merchant_id, customer_id, trg_id, trg = task_meta[i]
        trg_data = get_ctx("trigger", trg_id) or {}
        
        body_text = result.get("body", "")
        if not body_text:
            continue
        
        send_as = result.get("send_as", "vera")
        cta = result.get("cta", "open_ended")
        rationale = result.get("rationale", "")
        sup_key = trg_data.get("suppression_key", "")
        
        # Register conversation
        conversations[conv_id] = {
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "trigger_id": trg_id,
            "turns": [{"from": "vera", "body": body_text}],
            "suppressed": False,
            "auto_reply_count": 0,
        }
        if sup_key:
            suppression_log.add(sup_key)
        
        merchant = get_ctx("merchant", merchant_id) or {}
        owner_first = merchant.get("identity", {}).get("owner_first_name", "")
        
        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trg_id,
            "template_name": f"vera_{trg_data.get('kind', 'generic')}_v1",
            "template_params": [owner_first, body_text[:100]],
            "body": body_text,
            "cta": cta,
            "suppression_key": sup_key,
            "rationale": rationale,
        })
    
    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv = conversations.get(body.conversation_id)
    if not conv:
        # Unknown conversation — graceful end
        return {"action": "end", "rationale": "Unknown conversation ID; closing."}
    
    turns = conv.get("turns", [])
    
    # ── Auto-reply detection ──────────────────────────────────────────────────
    if is_auto_reply(body.message, turns):
        conv.setdefault("auto_reply_count", 0)
        conv["auto_reply_count"] += 1
        
        if conv["auto_reply_count"] == 1:
            turns.append({"from": "merchant", "body": body.message})
            return {
                "action": "send",
                "body": "Looks like an auto-reply 😊 Jab owner dekhein, just reply 'Yes' to proceed.",
                "cta": "binary_yes_no",
                "rationale": "First auto-reply detected; flagging for owner with a simple ask.",
            }
        elif conv["auto_reply_count"] == 2:
            turns.append({"from": "merchant", "body": body.message})
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Second identical auto-reply → owner not at phone. Waiting 24h.",
            }
        else:
            return {
                "action": "end",
                "rationale": "Auto-reply 3x in a row. No real engagement signal. Closing conversation.",
            }
    
    # ── Intent detection ─────────────────────────────────────────────────────
    intent = detect_intent(body.message)
    turns.append({"from": "merchant", "body": body.message})
    
    if intent == "decline":
        return {
            "action": "end",
            "rationale": "Merchant declined or expressed frustration. Closing gracefully without further engagement.",
        }
    
    if intent == "accept":
        # Immediate action mode — no more qualifying questions
        merchant_id = conv.get("merchant_id") or body.merchant_id
        customer_id = conv.get("customer_id") or body.customer_id
        trigger_id = conv.get("trigger_id", "")
        
        merchant = get_ctx("merchant", merchant_id) if merchant_id else None
        trg = get_ctx("trigger", trigger_id) if trigger_id else None
        customer = get_ctx("customer", customer_id) if customer_id else None
        
        if merchant and trg:
            category_slug = merchant.get("category_slug", "")
            category = get_ctx("category", category_slug)
            if category:
                try:
                    result = await compose_message(
                        category, merchant, trg, customer, turns,
                        is_reply=True, merchant_reply=body.message
                    )
                    reply_body = result.get("body", "")
                    turns.append({"from": "vera", "body": reply_body})
                    return {
                        "action": "send",
                        "body": reply_body,
                        "cta": result.get("cta", "open_ended"),
                        "rationale": result.get("rationale", "Merchant accepted; moving to action."),
                    }
                except Exception as e:
                    pass
        
        # Fallback accept response
        return {
            "action": "send",
            "body": "Perfect! Processing now — I'll have everything ready for you shortly. ✅",
            "cta": "none",
            "rationale": "Merchant accepted; confirming action.",
        }
    
    # ── Neutral reply — compose contextual follow-up ──────────────────────────
    merchant_id = conv.get("merchant_id") or body.merchant_id
    customer_id = conv.get("customer_id") or body.customer_id
    trigger_id = conv.get("trigger_id", "")
    
    merchant = get_ctx("merchant", merchant_id) if merchant_id else None
    trg = get_ctx("trigger", trigger_id) if trigger_id else None
    customer = get_ctx("customer", customer_id) if customer_id else None
    
    if not (merchant and trg):
        return {
            "action": "send",
            "body": "Got it! Kuch aur help chahiye? Just let me know.",
            "cta": "open_ended",
            "rationale": "Missing context; generic continuation.",
        }
    
    category_slug = merchant.get("category_slug", "")
    category = get_ctx("category", category_slug)
    if not category:
        return {
            "action": "send",
            "body": "Noted! Main aapke liye next steps prepare kar rahi hoon.",
            "cta": "open_ended",
            "rationale": "Missing category; generic continuation.",
        }
    
    try:
        result = await compose_message(
            category, merchant, trg, customer, turns,
            is_reply=True, merchant_reply=body.message
        )
        reply_body = result.get("body", "")
        
        # Anti-repetition check
        prev_vera = [t["body"] for t in turns if t["from"] == "vera"]
        if reply_body in prev_vera:
            reply_body = "Theek hai! Aur koi specific question ya detail chahiye to batayein."
        
        turns.append({"from": "vera", "body": reply_body})
        return {
            "action": "send",
            "body": reply_body,
            "cta": result.get("cta", "open_ended"),
            "rationale": result.get("rationale", "Neutral reply; continuing conversation."),
        }
    except Exception as e:
        return {
            "action": "send",
            "body": "Got it — aage ki planning kar rahi hoon. Ek minute.",
            "cta": "open_ended",
            "rationale": f"Composer error; fallback response. Error: {str(e)[:100]}",
        }
