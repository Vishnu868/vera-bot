# 10 Case Studies — What "Good" Looks Like

These case studies serve as anchor examples for the AI judge and as targets for candidates. Each shows:
- The 4-context input
- The good composed message
- The compulsion levers used
- The dimension-by-dimension score (out of 50)

Two cases per category × 5 categories = 10 case studies. Mix of merchant-facing and customer-facing scopes.

---

## Case Study 1 — Dentists / Research Digest (merchant-facing)

**Category**: `dentists`
**Merchant**: Dr. Meera's Dental Clinic, Lajpat Nagar Delhi, CTR 2.1% (below peer 3.0%), 124 high-risk adult patients in roster
**Trigger**: `research_digest` — JIDA Oct 2026 paper: 3-month fluoride recall vs 6-month, 38% caries reduction in high-risk adults
**Customer**: none

**Composed message** (sent as Vera)
```
Dr. Meera, JIDA's Oct issue landed. One item relevant to your high-risk adult
patients — 2,100-patient trial showed 3-month fluoride recall cuts caries
recurrence 38% better than 6-month. Worth a look (2-min abstract). Want me
to pull it + draft a patient-ed WhatsApp you can share?  — JIDA Oct 2026 p.14
```

**Compulsion levers**: source citation (credibility), merchant-specific anchor (her high-risk-adult cohort), reciprocity ("I'll pull it for you"), low-friction CTA, **specificity** (2,100 / 38% / page number).

**Score breakdown** (50/50 = perfect)
| Dimension | Score | Why |
|---|---|---|
| Specificity | 10 | Numbers + source + page reference |
| Category fit | 10 | Clinical vocabulary, peer tone, no overclaim |
| Merchant fit | 10 | "your high-risk adult patients" derived from her customer_aggregate |
| Trigger relevance | 10 | Explicitly references the digest as the reason for messaging |
| Engagement compulsion | 10 | Curiosity + reciprocity + clear next step |

---

## Case Study 2 — Dentists / Recall Reminder (customer-facing)

**Category**: `dentists` (customer-facing voice rules: no medical claims)
**Merchant**: Dr. Meera's Dental Clinic — active offer "Dental Cleaning @ ₹299", available slots Wed 5 Nov 6pm + Thu 6 Nov 5pm
**Trigger**: `recall_due` — Priya's 6-month cleaning recall window opened
**Customer**: Priya — lapsed_soft (5mo since last visit), prefers weekday evenings, hi-en mix

**Composed message** (sent as merchant_on_behalf, from Dr. Meera's WA number)
```
Hi Priya, Dr. Meera's clinic here 🦷 It's been 5 months since your last visit
— your 6-month cleaning recall is due. Apke liye 2 slots ready hain: Wed 5
Nov, 6pm ya Thu 6 Nov, 5pm. ₹299 cleaning + complimentary fluoride. Reply 1
for Wed, 2 for Thu, or tell us a time that works.
```

**Compulsion levers**: name personalization, language-pref match (hi-en mix), specific dates + price + free-add ("complimentary fluoride"), low-friction multi-choice CTA, dental emoji for warmth.

**Score breakdown** (49/50)
| Dimension | Score | Why |
|---|---|---|
| Specificity | 10 | Real slots + real price + concrete recall window |
| Category fit | 10 | No medical claims; warm-clinical tone |
| Merchant fit | 10 | Real offer from catalog; merchant's actual slots |
| Trigger relevance | 10 | Explicit 6-month recall framing |
| Engagement compulsion | 9 | Strong but Reply 1/2 multi-choice is borderline (binary preferred); the "or tell us a time that works" rescues it |

---

## Case Study 3 — Salons / Active Planning (merchant-facing)

**Category**: `salons`
**Merchant**: Studio11 Family Salon, Kapra Hyderabad, calls +20% w/w, retention 62%, peer-leading
**Trigger**: `active_planning_intent` — kids yoga summer camp planning conversation
> *Wait — this is the gym category. Let me redo with a salon trigger.*

**Trigger (corrected)**: `bridal_followup` — Kavya, bride-to-be (wedding 2026-11-08), did Bridal Trial 5 weeks ago, in skin-prep window now
**Customer**: Kavya — new state, wedding date 2026-11-08, opted in for bridal followup

> Note: this is also customer-facing. Treating this as a customer-facing case for variety.

**Composed message** (sent as merchant_on_behalf)
```
Hi Kavya 💍 Lakshmi from Studio11 Kapra here. 196 days to your wedding — perfect
window to start the 30-day skin-prep program before serious bridal bookings
roll in. ₹2,499 covers 4 sessions + a take-home kit. Want me to block your
preferred Saturday 4pm slot for the first session next week?
```

**Compulsion levers**: relationship continuity ("from your trial"), wedding-date specificity, urgency framing (window opens), preference honored (Saturday 4pm), single binary commit.

**Score breakdown** (47/50)
| Dimension | Score | Why |
|---|---|---|
| Specificity | 10 | Days-to-wedding count, program structure, price |
| Category fit | 9 | Warm-practical tone; emojis used appropriately |
| Merchant fit | 9 | References Lakshmi (owner first name) + Studio11 + her trial |
| Trigger relevance | 10 | Explicitly the bridal followup window |
| Engagement compulsion | 9 | Strong but assumes the program exists in catalog (verify in MerchantContext.offers) |

---

## Case Study 4 — Salons / Curious Ask (merchant-facing)

**Category**: `salons`
**Merchant**: Studio11 Family Salon, growing strongly, last Vera touch 3 days ago
**Trigger**: `curious_ask_due` — weekly "what's in demand?" cadence
**Customer**: none

**Composed message** (sent as Vera)
```
Hi Lakshmi! Quick check — what service has been most asked-for this week
at Studio11? I'll turn the answer into a Google post + a 4-line WhatsApp
reply you can use when customers ask about pricing. Takes 5 min.
```

**Compulsion levers**: low-stakes question (no commitment), reciprocity offered up-front (Google post + reply draft), effort externalization (5 min), respects merchant time.

**Score breakdown** (44/50)
| Dimension | Score | Why |
|---|---|---|
| Specificity | 8 | "5 min" anchor; could be sharper with a guess ("Is it the keratin treatment?") |
| Category fit | 9 | Warm tone, fellow-operator register |
| Merchant fit | 8 | Uses owner first name; could reference recent service mix |
| Trigger relevance | 10 | This IS the curious-ask family — exactly the right shape |
| Engagement compulsion | 9 | Asking-the-merchant lever is the strongest Cialdini hook for engaged merchants |

---

## Case Study 5 — Restaurants / IPL Match Day (merchant-facing)

**Category**: `restaurants`
**Merchant**: SK Pizza Junction, Sant Nagar Delhi, trial-tier subscriber, BOGO pizza Tue-Thu active
**Trigger**: `ipl_match_today` — DC vs MI at Arun Jaitley Stadium, 7:30pm match today (Sat — *not* a weeknight)
**Customer**: none

**Composed message** (sent as Vera)
```
Quick heads-up Suresh — DC vs MI at Arun Jaitley tonight, 7:30pm. Important:
Saturday IPL matches usually shift -12% restaurant covers (people watch at
home). Skip the match-night promo today; instead push your BOGO pizza
(already active) as a delivery-only Saturday special. Want me to draft the
Swiggy banner + an Insta story? Live in 10 min.
```

**Compulsion levers**: counter-intuitive specific data ("Saturday IPL = -12% covers"), saves merchant from a bad decision, leverages existing offer, concrete deliverables (Swiggy banner + Insta story), 10-min commitment cap.

**Score breakdown** (50/50)
| Dimension | Score | Why |
|---|---|---|
| Specificity | 10 | Match details, percentage delta, time-bound |
| Category fit | 10 | Operator-to-operator voice ("covers", "match-night") |
| Merchant fit | 10 | Existing BOGO referenced; Swiggy/Insta channels appropriate |
| Trigger relevance | 10 | Direct response to IPL trigger with informed nuance |
| Engagement compulsion | 10 | Loss aversion (-12%) + existing-offer leverage + 10-min effort cap |

This case is illustrative because it shows the bot **adding value beyond the trigger itself** — the trigger said "IPL match today"; the bot interpreted it correctly and gave a contrarian recommendation.

---

## Case Study 6 — Restaurants / Active Planning Intent (merchant-facing)

**Category**: `restaurants`
**Merchant**: Mylari South Indian Cafe, Indiranagar Bangalore — high-volume, weekday lunch thali doing 18 orders/day
**Trigger**: `active_planning_intent` — merchant asked about a corporate-bulk thali package; explicitly said "yes good idea, what would it look like"
**Customer**: none

**Composed message** (sent as Vera, in direct conversation)
```
Suresh, here's a starter version — you can edit:

Mylari Corporate Thali — for offices in Indiranagar
- 10 thalis @ ₹125 each (₹25 off retail) + free delivery
- 25 thalis @ ₹115 each + 2 free filter coffees
- 50+: ₹105 each + 1 free dosa platter
- WhatsApp the day-before by 5pm; we deliver between 12:30-1pm

3 offices in Indiranagar are in your delivery radius (Embassy Tech, RMZ Eco,
Sigma Soft). Want me to draft a 3-line WhatsApp to send their facilities
managers?
```

**Compulsion levers**: complete drafted artifact (no extra work for merchant), tiered pricing matches B2B logic, concrete radius + named buildings, follow-on offer to handle outreach.

**Score breakdown** (49/50)
| Dimension | Score | Why |
|---|---|---|
| Specificity | 10 | Tier prices, radius, building names, time windows |
| Category fit | 10 | Operator language ("delivery radius", "facilities managers") |
| Merchant fit | 10 | Indiranagar locality + filter-coffee dosa nods to South Indian cafe identity |
| Trigger relevance | 10 | Direct continuation of merchant's planning intent |
| Engagement compulsion | 9 | Strong but assumes building data is in MerchantContext or fabricates — judge will check |

---

## Case Study 7 — Gyms / Seasonal Dip Reframe (merchant-facing)

**Category**: `gyms`
**Merchant**: PowerHouse Fitness, HSR Layout Bangalore, views -30% w/w (April seasonal drop), 245 active members
**Trigger**: `seasonal_perf_dip` — expected April-June low; flagged as not-a-problem
**Customer**: none

**Composed message** (sent as Vera)
```
Karthik, your views are down 30% this week — but I want to flag this is the
normal April-June acquisition lull (every metro gym sees -25 to -35% in this
window). Action: skip ad spend now, save it for Sept-Oct when conversion is
2x. For now, focus retention on your 245 members. Want me to draft a "summer
attendance challenge" to keep them through the dip?
```

**Compulsion levers**: anxiety pre-emption (the dip is normal), data anchor (-25 to -35%), reframe as opportunity (save spend), specific member count, action proposal.

**Score breakdown** (48/50)
| Dimension | Score | Why |
|---|---|---|
| Specificity | 10 | Percentage range, months, member count |
| Category fit | 10 | Coach-to-operator tone; uses "ad spend", "conversion" correctly |
| Merchant fit | 10 | His exact dip number + his member count + HSR locality (peer benchmark) |
| Trigger relevance | 10 | Directly addresses the seasonal dip with reframe |
| Engagement compulsion | 8 | Could be sharper — "save it for Sept-Oct" is a bit hand-wavy without numbers |

---

## Case Study 8 — Gyms / Customer Lapse Winback (customer-facing)

**Category**: `gyms` (customer-facing voice: warm, no-shame, no guilt-trip)
**Merchant**: PowerHouse Fitness — first month @ ₹499 offer active
**Trigger**: `customer_lapsed_hard` — Rashmi, 57 days since last visit, previous focus was weight loss, was a 5-month member
**Customer**: Rashmi

**Composed message** (sent as merchant_on_behalf)
```
Hi Rashmi 👋 Karthik from PowerHouse here. It's been about 8 weeks — happens
to most members at some point, no judgment. We've added a Tue/Thu evening
HIIT class that fits weight-loss goals well (45 min, 6:30pm). Want me to
hold a free trial spot for you next Tue, 30 Apr? Reply YES — no commitment,
no auto-charge.
```

**Compulsion levers**: name + warmth, no-shame framing, addresses her past goal (weight loss), specific new offering matches goal, no-commitment trial, single binary CTA.

**Score breakdown** (50/50)
| Dimension | Score | Why |
|---|---|---|
| Specificity | 10 | Class name, time, day, date, duration |
| Category fit | 10 | Coach voice, no shame, evidence-based |
| Merchant fit | 10 | Karthik (owner first name), PowerHouse, real new class |
| Trigger relevance | 10 | Explicitly addresses lapse without being needy |
| Engagement compulsion | 10 | "no judgment" + "no commitment, no auto-charge" removes 2 common barriers in one line |

---

## Case Study 9 — Pharmacies / Compliance Alert (merchant-facing)

**Category**: `pharmacies`
**Merchant**: Apollo Health Plus Pharmacy, Malviya Nagar Jaipur — engaged, has 240 chronic-Rx customers
**Trigger**: `supply_alert` — voluntary recall on atorvastatin batches AT2024-1102 + AT2024-1108 by Mfr Z
**Customer**: none

**Composed message** (sent as Vera)
```
Ramesh, urgent: voluntary recall on 2 atorvastatin batches (AT2024-1102,
AT2024-1108) by Mfr Z — sub-potency, no safety risk, but customers should be
informed for replacement. Pulled your repeat-Rx list: 22 of your chronic-Rx
customers were dispensed these batches in last 90 days. Want me to draft
their WhatsApp note + the replacement-pickup workflow?
```

**Compulsion levers**: urgency + specificity (batch numbers), risk-but-bounded framing, **derived count from merchant data** (22 of 240), end-to-end workflow offer (note + pickup).

**Score breakdown** (50/50)
| Dimension | Score | Why |
|---|---|---|
| Specificity | 10 | Batch numbers, manufacturer, count of affected customers |
| Category fit | 10 | Trustworthy-precise tone; correct use of "sub-potency" without alarming |
| Merchant fit | 10 | Pulls from his customer aggregate; uses his name |
| Trigger relevance | 10 | Direct compliance action |
| Engagement compulsion | 10 | Urgency + clear offer of complete artifact |

---

## Case Study 10 — Pharmacies / Chronic Refill Reminder (customer-facing)

**Category**: `pharmacies` (customer-facing voice: trustworthy, precise, respectful of senior)
**Merchant**: Apollo Health Plus Pharmacy — free home delivery >₹499, senior 15% off active
**Trigger**: `chronic_refill_due` — Mr. Sharma's metformin/atorvastatin/telmisartan run out 2026-04-28
**Customer**: Mr. Sharma — 65-75 age band, senior citizen, channel via son's WhatsApp, address saved, opted in

**Composed message** (sent as merchant_on_behalf, addressed to son)
```
Namaste — Apollo Health Plus Malviya Nagar yahan. Sharma ji ki 3 monthly
medicines (metformin, atorvastatin, telmisartan) 28 April ko khatam hongi.
Same dose, same brand pack ready hai. Senior discount 15% applied — total
₹1,420 (₹240 saved). Free home delivery to saved address by 5pm tomorrow.
Reply CONFIRM to dispatch, or call 9876543210 if any change in dosage.
```

**Compulsion levers**: namaste salutation (respectful), full molecule names (precision), specific date, total + savings shown clearly, two-channel option (reply OR call), senior-citizen norms honored.

**Score breakdown** (49/50)
| Dimension | Score | Why |
|---|---|---|
| Specificity | 10 | Three molecule names, exact date, total + savings, time window |
| Category fit | 10 | Trustworthy-precise voice; namaste salutation appropriate |
| Merchant fit | 10 | Apollo's actual offers (free delivery, senior 15%); Malviya Nagar locality |
| Trigger relevance | 10 | Refill due date is the central anchor |
| Engagement compulsion | 9 | Strong; could nudge with "stocks may take 24h" if scarcity were real |

---

## Cross-case patterns the judge looks for

Reading the 10 cases together, here are the patterns that consistently score 9-10/10:

1. **Source citation when claiming research/compliance** — JIDA p.14, DCI circular, batch numbers. No citation = score capped at 7.
2. **Numbers from the contexts, not invented** — "22 of your chronic-Rx customers" is computed from the merchant's customer_aggregate; "245 active members" is from MerchantContext directly. Numbers without provenance get scored as fabrication.
3. **Owner/merchant first name when present** — Dr. Meera, Suresh, Karthik, Ramesh. Generic "Hi" loses 1 point on merchant fit.
4. **Single most important next step framed as low-friction commitment** — "Want me to draft X? Live in 10 min" / "Reply YES — no commitment, no auto-charge". Multi-action asks dilute.
5. **Customer-facing messages honor language preference + relationship state** — Hindi-English mix for Priya, namaste for Mr. Sharma's son. Treating every customer the same loses 2 points on customer fit.
6. **Domain-specific vocabulary used correctly** — "covers", "AOV", "sub-potency", "fluoride varnish", "ad spend", "conversion". Wrong vocabulary or absent vocabulary signals the bot didn't actually use the CategoryContext.voice.
7. **The bot adds judgment, not just templating** — Case Study 5 (IPL) shows the bot recommending *not* to push the IPL promo on a Saturday. That kind of contrarian, data-informed call is the highest signal of category understanding.
8. **The conversation_id is meaningful** — `conv_priya_recall_2026_11` is good (decodable, resumable). `conv_001` is acceptable. UUIDs without context lose nothing but help nothing.
9. **The rationale field is concise and reflects actual reasoning** — judge cross-checks rationale against the message; mismatch = penalty.
10. **No repetition, no fabrication** — these are the operational floor. Any of them in the message and the case is capped at 5/dimension regardless of quality.

---

## How the judge uses these cases

For each submission, the judge LLM:
1. Reads the candidate's composition for the same (category, merchant, trigger, customer) tuple.
2. Compares against the case-study output above.
3. Scores each of the 5 dimensions on a 0-10 scale, citing what's better/worse.
4. Aggregates into the per-test-pair score.

Candidates can review these cases as a north star, but **direct copying the body text of a case study counts as plagiarism** — the judge runs a similarity check on submissions vs the case studies and penalizes near-duplicates.

The cases are meant to teach the *shape* of good output: specificity, category fit, merchant fit, trigger relevance, compulsion. Your wording must be your own.
