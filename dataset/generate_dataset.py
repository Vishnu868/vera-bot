#!/usr/bin/env python3
"""
Expand seed JSON files into the full challenge dataset.

Usage:
    python generate_dataset.py --out ./expanded

Reads from:
    categories/*.json       — already-complete category contexts (5)
    merchants_seed.json     — 10 representative merchants (2 per category)
    customers_seed.json     — 15 representative customers
    triggers_seed.json      — 25 representative triggers

Writes to ./expanded/:
    categories/{slug}.json          (5 files, copied as-is)
    merchants/m_NNN_*.json          (50 files — seeds + 40 generated)
    customers/c_NNN_*.json          (200 files — seeds + 185 generated)
    triggers/trg_NNN_*.json         (100 files — seeds + 75 generated)
    test_pairs.json                 (30 canonical (merchant, trigger) pairs all
                                     candidates produce a message for)

Deterministic — fixed seed, same output for everyone.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

SEED = 20260426  # fixed so every candidate gets the same expanded dataset

# Indian city + locality pool for variation
LOCALITIES = {
    "Delhi": ["Lajpat Nagar", "Saket", "Karol Bagh", "Pitampura", "Dwarka", "Rohini", "Greater Kailash", "Vasant Kunj", "Connaught Place", "Hauz Khas"],
    "Mumbai": ["Andheri West", "Bandra", "Borivali", "Powai", "Lower Parel", "Goregaon", "Thane", "Vile Parle", "Juhu", "Worli"],
    "Bangalore": ["HSR Layout", "Indiranagar", "Whitefield", "Koramangala", "JP Nagar", "Marathahalli", "Bellandur", "Jayanagar", "BTM Layout", "Sarjapur"],
    "Hyderabad": ["Kapra", "Kondapur", "Madhapur", "Banjara Hills", "Jubilee Hills", "Kukatpally", "Gachibowli", "Begumpet", "Secunderabad", "LB Nagar"],
    "Chennai": ["Mylapore", "Adyar", "Velachery", "T Nagar", "Anna Nagar", "Tambaram", "OMR", "Nungambakkam", "Porur", "Besant Nagar"],
    "Pune": ["Aundh", "Baner", "Hadapsar", "Kothrud", "Wakad", "Hinjewadi", "Viman Nagar", "Kharadi", "Pimpri", "Magarpatta"],
    "Chandigarh": ["Sector 17", "Sector 22", "Sector 35", "Mohali", "Panchkula", "Sector 9", "Sector 11", "Manimajra", "Sector 8", "Sector 26"],
    "Jaipur": ["Malviya Nagar", "Vaishali Nagar", "Mansarovar", "Tonk Road", "C-Scheme", "Raja Park", "Civil Lines", "Jhotwara", "Bani Park", "Sodala"],
    "Lucknow": ["Gomti Nagar", "Hazratganj", "Indira Nagar", "Aliganj", "Aminabad", "Vibhuti Khand", "Mahanagar", "Aashiana", "Alambagh", "Janakipuram"],
    "Ahmedabad": ["Satellite", "Bodakdev", "Vastrapur", "Maninagar", "Naranpura", "Bopal", "SG Highway", "Navrangpura", "Thaltej", "Chandkheda"],
}

NAME_BANKS = {
    "dentists": [
        ("Dr. Asha", "Asha Dental Care"),
        ("Dr. Vikram", "Smile Crafters"),
        ("Dr. Neha", "Pearl Dental Studio"),
        ("Dr. Rajan", "City Dental Clinic"),
        ("Dr. Priya", "Family Dental Centre"),
        ("Dr. Sameer", "Bright Smile Dental"),
        ("Dr. Tara", "Crown Dental"),
        ("Dr. Karthik", "Apex Dental Care"),
    ],
    "salons": [
        ("Renu", "Beauty Lounge by Renu"),
        ("Karim", "Karim's Salon"),
        ("Anita", "Anita's Beauty Studio"),
        ("Salim", "Studio Cuts"),
        ("Manish", "Aesthetic Hair Studio"),
        ("Geeta", "Glow Up Salon"),
        ("Paras", "Paras Hair & Beauty"),
        ("Sushma", "The Beauty Bar"),
    ],
    "restaurants": [
        ("Suresh", "Madras Express"),
        ("Anand", "Chai Point Cafe"),
        ("Karim", "Kabab Junction"),
        ("Sandeep", "Tandoor Treats"),
        ("Ravi", "Veg Bowl"),
        ("Imran", "Biryani House"),
        ("Mukesh", "Pizza Spot"),
        ("Lalit", "Family Diner"),
    ],
    "gyms": [
        ("Karan", "Iron Forge Fitness"),
        ("Sneha", "Pulse Studio"),
        ("Akash", "Fit Republic"),
        ("Roshni", "Active Life Gym"),
        ("Vivek", "Strength Co."),
        ("Manisha", "Vyayam Yoga"),
        ("Deepak", "Body Mechanics"),
        ("Pooja", "Bend & Burn"),
    ],
    "pharmacies": [
        ("Anil", "Healthwell Pharmacy"),
        ("Rajesh", "MedPlus Express"),
        ("Sunita", "Reliable Medicos"),
        ("Vinod", "Family Health Pharmacy"),
        ("Bharti", "Wellness Cart"),
        ("Sanjay", "TrueCare Medicos"),
        ("Mohit", "QuickRx Pharmacy"),
        ("Komal", "Daily Care Medicos"),
    ],
}


def load_seeds(seed_dir: Path):
    categories = {}
    for f in (seed_dir / "categories").glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)
            categories[data["slug"]] = data
    with open(seed_dir / "merchants_seed.json") as fp:
        merchants = json.load(fp)["merchants"]
    with open(seed_dir / "customers_seed.json") as fp:
        customers = json.load(fp)["customers"]
    with open(seed_dir / "triggers_seed.json") as fp:
        triggers = json.load(fp)["triggers"]
    return categories, merchants, customers, triggers


def expand_merchants(seeds: list[dict], rnd: random.Random) -> list[dict]:
    """Generate 8 additional merchants per category (10 total per category, 50 overall)."""
    expanded = list(seeds)
    by_cat = {}
    for m in seeds:
        by_cat.setdefault(m["category_slug"], []).append(m)
    next_idx = len(seeds) + 1
    for cat_slug in NAME_BANKS:
        existing = len(by_cat.get(cat_slug, []))
        need = 10 - existing
        for i in range(need):
            owner_first, biz_name = rnd.choice(NAME_BANKS[cat_slug])
            city = rnd.choice(list(LOCALITIES.keys()))
            locality = rnd.choice(LOCALITIES[city])
            mid = f"m_{next_idx:03d}_{owner_first.lower().replace(' ', '_').replace('dr.', 'dr')}_{cat_slug.rstrip('s')}_{city.lower()}"
            views = rnd.randint(400, 6000)
            calls = rnd.randint(2, max(3, views // 80))
            ctr = round(rnd.uniform(0.015, 0.060), 3)
            verified = rnd.random() > 0.25
            sub_status = rnd.choices(["active", "expired", "trial"], weights=[7, 2, 1])[0]
            expanded.append({
                "merchant_id": mid,
                "category_slug": cat_slug,
                "identity": {
                    "name": biz_name, "city": city, "locality": locality,
                    "place_id": f"ChIJ_{locality.upper().replace(' ', '_')}_{cat_slug.upper()}_{next_idx:03d}",
                    "verified": verified,
                    "languages": ["en", "hi"] + (["mr"] if city == "Mumbai" else ["ta"] if city == "Chennai" else ["te"] if city == "Hyderabad" else ["kn"] if city == "Bangalore" else []),
                    "owner_first_name": owner_first,
                    "established_year": rnd.randint(2010, 2023),
                },
                "subscription": {"status": sub_status, "plan": "Pro" if sub_status != "trial" else "Trial",
                                 "days_remaining": rnd.randint(5, 300) if sub_status == "active" else (rnd.randint(1, 14) if sub_status == "trial" else 0),
                                 "days_since_expiry": rnd.randint(7, 90) if sub_status == "expired" else None},
                "performance": {"window_days": 30, "views": views, "calls": calls,
                                "directions": calls * 2 + rnd.randint(0, 30),
                                "ctr": ctr, "leads": rnd.randint(0, calls),
                                "delta_7d": {"views_pct": round(rnd.uniform(-0.30, 0.30), 2),
                                             "calls_pct": round(rnd.uniform(-0.30, 0.30), 2)}},
                "offers": [],
                "conversation_history": [],
                "customer_aggregate": {"total_unique_ytd": rnd.randint(50, 2000)},
                "signals": [],
                "review_themes": [],
            })
            next_idx += 1
    return expanded


def expand_customers(seeds: list[dict], merchants: list[dict], rnd: random.Random) -> list[dict]:
    """Generate ~3-5 customers per merchant up to 200 total."""
    expanded = list(seeds)
    next_idx = len(seeds) + 1
    target_per_merchant = 4
    have_per_merchant = {}
    for c in seeds:
        have_per_merchant[c["merchant_id"]] = have_per_merchant.get(c["merchant_id"], 0) + 1
    customer_names = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Ishaan", "Reyansh", "Aryan", "Ananya", "Aadhya", "Saanvi", "Kavya", "Diya", "Ira", "Myra", "Anika", "Riya", "Tara"]
    for m in merchants:
        cur = have_per_merchant.get(m["merchant_id"], 0)
        for i in range(max(0, target_per_merchant - cur)):
            if next_idx > 200 + len(seeds): break
            name = rnd.choice(customer_names)
            cid = f"c_{next_idx:03d}_{name.lower()}_for_{m['merchant_id']}"
            visits = rnd.randint(1, 12)
            state = rnd.choices(
                ["new", "active", "lapsed_soft", "lapsed_hard", "churned"],
                weights=[1, 4, 2, 1, 1]
            )[0]
            expanded.append({
                "customer_id": cid,
                "merchant_id": m["merchant_id"],
                "identity": {"name": name, "phone_redacted": "<phone>",
                             "language_pref": rnd.choice(["en", "hi-en mix", "hi"]),
                             "age_band": rnd.choice(["20-25", "25-35", "30-40", "40-50", "50-65"])},
                "relationship": {"first_visit": "2025-09-01", "last_visit": "2026-04-01",
                                 "visits_total": visits, "services_received": [],
                                 "lifetime_value": visits * rnd.randint(200, 1500)},
                "state": state,
                "preferences": {"channel": "whatsapp", "reminder_opt_in": rnd.random() > 0.2},
                "consent": {"opted_in_at": "2025-09-01", "scope": ["promotional_offers"]},
            })
            next_idx += 1
    return expanded


def expand_triggers(seeds: list[dict], merchants: list[dict], customers: list[dict], rnd: random.Random) -> list[dict]:
    """Generate ~75 additional triggers spread across kinds + merchants."""
    expanded = list(seeds)
    next_idx = len(seeds) + 1
    additional_kinds = [
        ("research_digest", "external", "merchant", 1),
        ("perf_dip", "internal", "merchant", 3),
        ("perf_spike", "internal", "merchant", 1),
        ("milestone_reached", "internal", "merchant", 1),
        ("dormant_with_vera", "internal", "merchant", 2),
        ("review_theme_emerged", "internal", "merchant", 3),
        ("competitor_opened", "external", "merchant", 2),
        ("festival_upcoming", "external", "merchant", 1),
        ("recall_due", "internal", "customer", 3),
        ("customer_lapsed_soft", "internal", "customer", 3),
        ("appointment_tomorrow", "internal", "customer", 2),
        ("chronic_refill_due", "internal", "customer", 2),
        ("trial_followup", "internal", "customer", 2),
        ("renewal_due", "internal", "merchant", 4),
        ("curious_ask_due", "internal", "merchant", 1),
    ]
    for kind, source, scope, urgency in additional_kinds:
        for _ in range(5):  # 5 of each kind
            if next_idx > 100: break
            m = rnd.choice(merchants)
            cust = None
            if scope == "customer":
                m_customers = [c for c in customers if c["merchant_id"] == m["merchant_id"]]
                if not m_customers: continue
                cust = rnd.choice(m_customers)
            expanded.append({
                "id": f"trg_{next_idx:03d}_{kind}_{m['merchant_id'][:20]}",
                "scope": scope, "kind": kind, "source": source,
                "merchant_id": m["merchant_id"],
                "customer_id": cust["customer_id"] if cust else None,
                "payload": {"placeholder": True, "metric_or_topic": kind},
                "urgency": urgency, "suppression_key": f"{kind}:{m['merchant_id']}:gen_{next_idx}",
                "expires_at": "2026-06-30T00:00:00Z",
            })
            next_idx += 1
    return expanded[:100]


def write_outputs(out_dir: Path, categories, merchants, customers, triggers):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "categories").mkdir(exist_ok=True)
    for slug, data in categories.items():
        with open(out_dir / "categories" / f"{slug}.json", "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    (out_dir / "merchants").mkdir(exist_ok=True)
    for m in merchants:
        with open(out_dir / "merchants" / f"{m['merchant_id']}.json", "w") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)
    (out_dir / "customers").mkdir(exist_ok=True)
    for c in customers:
        with open(out_dir / "customers" / f"{c['customer_id']}.json", "w") as f:
            json.dump(c, f, indent=2, ensure_ascii=False)
    (out_dir / "triggers").mkdir(exist_ok=True)
    for t in triggers:
        with open(out_dir / "triggers" / f"{t['id']}.json", "w") as f:
            json.dump(t, f, indent=2, ensure_ascii=False)


def write_test_pairs(out_dir: Path, triggers, rnd: random.Random):
    """Pick 30 (merchant, trigger) pairs covering all kinds. Same set for everyone."""
    by_kind = {}
    for t in triggers:
        by_kind.setdefault(t["kind"], []).append(t)
    pairs = []
    test_id = 1
    for kind, ts in sorted(by_kind.items()):
        for t in ts[:2]:  # take up to 2 per kind
            pairs.append({"test_id": f"T{test_id:02d}", "trigger_id": t["id"],
                          "merchant_id": t["merchant_id"], "customer_id": t.get("customer_id")})
            test_id += 1
            if len(pairs) >= 30: break
        if len(pairs) >= 30: break
    with open(out_dir / "test_pairs.json", "w") as f:
        json.dump({"pairs": pairs[:30]}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", default=".", help="Directory containing the seed JSON files")
    parser.add_argument("--out", default="./expanded", help="Output directory")
    args = parser.parse_args()

    rnd = random.Random(SEED)
    seed_dir = Path(args.seed_dir).resolve()
    out_dir = Path(args.out).resolve()
    print(f"Reading seeds from {seed_dir}")
    print(f"Writing to {out_dir}")

    categories, m_seeds, c_seeds, t_seeds = load_seeds(seed_dir)
    print(f"  Loaded {len(categories)} categories, {len(m_seeds)} merchant seeds, "
          f"{len(c_seeds)} customer seeds, {len(t_seeds)} trigger seeds")

    merchants = expand_merchants(m_seeds, rnd)
    customers = expand_customers(c_seeds, merchants, rnd)
    triggers = expand_triggers(t_seeds, merchants, customers, rnd)
    print(f"  Expanded to {len(merchants)} merchants, {len(customers)} customers, {len(triggers)} triggers")

    write_outputs(out_dir, categories, merchants, customers, triggers)
    write_test_pairs(out_dir, triggers, rnd)
    print(f"Done. Run: ls {out_dir}")


if __name__ == "__main__":
    main()
