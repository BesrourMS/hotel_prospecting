"""
hotel_agent.py — LangChain agent + Brave Search to find boutique hotels
with a minimal online presence (= good prospects for a booking solution).

Pipeline
  1. discover(): agent runs varied FR/EN/AR searches per city -> candidate hotels
  2. qualify():  agent investigates each candidate with two tools
                   - web_search     (Brave, throttled + cached)
                   - audit_website  (deterministic scrape: booking engine, SSL,
                                     contacts, socials, copyright year...)
                 and returns a validated, evidence-backed HotelLead
  3. score():    the opportunity score is computed in PYTHON from the fields,
                 not by the LLM, so results are comparable between runs

Install
  pip install -U langchain langchain-community langchain-groq \
                 requests beautifulsoup4 pydantic

Env
  BRAVE_API_KEY=...        GROQ_API_KEY=...
  LLM_MODEL (optional)     default: qwen/qwen3.8-27b
  TPM_BUDGET (optional)    default: 6000

Run
  python hotel_agent.py --cities "Sfax,Mahdia,Tozeur,Djerba" --per-city 15 --out leads.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
import traceback
import unicodedata
from collections import OrderedDict, deque
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

import groq
import requests
from bs4 import BeautifulSoup
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_community.tools import BraveSearch
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Config / constants
# --------------------------------------------------------------------------- #

CACHE_FILE = Path(".search_cache.json")
CACHE_MAX_ENTRIES = 400
CACHE_FLUSH_EVERY = 15

ENGINES = {
    "cloudbeds": "Cloudbeds",
    "beds24": "Beds24",
    "siteminder": "SiteMinder",
    "synxis": "SynXis",
    "mews.com": "Mews",
    "littlehotelier": "Little Hotelier",
    "hotelrunner": "HotelRunner",
    "wubook": "WuBook",
    "reservit": "Reservit",
    "d-edge": "D-EDGE",
    "fastbooking": "Fastbooking",
    "bookassist": "Bookassist",
    "simplebooking": "SimpleBooking",
    "roomcloud": "RoomCloud",
    "avirato": "Avirato",
    "hotelogix": "Hotelogix",
    "ezee": "eZee",
    "booking-engine": "generic booking engine",
    "secure-hotel-booking": "Secure Hotel Booking",
}

OTAS = ("booking.com", "tripadvisor", "expedia", "airbnb", "agoda", "hotels.com")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8,ar;q=0.7",
}

# Tunisian mobile / landline patterns (with optional +216 / 00216)
PHONE_RE = re.compile(
    r"(?:\+|00)?216[\s.\-]?(?:(?:2|4|5|9)\d|7[0-9])[\s.\-]?\d{3}[\s.\-]?\d{3}"
    r"|(?:\+|00)?216[\s.\-]?7[1-5][\s.\-]?\d{3}[\s.\-]?\d{3}"
)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# --------------------------------------------------------------------------- #
# Search cache (bounded OrderedDict) + Brave throttle
# --------------------------------------------------------------------------- #

_lock = threading.Lock()
_last_call = 0.0
_cache_writes = 0

def _load_cache() -> OrderedDict[str, str]:
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return OrderedDict(list(data.items())[-CACHE_MAX_ENTRIES:])
    except Exception:
        pass
    return OrderedDict()


_cache: OrderedDict[str, str] = _load_cache()


def _save_cache(force: bool = False) -> None:
    global _cache_writes
    _cache_writes += 1
    if not force and _cache_writes % CACHE_FLUSH_EVERY != 0:
        return
    while len(_cache) > CACHE_MAX_ENTRIES:
        _cache.popitem(last=False)
    try:
        CACHE_FILE.write_text(
            json.dumps(_cache, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        print(f"  [cache] write failed: {e}", file=sys.stderr)


_brave = BraveSearch.from_api_key(
    api_key=os.environ["BRAVE_API_KEY"],
    search_kwargs={"count": 5},
)


class SearchArgs(BaseModel):
    query: str = Field(
        description="The search query string, e.g. 'maison d'hôtes Tunis médina'"
    )


class AuditArgs(BaseModel):
    url: str = Field(description="Full URL of the hotel's own website")


@tool(args_schema=SearchArgs)
def web_search(query: str) -> str:
    """Search the web with Brave. Returns up to 5 results as 'title | url | snippet'.
    Use short, specific queries. Try French, English and Arabic variants."""
    global _last_call

    if query in _cache:
        _cache.move_to_end(query)
        return _cache[query]

    with _lock:  # Brave free tier ≈ 1 req/s
        wait = 1.1 - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            raw = _brave.run(query)
        except Exception as e:
            return f"SEARCH_ERROR: {e}"
        _last_call = time.time()

    try:
        items = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(items, list):
            items = []
        out = (
            "\n".join(
                f"- {i.get('title')} | {i.get('link')} | {(i.get('snippet') or '')[:160]}"
                for i in items
            )
            or "NO_RESULTS"
        )
    except Exception:
        out = (raw if isinstance(raw, str) else str(raw))[:3000]

    _cache[query] = out
    _cache.move_to_end(query)
    _save_cache()
    return out


# --------------------------------------------------------------------------- #
# Website audit
# --------------------------------------------------------------------------- #

def _extract_phones(html: str, text: str, hrefs: list[str]) -> list[str]:
    found: set[str] = set()
    for h in hrefs:
        if h.lower().startswith("tel:"):
            found.add(h[4:].strip())
    for m in PHONE_RE.findall(text):
        found.add(re.sub(r"[\s.\-]", "", m))
    # also scan raw HTML for common patterns
    for m in PHONE_RE.findall(html):
        found.add(re.sub(r"[\s.\-]", "", m))
    return sorted(found)[:5]


def _extract_emails(html: str) -> list[str]:
    return sorted(set(EMAIL_RE.findall(html)))[:5]


@tool(args_schema=AuditArgs)
def audit_website(url: str) -> dict:
    """Fetch a hotel's own website and return OBJECTIVE signals: reachability, https,
    mobile viewport, newest copyright year, embedded booking engine, OTA links,
    emails, phones, WhatsApp and social links. Call this on every candidate website."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(
                url, headers=HEADERS, timeout=12, allow_redirects=True
            )
            break
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    else:
        return {"reachable": False, "error": str(last_err)[:200]}

    html = r.text
    low = html.lower()
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    hrefs = [a["href"] for a in soup.find_all("a", href=True)]

    # Prefer copyright years that look like footer content (last ~30 % of HTML)
    tail = low[int(len(low) * 0.7) :]
    years = [
        int(y)
        for y in re.findall(
            r"(?:©|&copy;|copyright)\D{0,15}((?:19|20)\d{2})", tail
        )
    ]
    if not years:
        years = [
            int(y)
            for y in re.findall(
                r"(?:©|&copy;|copyright)\D{0,15}((?:19|20)\d{2})", low
            )
        ]

    return {
        "reachable": r.status_code < 400,
        "status": r.status_code,
        "final_url": r.url,
        "https": r.url.startswith("https"),
        "title": (
            soup.title.string.strip()
            if soup.title and soup.title.string
            else None
        ),
        "mobile_viewport": 'name="viewport"' in low or "name='viewport'" in low,
        "latest_copyright_year": max(years) if years else None,
        "booking_engines": sorted(
            {v for k, v in ENGINES.items() if k in low}
        ),
        "has_book_button": bool(
            re.search(r"book now|réserver|reservation|احجز|book online", low)
        ),
        "ota_links": sorted(
            {o for o in OTAS if any(o in h.lower() for h in hrefs)}
        ),
        "emails": _extract_emails(html),
        "phones": _extract_phones(html, text, hrefs),
        "whatsapp": [
            h for h in hrefs if "wa.me" in h.lower() or "whatsapp" in h.lower()
        ][:2],
        "facebook": next(
            (h for h in hrefs if "facebook.com" in h.lower()), None
        ),
        "instagram": next(
            (h for h in hrefs if "instagram.com" in h.lower()), None
        ),
        "word_count": len(text.split()),
    }


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class Candidate(BaseModel):
    name: str
    city: str
    source_url: str = Field(description="Page where this hotel was found")


class CandidateList(BaseModel):
    candidates: list[Candidate]


class HotelLead(BaseModel):
    name: str
    city: str
    category: Literal[
        "boutique", "maison_hotes", "small_hotel", "chain_or_large", "other"
    ]
    is_boutique: bool = Field(
        description="Independent, small (<~40 rooms), character/design/charm"
    )
    approx_rooms: Optional[int] = None
    website_url: Optional[str] = Field(
        None, description="Hotel's OWN site, never an OTA page"
    )
    website_status: Literal[
        "none", "social_only", "outdated", "basic", "modern"
    ]
    has_online_booking: bool = Field(
        description="True only if a real booking engine was detected"
    )
    booking_engine: Optional[str] = None
    ota_listings: list[str] = Field(
        default_factory=list, description="e.g. booking.com, tripadvisor"
    )
    email: Optional[str] = None
    phone: Optional[str] = None
    whatsapp: Optional[str] = None
    facebook_url: Optional[str] = None
    instagram_url: Optional[str] = None
    evidence_urls: list[str] = Field(default_factory=list)
    confidence: float = Field(
        ge=0, le=1, description="How sure you are of the classification"
    )
    notes: str = Field(
        "", description="One or two sentences: why this is / is not a good prospect"
    )


# --------------------------------------------------------------------------- #
# TPM limiter
# --------------------------------------------------------------------------- #

class TPMLimiter(BaseCallbackHandler):
    """Blocks before each LLM call so input tokens stay under Groq's per-minute cap."""

    def __init__(self, budget: int):
        self.budget = budget
        self.log: deque[tuple[float, int]] = deque()

    def on_chat_model_start(self, serialized, messages, **kwargs):
        # Conservative estimate: ~2.2 chars/token + overhead for tools/reasoning
        est = (
            sum(len(str(m.content)) for batch in messages for m in batch) // 2
            + 900
        )
        while True:
            now = time.time()
            while self.log and now - self.log[0][0] > 60:
                self.log.popleft()
            used = sum(t for _, t in self.log)
            if not self.log or used + est <= self.budget:
                break
            wait = 61 - (now - self.log[0][0])
            print(
                f"  [throttle] ~{used + est} tokens/min > {self.budget}, "
                f"waiting {wait:.0f}s"
            )
            time.sleep(max(1.0, wait))
        self.log.append((time.time(), est))


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #

MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
llm = ChatGroq(
    model=MODEL,
    temperature=0.5,
    reasoning_format="hidden",
    max_retries=6,
    callbacks=[TPMLimiter(int(os.getenv("TPM_BUDGET", "6000")))],
)

DISCOVERY_PROMPT = """You find boutique hotels, maisons d'hôtes and small charming independent
hotels in a given Tunisian city. Use web_search several times with VARIED queries
(French, English, Arabic; 'maison d'hôtes', 'riad', 'dar', 'boutique hotel',
'petit hôtel de charme', 'hôtel de charme', 'guest house', 'ecolodge'; directory pages,
blogs, Booking/Tripadvisor listings, Facebook pages).

Rules:
- Only return real, independent properties. Skip big chains and resort hotels.
- Names must come from search results; never invent one.
- Deduplicate (same hotel spelled differently = one entry).
- Tool call format: web_search takes exactly one argument named `query`.
- Use at most 4 web_search calls, then give your answer.
"""

QUALIFY_PROMPT = """You qualify ONE hotel as a prospect for an online booking solution.
Ideal prospect: independent boutique / guesthouse, small, with a MINIMAL online presence:
no website, Facebook/Instagram only, outdated site, or a site with no booking engine,
relying on Booking.com / phone / WhatsApp.

Method:
1. web_search for the hotel's own website, Facebook, Instagram, and OTA listings.
2. If you find its own website, ALWAYS call audit_website on it and rely on that output
   (booking_engines empty + old copyright year + no viewport = weak presence).
3. If there is no own website, check whether Facebook/Instagram is its only channel.
4. Classify website_status: none | social_only | outdated | basic | modern.

Hard rules:
- Contact details (email/phone/WhatsApp) must appear in tool output. NEVER guess or invent.
- website_url is the hotel's OWN domain, not booking.com / tripadvisor / facebook.
- Use at most 5 tool calls. Lower confidence when evidence is thin.
- Tool arguments: web_search(query=...), audit_website(url=...). Use these exact names.
"""

discovery_agent = create_agent(
    llm,
    tools=[web_search],
    system_prompt=DISCOVERY_PROMPT,
    response_format=ToolStrategy(CandidateList),
)

qualify_agent = create_agent(
    llm,
    tools=[web_search, audit_website],
    system_prompt=QUALIFY_PROMPT,
    response_format=ToolStrategy(HotelLead),
)


def _invoke(agent, content: str, limit: int, tries: int = 4) -> dict:
    """Retry on common Groq tool-call / rate-limit failures."""
    for attempt in range(1, tries + 1):
        try:
            return agent.invoke(
                {"messages": [{"role": "user", "content": content}]},
                config={"recursion_limit": limit},
            )
        except groq.BadRequestError as e:
            if "tool_use_failed" not in str(e) or attempt == tries:
                raise
            print(f"  [retry {attempt}/{tries - 1}] malformed tool call, retrying")
            time.sleep(1.5 * attempt)
        except groq.APIStatusError as e:
            if e.status_code not in (413, 429) or attempt == tries:
                raise
            print(f"  [rate limit {e.status_code}] waiting 30s")
            time.sleep(30)
    raise RuntimeError("unreachable")


def discover(city: str, n: int) -> list[Candidate]:
    res = _invoke(
        discovery_agent,
        f"Find up to {n} candidates in {city}, Tunisia.",
        limit=20,
    )
    return res["structured_response"].candidates[:n]


def qualify(c: Candidate) -> HotelLead:
    res = _invoke(
        qualify_agent,
        f"Hotel: {c.name}, {c.city}, Tunisia. First seen: {c.source_url}",
        limit=22,
    )
    return res["structured_response"]


# --------------------------------------------------------------------------- #
# Scoring (deterministic) + helpers
# --------------------------------------------------------------------------- #

def score(l: HotelLead) -> int:
    """Return an integer opportunity score roughly in 0–100 range."""
    s = 30 if l.is_boutique else 0
    s += {
        "none": 25,
        "social_only": 22,
        "outdated": 15,
        "basic": 8,
        "modern": 0,
    }[l.website_status]
    s += 0 if l.has_online_booking else 20
    s += 10 if l.ota_listings else 0  # proves demand, pays commissions today
    s += 10 if (l.email or l.phone or l.whatsapp) else 0  # reachable
    s += 5 if l.approx_rooms and 5 <= l.approx_rooms <= 40 else 0
    return int(s * (0.5 + 0.5 * l.confidence))


def _normalize_name(name: str) -> str:
    """Aggressive normalization for deduplication."""
    s = unicodedata.normalize("NFKD", name.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    for noise in (
        "maison d'hotes",
        "maison dhotes",
        "maison d'hôtes",
        "hotel de charme",
        "hôtel de charme",
        "boutique hotel",
        "guest house",
        "guesthouse",
        "riad",
        "dar ",
        "hotel ",
        "hôtel ",
    ):
        s = s.replace(noise, " ")
    s = re.sub(r"\W+", "", s)
    return s


def _lead_to_row(lead: HotelLead) -> dict:
    row = lead.model_dump()
    row["score"] = score(lead)
    row["ota_listings"] = ", ".join(lead.ota_listings)
    row["evidence_urls"] = " | ".join(lead.evidence_urls)
    return row


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="Find boutique hotels with weak online presence in Tunisia"
    )
    ap.add_argument("--cities", required=True, help="comma-separated city names")
    ap.add_argument("--per-city", type=int, default=15)
    ap.add_argument("--out", default="leads.csv")
    ap.add_argument(
        "--jsonl",
        default=None,
        help="optional path for JSONL export (one lead per line)",
    )
    args = ap.parse_args()

    seen: set[str] = set()
    rows: list[dict] = []

    for city in [c.strip() for c in args.cities.split(",") if c.strip()]:
        print(f"\n=== Discovering in {city} ===")
        try:
            candidates = discover(city, args.per_city)
        except Exception as e:
            print(f"[discover failed] {city}: {e}")
            traceback.print_exc()
            continue

        for cand in candidates:
            key = _normalize_name(cand.name) + city.lower()
            if key in seen or len(key) < 4:
                continue
            seen.add(key)

            try:
                lead = qualify(cand)
            except Exception as e:
                print(f"[skip] {cand.name}: {e}")
                continue

            row = _lead_to_row(lead)
            rows.append(row)
            print(
                f"{row['score']:>3}  {lead.name} ({city}) — {lead.website_status}"
            )

    rows.sort(key=lambda r: r["score"], reverse=True)

    if rows:
        with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        if args.jsonl:
            with open(args.jsonl, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _save_cache(force=True)
    print(f"\n{len(rows)} leads -> {args.out}")
    if args.jsonl:
        print(f"JSONL also written to {args.jsonl}")


if __name__ == "__main__":
    main()