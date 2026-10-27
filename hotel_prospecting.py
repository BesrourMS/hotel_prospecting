#!/usr/bin/env python3
"""McZEN Tunisia Hotel Prospecting Agent — improved single-file MVP.

Discovers candidate hotel websites with Brave Search, audits public pages,
and exports an evidence-oriented CSV. A missing public signal is never treated
as proof that a hotel does not use a technology.

Install:
    python -m pip install requests

Windows PowerShell:
    $env:BRAVE_SEARCH_API_KEY="YOUR_KEY"
    python hotel_prospecting_improved.py --max-results 30

Audit an existing CSV:
    python hotel_prospecting_improved.py --input hotels.csv

Optional:
    python hotel_prospecting_improved.py --max-results 50 --max-searches 80
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse, urldefrag

try:
    import requests
except ImportError:
    print("Missing dependency. Install it with: python -m pip install requests", file=sys.stderr)
    raise SystemExit(1)


APP_NAME = "McZEN Tunisia Hotel Prospecting Agent"
SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
USER_AGENT = "Mozilla/5.0 (compatible; McZENHotelProspector/0.2; public website audit)"
TIMEOUT = 15
MAX_HTML_CHARS = 1_500_000
MAX_AUDIT_PAGES = 8

CITIES = [
    "Tunis", "Sfax", "Sousse", "Monastir", "Mahdia", "Hammamet", "Nabeul",
    "Djerba", "Houmt Souk", "Midoun", "Zarzis", "Gabes", "Kairouan", "Bizerte",
    "Tabarka", "Tozeur", "Douz", "Gafsa", "Kef", "Beja", "Ain Draham", "Kebili",
    "Tataouine", "Kasserine", "Medenine", "La Marsa", "Gammarth", "Mahres",
]
QUERIES = [
    ("fr", "hôtel indépendant site officiel"),
    ("fr", "hôtel familial réservation directe contact"),
    ("fr", "hôtel de charme site officiel"),
    ("fr", "hôtel boutique chambres contact"),
    ("en", "hotel official website direct booking"),
    ("en", "independent hotel official website"),
    ("ar", "فندق موقع رسمي"),
    ("ar", "فندق حجز مباشر"),
]

# Known aggregators/social platforms are excluded from the HOTEL pipeline.
# They may still be useful in a separate partnership pipeline.
EXCLUDED_DOMAIN_PARTS = (
    "booking.com", "tripadvisor.", "facebook.com", "instagram.com",
    "expedia.", "hotels.com", "agoda.", "trivago.", "youtube.com",
    "wikipedia.org", "momondo.", "kayak.", "skyscanner.", "tiktok.com",
    "linkedin.com", "google.com", "maps.google.", "yelp.", "tunisietourisme.",
)

DIRECTORY_TERMS = (
    "travel agency", "agence de voyage", "tour operator", "comparateur de prix",
    "annuaire des hôtels", "annuaire hotel", "hotel booking portal",
    "réservation de voyage", "vols et hôtels", "flight and hotel",
)
HOTEL_TERMS = (
    "hotel", "hôtel", "resort", "riad", "guest house", "maison d'hôtes",
    "maison d'hotes", "auberge", "chambres d'hôtes", "chambres d'hotes",
    "فندق", "منتجع",
)

BOOKING_PATTERNS = [
    r"check availability", r"availability calendar", r"book now",
    r"book your stay", r"booking engine", r"réserver maintenant",
    r"verifier les disponibilites", r"vérifier les disponibilités",
    r"moteur de réservation", r"moteur de reservation",
    r"reservation widget", r"booking widget",
]
PAYMENT_PATTERNS = [
    r"stripe", r"paypal", r"paymee", r"konnect", r"flouci",
    r"checkout\.com", r"payment gateway", r"paiement en ligne",
    r"payer en ligne", r"pay online", r"secure payment",
]
CHANNEL_PATTERNS = [
    r"siteminder", r"site.?minder", r"channel manager",
    r"channelmanager", r"hotelrunner", r"synxis", r"rate.?gain",
    r"ezee", r"cloudbeds",
]
PMS_PATTERNS = [
    r"mews\.com", r"opera cloud", r"roommaster", r"protel",
    r"hotelogix", r"innroad", r"property management system",
    r"\bpms software\b", r"cloudbeds",
]
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []
        self.text_parts = []
        self.title_parts = []
        self.in_title = False
        self.forms = 0
        self.inputs = []
        self.meta = []
        self.scripts = []
        self.phones = []

    @property
    def title(self):
        return " ".join(self.title_parts).strip()

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        tag = tag.lower()

        if tag == "title":
            self.in_title = True
        elif tag == "a" and a.get("href"):
            self.links.append((
                a["href"],
                a.get("aria-label", "") or a.get("title", "") or "",
                a.get("href", ""),
            ))
        elif tag == "form":
            self.forms += 1
        elif tag == "input":
            self.inputs.append((
                (a.get("type") or "").lower(),
                (a.get("name") or "").lower(),
                (a.get("placeholder") or "").lower(),
            ))
        elif tag == "meta":
            self.meta.append(a)
        elif tag == "script" and a.get("src"):
            self.scripts.append(a["src"])
        elif tag == "a" and (a.get("href") or "").lower().startswith("tel:"):
            self.phones.append(a["href"][4:])

    def handle_endtag(self, tag):
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data):
        value = data.strip()
        if not value:
            return
        self.text_parts.append(value)
        if self.in_title:
            self.title_parts.append(value)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_url(value):
    value = (value or "").strip()
    if not value:
        return ""
    if not re.match(r"^https?://", value, re.I):
        value = "https://" + value
    try:
        parsed = urlparse(value)
        if not parsed.hostname or parsed.scheme.lower() not in ("http", "https"):
            return ""
        # Do not keep credentials, query strings or paths for the canonical homepage.
        host = parsed.hostname.lower()
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme.lower()}://{host}{port}/"
    except ValueError:
        return ""


def domain_key(url):
    try:
        host = (urlparse(url).hostname or "").lower()
        return host[4:] if host.startswith("www.") else host
    except ValueError:
        return ""


def is_excluded_domain(host):
    return any(part in host for part in EXCLUDED_DOMAIN_PARTS)


def classify_business(title, description, domain, page_text=""):
    text = f"{title} {description} {domain} {page_text[:5000]}".lower()
    if any(term in text for term in DIRECTORY_TERMS):
        return "directory_or_travel_business"
    if any(term in text for term in HOTEL_TERMS):
        return "possible_hotel"
    return "unknown"


def brave_search(session, api_key, query, language="fr", count=10):
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
        "User-Agent": USER_AGENT,
    }
    params = {
        "q": query,
        "count": min(max(int(count), 1), 20),
        "country": "ALL",
        "search_lang": language,
    }
    try:
        response = session.get(SEARCH_URL, headers=headers, params=params, timeout=TIMEOUT)
        if response.status_code == 429:
            print("[search] HTTP 429 rate limit; pausing discovery.", file=sys.stderr)
            return []
        if not response.ok:
            print(
                f"[search] HTTP {response.status_code} for {query!r}: "
                f"{response.text[:700].replace(chr(10), ' ')}",
                file=sys.stderr,
            )
            return []
        payload = response.json()
        return payload.get("web", {}).get("results", []) or []
    except (requests.RequestException, ValueError) as exc:
        print(f"[search] Failed for {query!r}: {exc}", file=sys.stderr)
        return []


def discover(api_key, max_results, max_searches=80, delay=0.3):
    """Collect candidates across cities instead of stopping at the first N results."""
    session = requests.Session()
    found = {}
    search_count = 0

    # Interleave cities and query types so early cities do not dominate.
    jobs = [
        (city, lang, query)
        for query_index in range(len(QUERIES))
        for city in CITIES
        for lang, query in [QUERIES[query_index]]
    ]

    for city, language, query_text in jobs:
        if search_count >= max_searches:
            break
        query = f"{query_text} {city} Tunisie"
        items = brave_search(session, api_key, query, language=language, count=10)
        search_count += 1

        for item in items:
            raw_url = item.get("url", "")
            url = normalize_url(raw_url)
            host = domain_key(url)
            title = (item.get("title") or "").strip()
            description = (item.get("description") or "").strip()

            if not url or not host or not title or is_excluded_domain(host):
                continue

            if host not in found:
                found[host] = {
                    "hotel_name": "",
                    "website": url,
                    "city": city,
                    "discovery_query": query,
                    "discovery_title": title,
                    "discovery_snippet": description,
                    "discovery_sources": [],
                    "business_type": classify_business(title, description, host),
                }
            else:
                # Keep the first city but collect more evidence and better descriptions.
                if len(description) > len(found[host]["discovery_snippet"]):
                    found[host]["discovery_snippet"] = description

            found[host]["discovery_sources"].append({
                "query": query,
                "title": title,
                "url": raw_url,
            })

        if delay > 0:
            time.sleep(delay)

    candidates = list(found.values())
    # Prefer likely hotels, then uncertain businesses; directories are retained
    # for transparency but sorted last and marked for review.
    candidates.sort(key=lambda row: (
        row["business_type"] == "directory_or_travel_business",
        row["business_type"] == "unknown",
        row["city"].lower(),
        row["website"],
    ))
    print(f"[search] Completed {search_count} searches; found {len(candidates)} unique domains.")
    return candidates[:max_results]


def fetch_page(session, url):
    try:
        response = session.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            timeout=TIMEOUT,
            allow_redirects=True,
        )
        content_type = response.headers.get("content-type", "")
        body = response.text[:MAX_HTML_CHARS]
        if "html" not in content_type.lower() and not body.lstrip().lower().startswith(
            ("<!doctype html", "<html", "<head")
        ):
            return {
                "status": response.status_code,
                "url": response.url,
                "parser": PageParser(),
                "html": "",
                "error": f"non_html_content_type:{content_type}",
            }
        parser = PageParser()
        parser.feed(body)
        return {
            "status": response.status_code,
            "url": response.url,
            "parser": parser,
            "html": body,
            "error": "",
        }
    except requests.RequestException as exc:
        return {"status": 0, "url": url, "parser": PageParser(), "html": "", "error": str(exc)}


def find_evidence(patterns, pages, category):
    matches = []
    for page in pages:
        corpus = (page["html"] + " " + " ".join(page["parser"].text_parts)).lower()
        for pattern in patterns:
            match = re.search(pattern, corpus, re.I)
            if match:
                excerpt = re.sub(r"\s+", " ", corpus[max(0, match.start()-90):match.end()+120])
                matches.append({
                    "pattern": pattern,
                    "url": page["url"],
                    "excerpt": excerpt[:240],
                })
                break
    if not matches:
        return {
            "status": "not_detected",
            "summary": "No qualifying public signal found; absence is not confirmed",
            "url": "",
            "details": "",
        }
    return {
        "status": "detected",
        "summary": f"{category} signal detected: {matches[0]['pattern']}",
        "url": matches[0]["url"],
        "details": json.dumps(matches[:5], ensure_ascii=False),
    }


def normalize_tunisian_phone(value):
    digits = re.sub(r"\D", "", value or "")
    if digits.startswith("00216"):
        digits = digits[5:]
    elif digits.startswith("216") and len(digits) == 11:
        digits = digits[3:]
    if len(digits) != 8:
        return ""
    # Common Tunisian mobile/fixed prefixes; retain this as a heuristic.
    if digits[0] not in "234579":
        return ""
    return "+216" + digits


def extract_contacts(pages, website_host):
    emails = []
    phones = []
    for page in pages:
        host = domain_key(page["url"])
        html = page["html"]
        visible = " ".join(page["parser"].text_parts)

        for email in EMAIL_RE.findall(html + " " + visible):
            email = email.strip(".,;:()[]{}<>").lower()
            if any(x in email for x in (
                "example.com", "sentry.io", "wixpress.com", "cloudflare.com",
            )):
                continue
            if email not in emails:
                emails.append(email)

        raw_phones = list(page["parser"].phones)
        raw_phones.extend(re.findall(
            r"(?<!\w)(?:\+216|00216)?[\s().-]*[2-9](?:[\s().-]*\d){7}(?!\w)",
            visible,
        ))
        for raw in raw_phones:
            phone = normalize_tunisian_phone(raw)
            if phone and phone not in phones:
                phones.append(phone)

    # Prefer contact details hosted on the same domain; third-party script emails
    # can still be present in HTML, so callers should treat these as unverified.
    return emails[:10], phones[:10]


def audit(hotel, delay=0.2):
    website = normalize_url(hotel.get("website") or hotel.get("url") or "")
    if not website:
        return base_result(hotel, "", "invalid_url", "No valid website URL supplied")

    session = requests.Session()
    base = fetch_page(session, website)
    if not base or base["status"] == 0 or not base["html"]:
        note = (base or {}).get("error") or "Homepage could not be fetched as HTML"
        result = base_result(hotel, website, "unreachable_or_blocked", note)
        result["http_status"] = (base or {}).get("status", 0)
        return result

    root = base["url"]
    root_host = domain_key(root)
    pages = [base]
    candidate_urls = []

    # Prioritize actual internal links. Fixed paths are only fallback candidates.
    link_words = re.compile(
        r"book|booking|reserv|contact|chambre|room|payment|paiement|checkout|"
        r"about|hotel|hébergement|hebergement|tarif|price|disponibil",
        re.I,
    )
    for href, label, raw_href in base["parser"].links:
        absolute = urljoin(root, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https"):
            continue
        if domain_key(absolute) == root_host and link_words.search(
            f"{href} {label} {raw_href}"
        ):
            candidate_urls.append(urldefrag(absolute)[0])

    candidate_urls.extend(urljoin(root, path) for path in (
        "/booking", "/book", "/reservation", "/reservations", "/contact",
        "/rooms", "/chambres", "/contact-us", "/fr/reservation",
    ))

    seen = {urldefrag(root)[0].rstrip("/")}
    for url in candidate_urls:
        clean_url = urldefrag(url)[0]
        if clean_url.rstrip("/") in seen:
            continue
        seen.add(clean_url.rstrip("/"))
        page = fetch_page(session, clean_url)
        if page and page["status"] and page["status"] < 500 and page["html"]:
            pages.append(page)
        if len(pages) >= MAX_AUDIT_PAGES:
            break
        if delay > 0:
            time.sleep(delay)

    visible_text = " ".join(
        " ".join(page["parser"].text_parts) for page in pages
    )
    title = base["parser"].title
    meta_text = " ".join(
        str(meta.get("content", ""))
        for page in pages for meta in page["parser"].meta
    )
    discovery_title = hotel.get("discovery_title", "")
    discovery_snippet = hotel.get("discovery_snippet", "")
    business_type = classify_business(
        discovery_title or title,
        discovery_snippet + " " + meta_text,
        root_host,
        visible_text,
    )

    booking = find_evidence(BOOKING_PATTERNS, pages, "booking")
    payment = find_evidence(PAYMENT_PATTERNS, pages, "payment")
    channel = find_evidence(CHANNEL_PATTERNS, pages, "channel manager")
    pms = find_evidence(PMS_PATTERNS, pages, "PMS")
    emails, phones = extract_contacts(pages, root_host)

    # Do not reward absence. Score only positive, supportable qualification.
    opportunity_score = 0
    reasons = []

    if business_type == "possible_hotel":
        opportunity_score += 25
        reasons.append("Hotel-related identity signals found")
    elif business_type == "directory_or_travel_business":
        reasons.append("Not classified as an individual hotel; manual review required")
    else:
        reasons.append("Business identity remains uncertain")

    if booking["status"] == "not_detected":
        reasons.append("No direct-booking signal detected; gap not confirmed")
    if payment["status"] == "not_detected":
        reasons.append("No online-payment signal detected; gap not confirmed")
    if channel["status"] == "not_detected":
        reasons.append("No public channel-manager signal; private setup unknown")
    if pms["status"] == "not_detected":
        reasons.append("No public PMS signal; private setup unknown")

    if emails or phones:
        opportunity_score += 10
        reasons.append("Public contact candidate found")

    # A basic-looking site is a weak signal, not proof of a commercial need.
    if len(visible_text) < 1000 or len(base["parser"].links) < 5:
        reasons.append("Website appears limited by a basic heuristic; review manually")

    # Cap preliminary score because the script cannot confirm actual business need.
    opportunity_score = min(opportunity_score, 35)

    pages_with_200 = sum(page["status"] == 200 for page in pages)
    has_evidence = any(
        item["status"] == "detected" for item in (booking, payment, channel, pms)
    )
    if business_type == "possible_hotel" and pages_with_200 >= 2 and (emails or phones):
        confidence = "medium"
    else:
        confidence = "low"
    if business_type == "possible_hotel" and pages_with_200 >= 3 and has_evidence:
        confidence = "medium"

    evidence_urls = list(dict.fromkeys(
        item["url"] for item in (booking, payment, channel, pms) if item["url"]
    ))
    email_text = "; ".join(emails)
    phone_text = "; ".join(phones)

    return {
        "hotel_name": hotel.get("hotel_name") or title or discovery_title or root_host,
        "city": hotel.get("city", ""),
        "website": website,
        "final_url": root,
        "business_type": business_type,
        "identity_status": "probable" if business_type == "possible_hotel" else "unverified",
        "http_status": base["status"],
        "email": email_text,
        "phone": phone_text,
        "contact_source": "public website HTML/text" if emails or phones else "",
        "contact_verified": "no",
        "booking_status": booking["status"],
        "payment_status": payment["status"],
        "channel_manager_status": channel["status"],
        "pms_status": pms["status"],
        "opportunity_score": opportunity_score,
        "lead_score": opportunity_score,  # compatibility with older CSV consumers
        "lead_reasons": "; ".join(reasons),
        "evidence_confidence": confidence,
        "confidence": confidence,  # compatibility with older CSV consumers
        "evidence_urls": "; ".join(evidence_urls),
        "booking_evidence": booking["summary"],
        "booking_evidence_url": booking["url"],
        "booking_evidence_details": booking["details"],
        "payment_evidence": payment["summary"],
        "payment_evidence_url": payment["url"],
        "payment_evidence_details": payment["details"],
        "channel_evidence": channel["summary"],
        "channel_evidence_url": channel["url"],
        "channel_evidence_details": channel["details"],
        "pms_evidence": pms["summary"],
        "pms_evidence_url": pms["url"],
        "pms_evidence_details": pms["details"],
        "pages_checked": len(pages),
        "discovery_query": hotel.get("discovery_query", ""),
        "discovery_title": discovery_title,
        "discovery_snippet": discovery_snippet,
        "discovery_sources": json.dumps(hotel.get("discovery_sources", []), ensure_ascii=False),
        "audit_status": "completed" if pages_with_200 else "partial",
        "audit_errors": "; ".join(
            f"{page['url']}: {page['error']}"
            for page in pages if page.get("error")
        ),
        "audited_at_utc": utc_now(),
    }


def base_result(hotel, website, status, note):
    row = {field: "" for field in FIELDS}
    row.update({
        "hotel_name": hotel.get("hotel_name") or hotel.get("name", ""),
        "city": hotel.get("city", ""),
        "website": website,
        "final_url": "",
        "business_type": hotel.get("business_type", "unknown"),
        "identity_status": "unverified",
        "http_status": 0,
        "email": "",
        "phone": "",
        "contact_source": "",
        "contact_verified": "no",
        "booking_status": "unknown",
        "payment_status": "unknown",
        "channel_manager_status": "unknown",
        "pms_status": "unknown",
        "opportunity_score": 0,
        "lead_score": 0,
        "lead_reasons": note,
        "evidence_confidence": "low",
        "confidence": "low",
        "evidence_urls": website,
        "booking_evidence": note,
        "payment_evidence": note,
        "channel_evidence": note,
        "pms_evidence": note,
        "discovery_query": hotel.get("discovery_query", ""),
        "discovery_title": hotel.get("discovery_title", ""),
        "discovery_snippet": hotel.get("discovery_snippet", ""),
        "discovery_sources": json.dumps(hotel.get("discovery_sources", []), ensure_ascii=False),
        "audit_status": status,
        "audit_errors": note,
        "audited_at_utc": utc_now(),
    })
    return row


FIELDS = [
    "hotel_name", "city", "website", "final_url", "business_type",
    "identity_status", "http_status", "email", "phone", "contact_source",
    "contact_verified", "booking_status", "payment_status",
    "channel_manager_status", "pms_status", "opportunity_score", "lead_score",
    "lead_reasons", "evidence_confidence", "confidence", "evidence_urls",
    "booking_evidence", "booking_evidence_url", "booking_evidence_details",
    "payment_evidence", "payment_evidence_url", "payment_evidence_details",
    "channel_evidence", "channel_evidence_url", "channel_evidence_details",
    "pms_evidence", "pms_evidence_url", "pms_evidence_details",
    "pages_checked", "discovery_query", "discovery_title", "discovery_snippet",
    "discovery_sources", "audit_status", "audit_errors", "audited_at_utc",
]


def load_input(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Input CSV is empty or missing a header row")
        rows = []
        for row in reader:
            lower = {str(k).strip().lower(): (v or "").strip() for k, v in row.items() if k}
            website = (
                lower.get("website") or lower.get("url") or lower.get("domain")
            )
            if not website:
                continue
            rows.append({
                "hotel_name": lower.get("hotel_name") or lower.get("name") or lower.get("hotel", ""),
                "city": lower.get("city", ""),
                "website": website,
                "discovery_query": "input CSV",
                "discovery_title": lower.get("hotel_name") or lower.get("name") or "",
                "discovery_snippet": "",
                "discovery_sources": [],
                "business_type": "unknown",
            })
        return rows


def export_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--input", help="CSV with website/url/domain and optional name/city columns")
    parser.add_argument("--output", default="tunisia_hotel_prospects_1.csv", help="Output CSV filename")
    parser.add_argument("--max-results", type=int, default=30, help="Maximum candidates to audit")
    parser.add_argument("--max-searches", type=int, default=80, help="Maximum Brave Search API calls")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between audit requests in seconds")
    parser.add_argument("--search-delay", type=float, default=0.3, help="Delay between search calls in seconds")
    args = parser.parse_args()

    if args.input:
        hotels = load_input(args.input)
        print(f"Loaded {len(hotels)} websites from {args.input}")
    else:
        api_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
        if not api_key:
            print(
                "Set BRAVE_SEARCH_API_KEY or provide --input hotels.csv.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        print("Discovering candidate hotel websites across Tunisia...")
        hotels = discover(
            api_key,
            max(1, args.max_results),
            max_searches=max(1, args.max_searches),
            delay=max(0, args.search_delay),
        )
        print(f"Selected {len(hotels)} candidate websites for auditing.")

    deduped = {}
    for hotel in hotels:
        url = normalize_url(hotel.get("website") or hotel.get("url") or "")
        host = domain_key(url)
        if url and host and host not in deduped:
            hotel["website"] = url
            deduped[host] = hotel

    results = []
    for index, hotel in enumerate(deduped.values(), 1):
        print(f"[{index}/{len(deduped)}] Auditing {hotel.get('hotel_name') or hotel.get('discovery_title') or hotel.get('website')} ...")
        try:
            results.append(audit(hotel, delay=min(max(args.delay, 0), 3)))
        except KeyboardInterrupt:
            print("\nInterrupted; exporting completed audits...")
            break
        except Exception as exc:
            print(f"  [audit warning] {exc}", file=sys.stderr)
            results.append(base_result(
                hotel,
                hotel.get("website", ""),
                "error",
                f"Audit error: {type(exc).__name__}: {exc}",
            ))
        if args.delay > 0:
            time.sleep(args.delay)

    results.sort(
        key=lambda row: (
            int(row.get("opportunity_score") or 0),
            row.get("evidence_confidence") == "high",
        ),
        reverse=True,
    )
    export_csv(args.output, results)
    print(f"\nDone. Exported {len(results)} rows to: {os.path.abspath(args.output)}")
    print("Reminder: not_detected means no qualifying public signal was found; it does not prove absence.")


if __name__ == "__main__":
    main()
