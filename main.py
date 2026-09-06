"""
MajorMove — Production Backend
FastAPI server: auth, catalog scraping, AI degree analysis, transcript parsing, analytics.

Run locally:
    pip install -r requirements.txt
    cp .env.example .env   # fill in keys
    uvicorn main:app --reload

Deploy: Railway / Render / Fly.io. Set env vars in the dashboard.
"""

import os
import io
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current folder into the environment — required for local dev
import json
import re
import base64
import time
import hashlib
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, Union

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# PDF handling: PyMuPDF converts PDF pages to images so the AI reads the
# transcript with real layout/vision understanding, instead of pypdf's
# plain text extraction, which loses structure on multi-column academic
# transcripts (side-by-side terms, tables) and was the root cause of
# inaccurate credit/major reads on complex real transcripts.
try:
    import pymupdf
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

# Postgres (production, persists across deploys) vs SQLite (local dev only —
# its local file gets wiped every time Railway rebuilds the container, which
# is exactly what was silently erasing all saved users/roadmaps/emails
# between deploys). Railway auto-injects DATABASE_URL when a Postgres
# service is attached; its presence is what decides which backend runs.
try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
DB_PATH = os.environ.get("DB_PATH", "majormove.db")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
USE_POSTGRES = bool(DATABASE_URL) and HAS_PSYCOPG2
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(title="MajorMove API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------
# Database (Postgres in production — persists across deploys; SQLite as a
# local-dev fallback when DATABASE_URL isn't set, so `uvicorn main:app` still
# works on a laptop with zero extra setup)
# ----------------------------------------------------------------------------
class _PGCursor:
    """Wraps a psycopg2 cursor so calling code can use it exactly like a
    sqlite3 cursor: conn.execute(sql, params).fetchone()/.fetchall(), plus
    row["column"] dict-style access (via RealDictCursor) and .lastrowid
    for INSERTs that need the new row's id back — none of which psycopg2
    provides natively, but sqlite3 does, and the rest of this file was
    written against sqlite3's interface."""
    def __init__(self, raw_cursor):
        self._cur = raw_cursor
        self.lastrowid = None

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()


class _PGConn:
    def __init__(self, raw_conn):
        self._conn = raw_conn

    def execute(self, sql, params=()):
        pg_sql = sql.replace("?", "%s")
        is_insert = pg_sql.strip().upper().startswith("INSERT")
        has_returning = "RETURNING" in pg_sql.upper()
        # INSERT OR REPLACE (SQLite) has no Postgres equivalent syntax —
        # every call site using it in this file targets catalog_cache,
        # whose primary key is cache_key, so translate it to a real
        # Postgres upsert rather than just swapping placeholders.
        if pg_sql.strip().upper().startswith("INSERT OR REPLACE INTO CATALOG_CACHE"):
            pg_sql = pg_sql.replace("INSERT OR REPLACE INTO catalog_cache", "INSERT INTO catalog_cache")
            pg_sql += (" ON CONFLICT (cache_key) DO UPDATE SET "
                       "school=EXCLUDED.school, major=EXCLUDED.major, content=EXCLUDED.content, "
                       "source_url=EXCLUDED.source_url, verified=EXCLUDED.verified, scraped_at=EXCLUDED.scraped_at")
        elif is_insert and not has_returning and "INSERT INTO users " in pg_sql:
            # The only INSERT in this file whose caller reads .lastrowid
            # right after — give Postgres a way to hand that id back too.
            pg_sql += " RETURNING id"
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(pg_sql, params)
        wrapped = _PGCursor(cur)
        if "RETURNING id" in pg_sql:
            row = cur.fetchone()
            wrapped.lastrowid = row["id"] if row else None
        return wrapped

    def executescript(self, sql):
        cur = self._conn.cursor()
        cur.execute(sql)  # Postgres accepts a multi-statement DDL string directly
        cur.close()

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()


@contextmanager
def db():
    if USE_POSTGRES:
        raw = psycopg2.connect(DATABASE_URL)
        conn = _PGConn(raw)
    else:
        raw = sqlite3.connect(DB_PATH)
        raw.row_factory = sqlite3.Row
        conn = raw
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as conn:
        if USE_POSTGRES:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                school TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS roadmaps (
                id SERIAL PRIMARY KEY,
                user_id INTEGER,
                email TEXT,
                school TEXT, year TEXT, major TEXT,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS catalog_cache (
                cache_key TEXT PRIMARY KEY,
                school TEXT, major TEXT,
                content TEXT NOT NULL,
                source_url TEXT,
                verified INTEGER DEFAULT 0,
                scraped_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id SERIAL PRIMARY KEY,
                anon_id TEXT,
                user_id INTEGER,
                name TEXT NOT NULL,
                props TEXT,
                school TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS outcomes (
                id SERIAL PRIMARY KEY,
                roadmap_id INTEGER,
                email TEXT,
                self_reported_outcome TEXT NOT NULL,
                new_major TEXT,
                notes TEXT,
                reported_at TEXT NOT NULL,
                FOREIGN KEY(roadmap_id) REFERENCES roadmaps(id)
            );
            """)
        else:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                school TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS roadmaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                email TEXT,
                school TEXT, year TEXT, major TEXT,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            -- Verified catalog cache: scrape-on-demand, then reuse
            CREATE TABLE IF NOT EXISTS catalog_cache (
                cache_key TEXT PRIMARY KEY,      -- school|major (lowercased)
                school TEXT, major TEXT,
                content TEXT NOT NULL,
                source_url TEXT,
                verified INTEGER DEFAULT 0,      -- 1 = human/UNL-verified, 0 = scraped
                scraped_at TEXT NOT NULL
            );
            -- Analytics events
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                anon_id TEXT,
                user_id INTEGER,
                name TEXT NOT NULL,
                props TEXT,
                school TEXT,
                created_at TEXT NOT NULL
            );
            -- Long-term outcome tracking: did a student who got an analysis
            -- actually switch majors? Self-reported, since there's no real
            -- integration with a university's official student records —
            -- this is the honest, buildable version of that question, and
            -- the exact dataset that raises MajorMove's value to almost
            -- every realistic acquirer, not just one.
            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roadmap_id INTEGER,
                email TEXT,
                self_reported_outcome TEXT NOT NULL,  -- "stayed" or "switched"
                new_major TEXT,                        -- if switched, to what
                notes TEXT,
                reported_at TEXT NOT NULL,
                FOREIGN KEY(roadmap_id) REFERENCES roadmaps(id)
            );
            """)


init_db()

# ----------------------------------------------------------------------------
# Auth helpers
# ----------------------------------------------------------------------------
def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()


def new_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = datetime.utcnow()
    with db() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?,?,?,?)",
            (token, user_id, now.isoformat(), (now + timedelta(days=30)).isoformat()),
        )
    return token


def current_user(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """Optional auth — returns user dict or None. Endpoints decide if required."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    with db() as conn:
        row = conn.execute(
            "SELECT s.user_id, s.expires_at, u.email, u.school "
            "FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token = ?",
            (token,),
        ).fetchone()
    if not row:
        return None
    if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
        return None
    return {"id": row["user_id"], "email": row["email"], "school": row["school"]}

# ----------------------------------------------------------------------------
# UNL deep catalog (seed of verified data — expand via scraper / manual curation)
# In production this table is filled by a scheduled scraper over catalog.unl.edu
# ----------------------------------------------------------------------------
UNL_SEED = {
    "finance": {
        "source_url": "https://catalog.unl.edu/undergraduate/business/finance/",
        "content": "UNL Finance (College of Business). Core: ECON 211 & 212, ACCT 201 & 202, "
                   "FINA 361 (Finance), FINA 362, FINA 366, MNGT 360. Math: MATH 104 or 106. "
                   "Advising: College of Business Undergraduate Advising, HLH. "
                   "Career: Business Career Center.",
    },
    "computer science": {
        "source_url": "https://catalog.unl.edu/undergraduate/engineering/computer-science/",
        "content": "UNL Computer Science (College of Engineering / Raikes School option). "
                   "Core: CSCE 155A/155E, CSCE 156, CSCE 230, CSCE 235, CSCE 310, CSCE 322, CSCE 361. "
                   "Math: MATH 106, 107, 208. Raikes School: RAIK 183H/184H sequence. "
                   "Advising: SOFT/CSE advising, Avery Hall.",
    },
    "psychology": {
        "source_url": "https://catalog.unl.edu/undergraduate/arts-sciences/psychology/",
        "content": "UNL Psychology (College of Arts & Sciences). Core: PSYC 181, PSYC 288 (stats), "
                   "PSYC 350, plus breadth across developmental/cognitive/social. "
                   "Advising: CAS Advising Center, 107 Oldfather.",
    },
}


def seed_unl():
    now = datetime.utcnow().isoformat()
    with db() as conn:
        for major, data in UNL_SEED.items():
            key = f"university of nebraska-lincoln|{major}"
            exists = conn.execute("SELECT 1 FROM catalog_cache WHERE cache_key=?", (key,)).fetchone()
            if not exists:
                conn.execute(
                    "INSERT INTO catalog_cache (cache_key, school, major, content, source_url, verified, scraped_at) "
                    "VALUES (?,?,?,?,?,1,?)",
                    (key, "University of Nebraska-Lincoln", major, data["content"], data["source_url"], now),
                )


seed_unl()

# ----------------------------------------------------------------------------
# JSON repair — the AI occasionally uses a literal " for emphasis inside a
# text field despite explicit instructions not to (e.g. explore this
# "seriously" before committing), which breaks JSON parsing. A legitimate
# JSON quote always has at least one structural neighbor (: , [ { on one
# side, or , } ] : on the other, ignoring whitespace). A quote with plain
# text on BOTH sides is never legitimate JSON — it's a stray quote used
# inside a sentence, and gets converted to a single quote instead.
# ----------------------------------------------------------------------------
_STRUCTURAL_BEFORE = (":", ",", "[", "{")
_STRUCTURAL_AFTER = (",", "}", "]", ":")


def repair_stray_quotes(text: str) -> str:
    out = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch != '"':
            out.append(ch)
            continue
        prev_non_ws = next((out[j] for j in range(len(out) - 1, -1, -1) if not out[j].isspace()), None)
        k = i + 1
        while k < n and text[k].isspace():
            k += 1
        next_non_ws = text[k] if k < n else None
        legit_before = prev_non_ws is None or prev_non_ws in _STRUCTURAL_BEFORE
        legit_after = next_non_ws is None or next_non_ws in _STRUCTURAL_AFTER
        out.append('"' if (legit_before or legit_after) else "'")
    return "".join(out)


def repair_missing_commas(text: str) -> str:
    """A `}` immediately followed by a `{` (ignoring whitespace) is never
    valid JSON on its own — it always needs a comma between two sibling
    objects in an array. The AI occasionally drops this comma. Inserting
    one is always safe: it either fixes a real omission or, if the objects
    weren't meant to be siblings, fails parsing the same way it already
    would have without the insertion."""
    return re.sub(r"\}(\s*)\{", r"},\1{", text)


# ----------------------------------------------------------------------------
# Catalog scraping — scrape-on-demand with cache
# ----------------------------------------------------------------------------
def is_unl(school: str) -> bool:
    return any(t in school.lower() for t in ("nebraska", "unl", "husker"))


async def fetch_catalog(school: str, major: str) -> dict:
    """Return {content, source_url, verified}. Cache first, then scrape, then empty."""
    key = f"{school.strip().lower()}|{major.strip().lower()}"
    with db() as conn:
        row = conn.execute("SELECT * FROM catalog_cache WHERE cache_key=?", (key,)).fetchone()
        if row:
            return {"content": row["content"], "source_url": row["source_url"], "verified": bool(row["verified"])}

    # Not cached — scrape via Serper search → fetch top result
    content, source_url = "", ""
    if SERPER_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=12) as client:
                sr = await client.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                    json={"q": f"{school} {major} major requirements course catalog", "num": 3},
                )
                organic = sr.json().get("organic", [])
                for result in organic:
                    url = result.get("link", "")
                    if not url:
                        continue
                    try:
                        page = await client.get(url, timeout=12, follow_redirects=True)
                        soup = BeautifulSoup(page.content, "html.parser")
                        for tag in soup(["nav", "footer", "header", "script", "style"]):
                            tag.decompose()
                        text = soup.get_text(separator=" ", strip=True)
                        if len(text) > 400:
                            content = text[:4000]
                            source_url = url
                            break
                    except Exception:
                        continue
        except Exception:
            pass

    if content:
        with db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO catalog_cache (cache_key, school, major, content, source_url, verified, scraped_at) "
                "VALUES (?,?,?,?,?,0,?)",
                (key, school, major, content, source_url, datetime.utcnow().isoformat()),
            )
    return {"content": content, "source_url": source_url, "verified": False}

# ----------------------------------------------------------------------------
# Transcript parsing
# ----------------------------------------------------------------------------
def pdf_to_images(data: bytes, max_pages: int = 4) -> list[dict]:
    """Convert PDF pages to base64 PNG images so the AI reads the transcript
    visually (correct column/table layout) instead of via flattened text
    extraction, which scrambles multi-column academic transcripts."""
    if not HAS_PYMUPDF:
        return []
    images = []
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
        for page in doc[:max_pages]:
            pix = page.get_pixmap(dpi=150)
            img_b64 = base64.b64encode(pix.tobytes("png")).decode()
            images.append({"media_type": "image/png", "data": img_b64})
        doc.close()
    except Exception:
        return []
    return images


def extract_pdf_text(data: bytes) -> str:
    """Fallback text extraction — only used if PyMuPDF is unavailable.
    Known to scramble multi-column transcript layouts; image conversion
    above is the primary, more accurate path."""
    if not HAS_PYPDF:
        return ""
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)[:8000]
    except Exception:
        return ""

# ----------------------------------------------------------------------------
# AI analysis generation — MajorMove dashboard schema
# ----------------------------------------------------------------------------
ANALYSIS_SCHEMA = """Respond ONLY with valid JSON, no markdown:
{
  "current": {
    "major": "their current major",
    "status_line": "One warm sentence assessing where they stand right now",
    "credits_completed": "estimate from transcript, e.g. '75 of ~120'",
    "on_track_note": "one honest sentence on whether their current path fits THEM, given stated interests/values"
  },
  "paths": [
    {
      "major": "Major name — include current major as path 0 (is_current true) so they can compare against staying",
      "is_current": true or false,
      "success_likelihood": 82,
      "likelihood_reason": "one short sentence grounding the percent — fit to interests/values, market outlook, workload realism",
      "credits_transfer": "how many completed credits carry over, from transcript — 'All of them, you're already here' for current major",
      "additional_credits_needed": 0,
      "extra_time": "0 semesters / 1 semester / 2 semesters — with a short reason",
      "honest_take": "one honest, specific sentence about real outcomes",
      "reasoning_points": [
        "specific, checkable reason — e.g. exact completed courses that count toward this major",
        "specific, checkable reason — e.g. their strongest grades/subjects and how they connect",
        "specific, checkable reason — e.g. exact credits or courses still needed",
        "specific, checkable reason — e.g. how their stated interests connect to this major's careers"
      ],
      "fit_scores": {
        "income": 75,
        "balance": 60,
        "creative_freedom": 40
      },
      "careers": [{"title":"...","salary":"$X-$Y"},{"title":"...","salary":"$X-$Y"},{"title":"...","salary":"$X-$Y"}],
      "first_course": "specific course code + name to take first at their school",
      "financial_note": "scholarship/aid impact given their financial situation",
      "why_fit": "one sentence on why this fits (or doesn't) THIS student specifically"
    }
  ],
  "retention_nudge": "one concrete, encouraging next step + specific office/advisor to visit — written to keep this student engaged and enrolled",
  "closing": "short warm sign-off"
}
Include the current major as path 0 (is_current: true) plus exactly 3 alternative paths (is_current: false).
Success likelihood should vary realistically (not all 80+). Be honest about salaries with real market data.
For fit_scores (0-100 each, per path): "income" = how well this path's realistic earning potential
matches a high-income priority; "balance" = how well the typical workload/hours in this field support
work-life balance; "creative_freedom" = how much genuine creative or independent-thinking latitude the
work involves day to day. These should vary meaningfully across paths and be honest, not padded — a
path can honestly score low on one dimension while being strong on the others.

For additional_credits_needed: your best real estimate of how many NEW credits (beyond what already
transfers) this path requires, based on the transcript and catalog data — 0 for the current major.
This number is used for real downstream cost math, so make it as accurate as you can, not a round guess.

For reasoning_points: this is what makes a student actually trust the recommendation instead of treating
it as a black box. Each point must reference something SPECIFIC and CHECKABLE — an actual completed
course by name/code, an actual grade pattern, an actual credit count, or an actual stated interest —
never a vague generic statement like "this seems like a good fit." A student reading these should be
able to verify each one against their own transcript.

ABSOLUTE RULE — this breaks the entire response if violated, follow it with zero exceptions:
The double-quote character (") may ONLY appear as JSON structure (wrapping keys and string values).
It must NEVER appear inside the text of any string value, for ANY reason — not for emphasis, not to
quote a phrase, not for scare quotes, not for anything.
WRONG (breaks parsing): "honest_take": "This is a "great fit" if you like data."
WRONG (breaks parsing): "why_fit": "Explore this "seriously" before committing."
RIGHT: "honest_take": "This is a great fit if you like data."
RIGHT: "why_fit": "Seriously consider exploring this before committing."
Simply do not use quotation marks of any kind inside your sentences. Rephrase instead of quoting."""


async def call_ai_with_retry(content, validate_fn, max_tokens: int = 16000) -> dict:
    """Shared, battle-tested AI-calling logic: retries once on any failure,
    repairs common malformed-JSON patterns (stray quotes, missing commas),
    and validates the parsed shape before accepting it — not just that it's
    valid JSON, but that it's the COMPLETE shape expected. `validate_fn`
    takes the parsed dict and returns (is_valid: bool, reason: str) so each
    caller can define its own "did I actually get everything I asked for"
    check without duplicating this retry machinery.
    """
    last_error = None
    last_debug = None
    for attempt_num in range(2):  # try once, then retry once more on any failure
        try:
            # 180s — real multi-page transcript images plus a large token
            # budget (which can include heavy internal reasoning) genuinely
            # need more room than a short timeout; a short timeout was
            # cutting off real requests mid-flight, surfacing as a raw
            # connection failure ("Load failed") on the client instead of
            # a clean error.
            async with httpx.AsyncClient(timeout=180) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": ANTHROPIC_API_KEY,
                             "anthropic-version": "2023-06-01",
                             "content-type": "application/json"},
                    json={"model": ANTHROPIC_MODEL, "max_tokens": max_tokens,
                          "messages": [{"role": "user", "content": content}]},
                )
        except httpx.TimeoutException:
            last_error = f"Request to AI timed out after 180s"
            continue
        except httpx.HTTPError as e:
            last_error = f"HTTP error contacting AI: {e}"
            continue
        data = resp.json()
        if "content" not in data:
            last_error = f"AI error: {data.get('error', {}).get('message', 'unknown')}"
            continue
        text = "".join(b["text"] for b in data["content"] if b.get("type") == "text")
        if not text.strip():
            last_error = f"AI returned no text content. stop_reason: {data.get('stop_reason')}"
            continue
        clean = text.replace("```json", "").replace("```", "").strip()

        # Isolate the outermost {...} block (handles any stray prose the
        # model adds before/after despite instructions not to).
        start = clean.find("{")
        end = clean.rfind("}")
        candidate = clean[start:end + 1] if (start != -1 and end != -1 and end > start) else clean

        parsed = None
        for repaired in (candidate, repair_stray_quotes(candidate), repair_missing_commas(candidate),
                         repair_missing_commas(repair_stray_quotes(candidate))):
            try:
                parsed = json.loads(repaired)
                break
            except json.JSONDecodeError as e:
                last_error = str(e)

        if parsed is None:
            last_debug = (f"stop_reason: {data.get('stop_reason')}, usage: {data.get('usage')}, "
                           f"response ({len(candidate)} chars): {candidate}")
            continue  # retry — every repair failed to even parse

        # Parsed successfully, but only valid if it's actually the COMPLETE
        # shape expected — a truncated/early-stopped response can sometimes
        # still parse as valid JSON if cut off at a spot that closes cleanly,
        # while still being genuinely incomplete.
        is_valid, reason = validate_fn(parsed)
        if not is_valid:
            last_error = reason
            last_debug = f"stop_reason: {data.get('stop_reason')}, usage: {data.get('usage')}"
            continue  # retry — parsed fine but incomplete/wrong shape

        return parsed  # success

    raise HTTPException(502, f"Could not get a complete response after 2 attempts. "
                              f"Last error: {last_error}. Debug: {last_debug}")


def _validate_analysis(parsed: dict) -> tuple[bool, str]:
    n = len(parsed.get("paths", []))
    if n != 4:
        return False, f"Got {n} paths, expected 4 (incomplete response)"
    return True, ""


async def generate_analysis(answers: dict, catalog: dict, transcript_text: str,
                             transcript_images: list[dict]) -> dict:
    unl = is_unl(answers.get("school", ""))
    unl_block = (
        "This is a University of Nebraska-Lincoln student. Use SPECIFIC real UNL course codes "
        "(ECON 211, FINA 361, CSCE 156, PSYC 181, RAIK 184H), UNL colleges, and UNL resources "
        "(Explore Center, Business Career Center, CAS Advising). Reference Husker culture warmly."
        if unl else
        "Use the most accurate course/program info available. If unsure of exact codes, give "
        "realistic ones and note they should verify in their catalog."
    )
    catalog_block = (
        f"\nVERIFIED CATALOG DATA{' (UNL, verified)' if catalog.get('verified') else ' (scraped)'} "
        f"— ground course codes and requirements in this:\n{catalog['content']}\n"
        if catalog.get("content") else
        "\n(No catalog data retrieved — use best known info and flag for verification.)\n"
    )
    transcript_block = (
        f"\nTRANSCRIPT (pasted) — base credit-transfer and graduation timing on this:\n{transcript_text}\n"
        if transcript_text else ""
    )

    transcript_accuracy_rules = """
If a transcript (image or text) is provided, read it with extreme care — this is a real
academic record, not a summary, and mistakes here undermine the whole analysis:
- Scan EVERY page and EVERY term, not just the most recent one.
- Find the FINAL/most recent "Program:", "Major:", "Minor:", "Option:" declarations —
  these change over time as a student changes majors, so use only the LATEST set, not
  an earlier term's declarations.
- The student may have MULTIPLE currently declared majors and/or minors simultaneously
  (e.g. a double major, or a major plus one or more minors). List ALL of them in your
  understanding of "current major" — do not silently pick just one if several are declared
  in the latest term.
- For "credits completed", use the CUMULATIVE EARNED HOURS (often labeled EHRS or
  "Earned Hours") from the LAST/most recent term summary — not attempted hours (AHRS),
  not an early term's total, and not a rough guess. Read the actual cumulative row.
- When evaluating an alternative major, check the ENTIRE course history for classes that
  would already count toward it (e.g. if evaluating Computer Science, check for any CS
  courses already completed in ANY term) — do not assume "starting from scratch" without
  checking.
"""

    prompt = f"""You are MajorMove, an AI academic advisor whose purpose is to help a college student
make a clearer, better decision about their major — and, in aggregate, to help their university
retain and graduate more students. Be warm, honest, specific, never generic.

Student:
- School: {answers.get('school')}
- Year: {answers.get('year')}
- Current major (as self-reported in the form — verify/expand using the transcript if provided): {answers.get('major')}
- Interests: {', '.join(answers.get('interests', []))}
- Career values: {', '.join(answers.get('values', []))}
- Financial: {answers.get('financial')}
{catalog_block}{transcript_block}
{transcript_accuracy_rules}
{unl_block}

{ANALYSIS_SCHEMA}"""

    if transcript_images:
        content = [
            {"type": "image", "source": {"type": "base64",
             "media_type": img["media_type"], "data": img["data"]}}
            for img in transcript_images
        ] + [{"type": "text", "text": prompt +
              f"\n\n{len(transcript_images)} transcript page image(s) are attached above — "
              f"read every page carefully per the accuracy rules."}]
    else:
        content = prompt

    return await call_ai_with_retry(content, _validate_analysis, max_tokens=16000)


# ----------------------------------------------------------------------------
# Career exploration — a SEPARATE, lightweight, text-only call. Deliberately
# does NOT re-read the transcript images: career fit comes from interests,
# values, and declared major (all already-known form inputs), not from
# precise credit-by-credit transcript accuracy. Keeping this call text-only
# and each card lean is what keeps ~18-20 careers fast and cheap instead of
# repeating the token-budget/timeout problems a giant single call caused.
# ----------------------------------------------------------------------------
CAREERS_SCHEMA = """Respond ONLY with valid JSON, no markdown:
{
  "careers": [
    {
      "title": "Job title",
      "why_it_fits": "One honest sentence connecting this to their specific interests/values/major",
      "salary_range": "$X-$Y, realistic entry-level to a few years in",
      "day_in_the_life": "One or two sentences, concrete and specific, not generic",
      "how_to_get_there": "One sentence — from their current position, what's the realistic first step",
      "growth_outlook": "One short phrase on demand/growth for this role"
    }
  ]
}
Generate exactly 18 careers. They should span a real range — some closely tied to their
current/likely major, some more exploratory based on interests they mentioned, some that
connect two interests together in a way they may not have considered. Vary salary ranges
honestly — not everything should be high-paying. No duplicates. No generic filler titles.
CRITICAL: never use a double-quote character (") inside any string value — it breaks JSON
parsing. Use single quotes ('like this') if you need to quote a phrase, or better, just
rephrase to avoid quoting at all."""


def _validate_careers(parsed: dict) -> tuple[bool, str]:
    n = len(parsed.get("careers", []))
    if n < 12:  # allow some slack below the requested 18, but not a near-empty response
        return False, f"Got only {n} careers, expected around 18 (incomplete response)"
    return True, ""


async def generate_careers(answers: dict, unl: bool) -> dict:
    unl_block = (
        "This is a University of Nebraska-Lincoln student — where relevant, mention real "
        "UNL resources like the Business Career Center or Explore Center as a next step."
        if unl else ""
    )
    prompt = f"""You are MajorMove's career exploration feature. Based on this student's
profile, generate a broad, honest, specific set of career options they may not have
fully considered — this is meant to be explored and scrolled through, not just their
one "correct" answer.

Student:
- School: {answers.get('school')}
- Year: {answers.get('year')}
- Current/declared major: {answers.get('major')}
- Interests: {', '.join(answers.get('interests', []))}
- Career values: {', '.join(answers.get('values', []))}
{unl_block}

{CAREERS_SCHEMA}"""

    return await call_ai_with_retry(prompt, _validate_careers, max_tokens=6000)


# ----------------------------------------------------------------------------
# Major browsing + search — a broad, browsable list of compatible majors,
# separate from the deep 4-path analysis. Same lightweight, text-only, no
# transcript-images design as careers, for the same reason: this doesn't
# need transcript-level precision, just interests/values/major context.
# ----------------------------------------------------------------------------
MAJORS_LIST_SCHEMA = """Respond ONLY with valid JSON, no markdown:
{
  "majors": [
    {
      "name": "Major name",
      "fit_percentage": 82,
      "one_liner": "One honest sentence on why this fits their specific interests/values",
      "salary_range": "$X-$Y, realistic entry-level"
    }
  ]
}
Generate exactly 10 majors, ranked by fit_percentage descending. Include their current major
if it genuinely belongs in a top-10 fit list — don't force it in artificially if it doesn't.
Vary fit_percentage honestly (not all 80+). No duplicates. No generic filler names.
CRITICAL: never use a double-quote character (") inside any string value — use single quotes
('like this') instead, or rephrase to avoid quoting entirely."""

MAJOR_SEARCH_SCHEMA = """Respond ONLY with valid JSON, no markdown:
{
  "name": "the exact major name searched for",
  "fit_percentage": 68,
  "one_liner": "One honest sentence on why this does or doesn't fit their specific interests/values",
  "salary_range": "$X-$Y, realistic entry-level"
}
Be honest — if this major is a poor fit for their stated interests/values, say so plainly and
give an honest, lower fit_percentage rather than padding it. CRITICAL: never use a double-quote
character (") inside any string value."""


def _validate_majors_list(parsed: dict) -> tuple[bool, str]:
    n = len(parsed.get("majors", []))
    if n < 7:  # allow some slack below the requested 10, but not a near-empty response
        return False, f"Got only {n} majors, expected around 10 (incomplete response)"
    return True, ""


def _validate_major_search(parsed: dict) -> tuple[bool, str]:
    if not parsed.get("name") or "fit_percentage" not in parsed:
        return False, "Missing required fields in single-major search result"
    return True, ""


async def generate_majors_list(answers: dict, unl: bool) -> dict:
    unl_block = (
        "This is a University of Nebraska-Lincoln student — favor real UNL majors where possible."
        if unl else ""
    )
    prompt = f"""You are MajorMove's major exploration feature. Based on this student's profile,
generate a broad, honest, ranked list of majors that could fit them — meant to be browsed,
not just their one deep-dive comparison.

Student:
- School: {answers.get('school')}
- Year: {answers.get('year')}
- Current/declared major: {answers.get('major')}
- Interests: {', '.join(answers.get('interests', []))}
- Career values: {', '.join(answers.get('values', []))}
{unl_block}

{MAJORS_LIST_SCHEMA}"""

    return await call_ai_with_retry(prompt, _validate_majors_list, max_tokens=3000)


async def generate_major_search(answers: dict, search_major: str, unl: bool) -> dict:
    unl_block = (
        f"This is a University of Nebraska-Lincoln student — if {search_major} is offered at "
        f"UNL, reference it specifically; if you're not certain it's offered there, say so."
        if unl else ""
    )
    prompt = f"""You are MajorMove's major search feature. A student has a SPECIFIC major in
mind — evaluate honestly whether it fits them, don't just confirm whatever they typed.

Student:
- School: {answers.get('school')}
- Year: {answers.get('year')}
- Current/declared major: {answers.get('major')}
- Interests: {', '.join(answers.get('interests', []))}
- Career values: {', '.join(answers.get('values', []))}
- Major they're asking about: {search_major}
{unl_block}

{MAJOR_SEARCH_SCHEMA}"""

    return await call_ai_with_retry(prompt, _validate_major_search, max_tokens=800)

# ----------------------------------------------------------------------------
# Analytics
# ----------------------------------------------------------------------------
def log_event(name: str, anon_id: str = None, user_id: int = None, props: dict = None, school: str = None):
    with db() as conn:
        conn.execute(
            "INSERT INTO events (anon_id, user_id, name, props, school, created_at) VALUES (?,?,?,?,?,?)",
            (anon_id, user_id, name, json.dumps(props or {}), school, datetime.utcnow().isoformat()),
        )

# ----------------------------------------------------------------------------
# Request models
# ----------------------------------------------------------------------------
class SignupReq(BaseModel):
    email: str
    password: str
    school: Optional[str] = None

class LoginReq(BaseModel):
    email: str
    password: str

class EventReq(BaseModel):
    name: str
    anon_id: Optional[str] = None
    props: Optional[dict] = None
    school: Optional[str] = None

class OutcomeReq(BaseModel):
    email: str
    self_reported_outcome: str  # "stayed" or "switched"
    new_major: Optional[str] = None
    notes: Optional[str] = None

# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "pypdf": HAS_PYPDF, "pymupdf": HAS_PYMUPDF,
            "serper": bool(SERPER_API_KEY), "ai": bool(ANTHROPIC_API_KEY)}


@app.post("/auth/signup")
async def signup(req: SignupReq):
    salt = secrets.token_hex(16)
    pw = hash_password(req.password, salt)
    duplicate_email_errors = (sqlite3.IntegrityError,)
    if USE_POSTGRES:
        duplicate_email_errors = (sqlite3.IntegrityError, psycopg2.IntegrityError)
    try:
        with db() as conn:
            cur = conn.execute(
                "INSERT INTO users (email, password_hash, salt, school, created_at) VALUES (?,?,?,?,?)",
                (req.email.lower(), pw, salt, req.school, datetime.utcnow().isoformat()),
            )
            uid = cur.lastrowid
    except duplicate_email_errors:
        raise HTTPException(400, "Email already registered")
    token = new_session(uid)
    log_event("signup", user_id=uid, school=req.school)
    return {"token": token, "user": {"id": uid, "email": req.email.lower(), "school": req.school}}


@app.post("/auth/login")
async def login(req: LoginReq):
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email=?", (req.email.lower(),)).fetchone()
    if not row or hash_password(req.password, row["salt"]) != row["password_hash"]:
        raise HTTPException(401, "Invalid credentials")
    token = new_session(row["id"])
    return {"token": token, "user": {"id": row["id"], "email": row["email"], "school": row["school"]}}


@app.post("/analyze")
async def analyze(
    school: str = Form(...),
    year: str = Form(...),
    major: str = Form(...),
    interests: str = Form(""),       # comma-separated
    values: str = Form(""),
    financial: str = Form(""),
    transcript_text: str = Form(""),
    anon_id: str = Form(None),
    email: str = Form(None),
    cost_per_credit: str = Form(""),  # optional, student-provided, e.g. "450" — used for
                                       # real deterministic cost math, never AI-guessed
    transcript_file: Union[UploadFile, str, None] = File(None),
    user: Optional[dict] = Depends(current_user),
):
    answers = {
        "school": school, "year": year, "major": major,
        "interests": [i for i in interests.split(",") if i],
        "values": [v for v in values.split(",") if v],
        "financial": financial,
    }

    # Transcript: PDF → converted to page images (accurate layout reading);
    # image upload → passed through directly; both go to the model as vision
    # input, since that reads real academic transcripts far more accurately
    # than flattened text extraction (which scrambles multi-column layouts).
    # (Swagger's "Try it out" UI sometimes sends an empty string instead of
    # omitting the file entirely — treat anything that isn't a real UploadFile as "no file")
    t_text = transcript_text or ""
    t_images: list[dict] = []
    # Detect a real uploaded file by attributes, not by isinstance(UploadFile) —
    # Starlette/FastAPI can hand back starlette.datastructures.UploadFile vs
    # fastapi.datastructures.UploadFile depending on version, and isinstance
    # against the wrong one silently (and incorrectly) treats a real upload
    # as "no file", which is what was actually happening here.
    is_real_upload = (
        transcript_file is not None
        and not isinstance(transcript_file, str)
        and hasattr(transcript_file, "filename")
        and hasattr(transcript_file, "read")
    )
    if is_real_upload:
        raw = await transcript_file.read()
        ctype = transcript_file.content_type or ""
        if ctype == "application/pdf" or transcript_file.filename.lower().endswith(".pdf"):
            t_images = pdf_to_images(raw)
            if not t_images:  # PyMuPDF unavailable or conversion failed — fall back to text
                extracted = extract_pdf_text(raw)
                t_text = (t_text + "\n" + extracted).strip()
        elif ctype.startswith("image/"):
            t_images = [{"media_type": ctype, "data": base64.b64encode(raw).decode()}]

    catalog = await fetch_catalog(school, major)
    resolved_email = email or (user["email"] if user else None)
    log_event("analysis_started", anon_id=anon_id,
              user_id=user["id"] if user else None, school=school,
              props={"major": major, "has_transcript": bool(t_text or t_images),
                     "transcript_kind": "text" if t_text else ("image" if t_images else "none"),
                     "transcript_page_count": len(t_images),
                     "email_provided": bool(resolved_email)})

    try:
        result = await generate_analysis(answers, catalog, t_text, t_images)
    except json.JSONDecodeError:
        raise HTTPException(502, "Could not parse AI response")

    result["_catalog_source"] = catalog.get("source_url")
    result["_catalog_verified"] = catalog.get("verified", False)
    result["_transcript_pages_received"] = len(t_images)
    result["_transcript_text_received"] = bool(t_text)
    result["_transcript_file_uploaded"] = is_real_upload

    # Real cost math, computed here in Python — never left to the AI to guess.
    # Tuition varies wildly by school (in-state/out-of-state, public/private,
    # per-credit rates), and a confidently-stated wrong dollar figure is worse
    # than showing none at all. Only compute this if the student gave us their
    # own real per-credit rate; simple multiplication on their real number.
    try:
        rate = float(cost_per_credit) if cost_per_credit.strip() else None
    except ValueError:
        rate = None
    for path in result.get("paths", []):
        extra_credits = path.get("additional_credits_needed")
        if rate is not None and isinstance(extra_credits, (int, float)):
            path["estimated_cost_delta"] = round(extra_credits * rate, 2)
        else:
            path["estimated_cost_delta"] = None

    with db() as conn:
        conn.execute(
            "INSERT INTO roadmaps (user_id, email, school, year, major, payload, created_at) VALUES (?,?,?,?,?,?,?)",
            (user["id"] if user else None, resolved_email, school, year, major,
             json.dumps(result), datetime.utcnow().isoformat()),
        )
    log_event("analysis_completed", anon_id=anon_id,
              user_id=user["id"] if user else None, school=school)
    return result


@app.post("/careers")
async def careers(
    school: str = Form(...),
    year: str = Form(...),
    major: str = Form(...),
    interests: str = Form(""),
    values: str = Form(""),
    anon_id: str = Form(None),
    user: Optional[dict] = Depends(current_user),
):
    """Lightweight, text-only career exploration — deliberately separate
    from /analyze. No transcript, no images, no catalog scraping: career
    fit comes from interests/values/major, which are already-known form
    inputs. This keeps it fast and cheap, and avoids the exact token-budget
    and timeout failure modes that a giant single call caused before."""
    answers = {
        "school": school, "major": major,
        "interests": [i for i in interests.split(",") if i],
        "values": [v for v in values.split(",") if v],
    }
    log_event("careers_started", anon_id=anon_id,
              user_id=user["id"] if user else None, school=school,
              props={"major": major})

    result = await generate_careers(answers, is_unl(school))
    log_event("careers_completed", anon_id=anon_id,
              user_id=user["id"] if user else None, school=school,
              props={"career_count": len(result.get("careers", []))})
    return result


@app.post("/majors")
async def majors_list(
    school: str = Form(...),
    year: str = Form(...),
    major: str = Form(...),
    interests: str = Form(""),
    values: str = Form(""),
    anon_id: str = Form(None),
    user: Optional[dict] = Depends(current_user),
):
    """Top 10 compatible majors — a broad browse list, separate from the
    deep 4-path analysis. Same lightweight design as /careers."""
    answers = {
        "school": school, "major": major,
        "interests": [i for i in interests.split(",") if i],
        "values": [v for v in values.split(",") if v],
    }
    log_event("majors_list_started", anon_id=anon_id,
              user_id=user["id"] if user else None, school=school, props={"major": major})

    result = await generate_majors_list(answers, is_unl(school))
    log_event("majors_list_completed", anon_id=anon_id,
              user_id=user["id"] if user else None, school=school,
              props={"major_count": len(result.get("majors", []))})
    return result


@app.post("/majors/search")
async def majors_search(
    school: str = Form(...),
    year: str = Form(...),
    major: str = Form(...),
    interests: str = Form(""),
    values: str = Form(""),
    search_major: str = Form(...),
    anon_id: str = Form(None),
    user: Optional[dict] = Depends(current_user),
):
    """A student has a specific major in mind — evaluate it honestly,
    same lightweight pattern as the top-10 list, just for one item."""
    if not search_major.strip():
        raise HTTPException(400, "search_major is required")
    answers = {
        "school": school, "major": major,
        "interests": [i for i in interests.split(",") if i],
        "values": [v for v in values.split(",") if v],
    }
    log_event("major_search", anon_id=anon_id,
              user_id=user["id"] if user else None, school=school,
              props={"searched": search_major})

    return await generate_major_search(answers, search_major.strip(), is_unl(school))


@app.get("/me/roadmaps")
async def my_roadmaps(user: Optional[dict] = Depends(current_user)):
    if not user:
        raise HTTPException(401, "Login required")
    with db() as conn:
        rows = conn.execute(
            "SELECT id, school, major, payload, created_at FROM roadmaps WHERE user_id=? ORDER BY created_at DESC",
            (user["id"],),
        ).fetchall()
    return [{"id": r["id"], "school": r["school"], "major": r["major"],
             "created_at": r["created_at"], "payload": json.loads(r["payload"])} for r in rows]


@app.post("/event")
async def event(req: EventReq, user: Optional[dict] = Depends(current_user)):
    log_event(req.name, anon_id=req.anon_id, user_id=user["id"] if user else None,
              props=req.props, school=req.school)
    return {"ok": True}


@app.get("/admin/metrics")
async def metrics(key: str):
    """Simple founder dashboard. Protect with ADMIN_KEY env in production."""
    if key != os.environ.get("ADMIN_KEY", "changeme"):
        raise HTTPException(403, "Forbidden")
    with db() as conn:
        total_users = conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"]
        total_roadmaps = conn.execute("SELECT COUNT(*) c FROM roadmaps").fetchone()["c"]
        total_emails = conn.execute(
            "SELECT COUNT(DISTINCT email) c FROM roadmaps WHERE email IS NOT NULL"
        ).fetchone()["c"]
        schools = conn.execute(
            "SELECT school, COUNT(*) c FROM roadmaps GROUP BY school ORDER BY c DESC LIMIT 25"
        ).fetchall()
        # Returning: same email showing up on 2+ distinct days — works regardless of
        # device/browser, since email (not just user_id) is captured on every analysis.
        returning = conn.execute("""
            SELECT COUNT(*) c FROM (
                SELECT email FROM roadmaps WHERE email IS NOT NULL
                GROUP BY email HAVING COUNT(DISTINCT substr(created_at,1,10)) >= 2
            )
        """).fetchone()["c"]
    return {
        "total_users": total_users,
        "total_roadmaps": total_roadmaps,
        "total_emails_collected": total_emails,
        "returning_users": returning,
        "schools": [{"school": s["school"], "count": s["c"]} for s in schools],
    }


@app.get("/admin/emails")
async def emails(key: str):
    """Raw email + school + major list, most recent first — for outreach or CSV export."""
    if key != os.environ.get("ADMIN_KEY", "changeme"):
        raise HTTPException(403, "Forbidden")
    with db() as conn:
        rows = conn.execute(
            "SELECT email, school, major, created_at FROM roadmaps "
            "WHERE email IS NOT NULL ORDER BY created_at DESC"
        ).fetchall()
    return [{"email": r["email"], "school": r["school"], "major": r["major"],
             "created_at": r["created_at"]} for r in rows]


@app.get("/admin/last_analysis")
async def last_analysis(key: str):
    """The most recently saved analysis, including its transcript-received
    diagnostics — a quick way to check whether a specific run actually got
    a transcript, without needing browser dev tools."""
    if key != os.environ.get("ADMIN_KEY", "changeme"):
        raise HTTPException(403, "Forbidden")
    with db() as conn:
        row = conn.execute(
            "SELECT email, school, major, payload, created_at FROM roadmaps "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    if not row:
        return {"message": "No analyses saved yet"}
    payload = json.loads(row["payload"])
    return {
        "email": row["email"], "school": row["school"], "major": row["major"],
        "created_at": row["created_at"],
        "_transcript_pages_received": payload.get("_transcript_pages_received"),
        "_transcript_text_received": payload.get("_transcript_text_received"),
        "_transcript_file_uploaded": payload.get("_transcript_file_uploaded"),
        "_catalog_verified": payload.get("_catalog_verified"),
        "credits_completed_shown": payload.get("current", {}).get("credits_completed"),
    }


@app.post("/outcome")
async def report_outcome(req: OutcomeReq):
    """A student self-reports, weeks or months later, whether they actually
    switched majors after using MajorMove. Honest by design: there's no real
    integration with a university's official student records, so this is
    the buildable, truthful version — self-reported, not auto-verified.
    Linked to their most recent saved roadmap by email, if one exists.
    This is the exact dataset that makes MajorMove's real-world impact
    provable to almost any acquirer, not just one."""
    if req.self_reported_outcome not in ("stayed", "switched"):
        raise HTTPException(400, "self_reported_outcome must be 'stayed' or 'switched'")
    with db() as conn:
        roadmap = conn.execute(
            "SELECT id FROM roadmaps WHERE email=? ORDER BY created_at DESC LIMIT 1",
            (req.email.lower(),),
        ).fetchone()
        roadmap_id = roadmap["id"] if roadmap else None
        conn.execute(
            "INSERT INTO outcomes (roadmap_id, email, self_reported_outcome, new_major, notes, reported_at) "
            "VALUES (?,?,?,?,?,?)",
            (roadmap_id, req.email.lower(), req.self_reported_outcome, req.new_major, req.notes,
             datetime.utcnow().isoformat()),
        )
    log_event("outcome_reported", props={"outcome": req.self_reported_outcome})
    return {"ok": True, "message": "Thanks for letting us know — this genuinely helps."}


@app.get("/admin/outcomes")
async def admin_outcomes(key: str):
    """All self-reported outcomes so far — the real-world impact dataset."""
    if key != os.environ.get("ADMIN_KEY", "changeme"):
        raise HTTPException(403, "Forbidden")
    with db() as conn:
        rows = conn.execute(
            "SELECT email, self_reported_outcome, new_major, notes, reported_at FROM outcomes "
            "ORDER BY reported_at DESC"
        ).fetchall()
        switched = conn.execute(
            "SELECT COUNT(*) c FROM outcomes WHERE self_reported_outcome='switched'"
        ).fetchone()["c"]
        stayed = conn.execute(
            "SELECT COUNT(*) c FROM outcomes WHERE self_reported_outcome='stayed'"
        ).fetchone()["c"]
    return {
        "total_reported": switched + stayed,
        "switched": switched,
        "stayed": stayed,
        "reports": [{"email": r["email"], "outcome": r["self_reported_outcome"],
                     "new_major": r["new_major"], "notes": r["notes"],
                     "reported_at": r["reported_at"]} for r in rows],
    }
