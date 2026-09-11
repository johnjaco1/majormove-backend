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
      "financial_note": "scholarship/aid impact given their financial situation — ONE short sentence, max 18 words",
      "why_fit": "one short sentence, max 15 words, on why this fits (or doesn't) THIS student specifically"
    }
  ],
  "retention_nudge": "ONE short, specific next action — name the office/advisor and what to ask, max 20 words total, no compound run-on sentences",
  "closing": "short warm sign-off, max 12 words"
}
Include the current major as path 0 (is_current: true) plus exactly 3 alternative paths (is_current: false).
Success likelihood should vary realistically (not all 80+). Be honest about salaries with real market data.

BREVITY IS MANDATORY, not a style preference. Every sentence in every field must be short and scannable —
this is read on a phone, not a report. Never write a compound sentence joining two ideas with "though",
"but", "which", or a comma-and-clause — if you have two ideas, that's two reasoning_points, not one long
sentence. Maximum ~18 words per sentence everywhere in this response, no exceptions.

ACCURACY RULE for anything citing a specific grade or number from the transcript: only state a specific
letter grade, GPA, or credit count if you are CERTAIN of it from what's actually shown. If you're not
certain of the exact grade, describe the pattern instead (e.g. "solid performance in your econ courses")
rather than stating a specific letter grade you might get wrong — a wrong specific claim is worse than an
honest general one.

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
never a vague generic statement like "this seems like a good fit." Each point is ONE short sentence,
max 16 words. A student reading these should be able to verify each one against their own transcript.

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


EXTRACTION_SCHEMA = """Respond ONLY with valid JSON, no markdown:
{
  "declared_majors": ["exact major name(s) from the LATEST term's declaration"],
  "declared_minors": ["exact minor name(s) from the LATEST term's declaration, if any"],
  "cumulative_credits_completed": 83,
  "cumulative_credits_note": "e.g. 'of ~120 required' if visible on the transcript",
  "gpa": "cumulative GPA if visible, else null",
  "completed_courses": [
    {"code": "CSCE 155H", "grade": "A"}
  ]
}
List EVERY completed course from EVERY term, not just the most recent one — this is used
to check what already counts toward alternative majors, so completeness matters more than
brevity here. Use cumulative EARNED HOURS (EHRS), not attempted hours (AHRS), for credits.
Find the LATEST "Program:"/"Major:"/"Minor:" declarations — a student's major changes over
time, so use only the most recent set, not an earlier term's. CRITICAL: never use a
double-quote character (") inside any string value."""


def _validate_extraction(parsed: dict) -> tuple[bool, str]:
    if not isinstance(parsed.get("declared_majors"), list) or not parsed.get("declared_majors"):
        return False, "Missing or empty declared_majors in extraction"
    if "cumulative_credits_completed" not in parsed:
        return False, "Missing cumulative_credits_completed in extraction"
    return True, ""


async def extract_transcript_facts(transcript_images: list[dict]) -> dict:
    """Stage 1 of the analysis pipeline: a fast, narrow, vision-based pass that
    only pulls out raw facts (majors, minors, credits, course list) — no
    reasoning, no prose, no 4-path comparison. Deliberately lean output (a
    structured list, not paragraphs) so this call is genuinely fast, unlike
    the old single combined call that did vision AND full analysis writing
    at once. Stage 2 (generate_analysis below) then reasons over these
    extracted facts as plain text, with no images — removing the vision
    overhead from the slow, heavy-output reasoning step entirely."""
    prompt = f"""Extract structured facts from this academic transcript. Do NOT analyze,
compare majors, or write any reasoning — just extract what's actually on the page.

{EXTRACTION_SCHEMA}"""
    content = [
        {"type": "image", "source": {"type": "base64",
         "media_type": img["media_type"], "data": img["data"]}}
        for img in transcript_images
    ] + [{"type": "text", "text": prompt}]

    return await call_ai_with_retry(content, _validate_extraction, max_tokens=4000)


def format_extracted_facts(facts: dict) -> str:
    """Turn the lean extraction result into a clean, readable text block for
    the (now always text-only) main analysis prompt."""
    lines = [
        f"Declared major(s): {', '.join(facts.get('declared_majors', [])) or 'none listed'}",
    ]
    if facts.get("declared_minors"):
        lines.append(f"Declared minor(s): {', '.join(facts['declared_minors'])}")
    credits = facts.get("cumulative_credits_completed")
    note = facts.get("cumulative_credits_note", "")
    lines.append(f"Cumulative credits completed: {credits}{f' ({note})' if note else ''}")
    if facts.get("gpa"):
        lines.append(f"Cumulative GPA: {facts['gpa']}")
    courses = facts.get("completed_courses", [])
    if courses:
        lines.append(f"Completed courses ({len(courses)} total):")
        lines.extend(f"  - {c.get('code', '?')}: {c.get('grade', '?')}" for c in courses)
    return "\n".join(lines)


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

    # Stage 1: if we have transcript IMAGES (not already-pasted text), run the
    # fast extraction pass first, then reason over its plain-text output —
    # this keeps the expensive/slow vision step narrow, and makes the main
    # reasoning call below always text-only, regardless of the original
    # input type. If extraction itself fails for any reason, fall back to
    # the original (slower, but proven) approach of sending images directly
    # to the main call, rather than letting the whole request fail.
    fallback_to_images = False
    if transcript_images and not transcript_text:
        try:
            facts = await extract_transcript_facts(transcript_images)
            transcript_text = format_extracted_facts(facts)
        except HTTPException:
            fallback_to_images = True

    transcript_block = (
        f"\nTRANSCRIPT FACTS — base credit-transfer and graduation timing on this:\n{transcript_text}\n"
        if transcript_text else ""
    )

    transcript_accuracy_rules = """
If transcript facts are provided above, read them with extreme care — this is a real
academic record, not a summary, and mistakes here undermine the whole analysis:
- The student may have MULTIPLE currently declared majors and/or minors simultaneously
  (e.g. a double major, or a major plus one or more minors). List ALL of them in your
  understanding of "current major" — do not silently pick just one.
- Use the cumulative credits figure given — do not guess a different number.
- When evaluating an alternative major, check the FULL completed-courses list for classes
  that would already count toward it (e.g. if evaluating Computer Science, check for any
  CS courses already completed) — do not assume "starting from scratch" without checking.
"""

    prompt = f"""You are MajorMove, an AI academic advisor whose purpose is to help a college student
make a clearer, better decision about their major — and, in aggregate, to help their university
retain and graduate more students. Be warm, honest, specific, never generic.

Student:
- School: {answers.get('school')}
- Year: {answers.get('year')}
- Current major (as self-reported in the form — verify/expand using the transcript facts if provided): {answers.get('major')}
- Interests: {', '.join(answers.get('interests', []))}
- Career values: {', '.join(answers.get('values', []))}
- Financial: {answers.get('financial')}
{catalog_block}{transcript_block}
{transcript_accuracy_rules}
{unl_block}

{ANALYSIS_SCHEMA}"""

    if fallback_to_images:
        content = [
            {"type": "image", "source": {"type": "base64",
             "media_type": img["media_type"], "data": img["data"]}}
            for img in transcript_images
        ] + [{"type": "text", "text": prompt +
              f"\n\n{len(transcript_images)} transcript page image(s) are attached above — "
              f"read every page carefully."}]
    else:
        content = prompt  # always text-only now in the normal (non-fallback) case

    return await call_ai_with_retry(content, _validate_analysis, max_tokens=16000)


# ----------------------------------------------------------------------------
# Career exploration — a SEPARATE, lightweight, text-only call. Deliberately
# does NOT re-read the transcript images: career fit comes from interests,
# values, and declared major (all already-known form inputs), not from
# precise credit-by-credit transcript accuracy. Keeping this call text-only
# and each card lean is what keeps ~18-20 careers fast and cheap instead of
# repeating the token-budget/timeout problems a giant single call caused.
# ----------------------------------------------------------------------------
# Career databank — a static, curated list of real careers for common majors.
# For any major in this databank, careers return INSTANTLY with zero AI call
# at all — a genuine, meaningful speed win. Tradeoff, worth being explicit
# about: this loses fine-grained personalization to the student's specific
# interests/values (every student in the same major sees the same list),
# unlike the AI-generated version. Majors NOT in this databank still fall
# back to the full personalized AI generation below.
# ----------------------------------------------------------------------------
CAREER_DATABANK = {
    "finance": [
        {"title": "Financial Analyst", "why_it_fits": "Core path for finance majors", "salary_range": "$60k-$80k", "day_in_the_life": "Build models, analyze statements, support investment decisions", "how_to_get_there": "Apply to rotational analyst programs", "growth_outlook": "Stable, competitive"},
        {"title": "Investment Banking Analyst", "why_it_fits": "Highest-paying direct finance track", "salary_range": "$95k-$150k+", "day_in_the_life": "Build pitch decks, model deals, long hours", "how_to_get_there": "Target recruiting at bulge-bracket banks", "growth_outlook": "Competitive, cyclical"},
        {"title": "Corporate FP&A Analyst", "why_it_fits": "In-house finance, better hours than banking", "salary_range": "$65k-$90k", "day_in_the_life": "Build budgets, forecast revenue, present to leadership", "how_to_get_there": "Apply to FP&A rotational programs", "growth_outlook": "Stable"},
        {"title": "Commercial Banking Analyst", "why_it_fits": "Relationship-driven finance track", "salary_range": "$60k-$85k", "day_in_the_life": "Underwrite loans, manage client relationships", "how_to_get_there": "Apply to regional bank credit programs", "growth_outlook": "Stable"},
        {"title": "Equity Research Associate", "why_it_fits": "Analytical, markets-focused", "salary_range": "$75k-$110k", "day_in_the_life": "Research companies, write reports, build models", "how_to_get_there": "Network into a sell-side research team", "growth_outlook": "Competitive"},
        {"title": "Wealth Management Advisor", "why_it_fits": "Client-facing, uncapped upside", "salary_range": "$55k-$90k+ (plus commission)", "day_in_the_life": "Manage client portfolios, build referral network", "how_to_get_there": "Get Series 7/66 licensed", "growth_outlook": "Growing"},
        {"title": "Credit Analyst", "why_it_fits": "Risk-focused finance role", "salary_range": "$55k-$75k", "day_in_the_life": "Assess borrower risk, write credit memos", "how_to_get_there": "Apply to bank credit analyst programs", "growth_outlook": "Stable"},
        {"title": "Treasury Analyst", "why_it_fits": "Cash and liquidity management", "salary_range": "$60k-$80k", "day_in_the_life": "Manage company cash, forecast liquidity needs", "how_to_get_there": "Apply to corporate treasury teams", "growth_outlook": "Stable"},
        {"title": "Private Equity Analyst", "why_it_fits": "Highest-prestige buy-side track", "salary_range": "$100k-$150k+", "day_in_the_life": "Evaluate deals, build LBO models", "how_to_get_there": "Usually recruited from investment banking", "growth_outlook": "Very competitive"},
        {"title": "Insurance Underwriter", "why_it_fits": "Stable, analytical, good work-life balance", "salary_range": "$55k-$75k", "day_in_the_life": "Assess risk, price policies", "how_to_get_there": "Apply directly to insurance carriers", "growth_outlook": "Stable"},
        {"title": "Real Estate Analyst", "why_it_fits": "Tangible-asset finance track", "salary_range": "$60k-$85k", "day_in_the_life": "Underwrite deals, model property returns", "how_to_get_there": "Apply to REITs or real estate PE firms", "growth_outlook": "Growing"},
        {"title": "Financial Planner", "why_it_fits": "Personal finance, client relationships", "salary_range": "$50k-$80k+ (plus fees)", "day_in_the_life": "Build financial plans for individual clients", "how_to_get_there": "Get CFP certification over time", "growth_outlook": "Growing"},
        {"title": "Actuarial Analyst", "why_it_fits": "Highly analytical, strong work-life balance", "salary_range": "$65k-$85k", "day_in_the_life": "Model risk and pricing for insurers", "how_to_get_there": "Pass actuarial exams while working", "growth_outlook": "Growing steadily"},
        {"title": "Corporate Development Analyst", "why_it_fits": "M&A-focused, in-house strategy", "salary_range": "$75k-$100k", "day_in_the_life": "Evaluate acquisition targets, model synergies", "how_to_get_there": "Often a step up from banking/consulting", "growth_outlook": "Competitive"},
        {"title": "Risk Management Analyst", "why_it_fits": "Growing, in-demand specialty", "salary_range": "$65k-$90k", "day_in_the_life": "Model and monitor financial risk exposure", "how_to_get_there": "Apply to bank/corporate risk teams", "growth_outlook": "Growing"},
    ],
    "economics": [
        {"title": "Economic Analyst", "why_it_fits": "Direct application of econ training", "salary_range": "$60k-$85k", "day_in_the_life": "Analyze economic data, write reports", "how_to_get_there": "Apply to consulting firms or government agencies", "growth_outlook": "Stable"},
        {"title": "Policy Analyst", "why_it_fits": "Applies economics to real-world policy", "salary_range": "$55k-$80k", "day_in_the_life": "Research policy impact, brief decision-makers", "how_to_get_there": "Apply to think tanks or government roles", "growth_outlook": "Stable"},
        {"title": "Management Consultant", "why_it_fits": "Popular econ-major destination", "salary_range": "$85k-$110k", "day_in_the_life": "Solve client business problems, travel frequently", "how_to_get_there": "Target consulting firm recruiting", "growth_outlook": "Competitive"},
        {"title": "Data Analyst", "why_it_fits": "Quantitative econ skills transfer directly", "salary_range": "$60k-$85k", "day_in_the_life": "Clean data, build dashboards, find trends", "how_to_get_there": "Build a portfolio with SQL/Python projects", "growth_outlook": "Growing fast"},
        {"title": "Market Research Analyst", "why_it_fits": "Applies economic reasoning to consumer behavior", "salary_range": "$55k-$75k", "day_in_the_life": "Design surveys, analyze market trends", "how_to_get_there": "Apply directly to research or marketing teams", "growth_outlook": "Stable"},
        {"title": "Financial Analyst", "why_it_fits": "Common econ-to-finance crossover", "salary_range": "$60k-$80k", "day_in_the_life": "Build models, analyze statements", "how_to_get_there": "Apply to rotational analyst programs", "growth_outlook": "Stable"},
        {"title": "Actuarial Analyst", "why_it_fits": "Quantitative, strong work-life balance", "salary_range": "$65k-$85k", "day_in_the_life": "Model risk and pricing for insurers", "how_to_get_there": "Pass actuarial exams while working", "growth_outlook": "Growing steadily"},
        {"title": "Urban/Regional Planner", "why_it_fits": "Applies economics to city-level decisions", "salary_range": "$55k-$75k", "day_in_the_life": "Analyze zoning, transit, and growth data", "how_to_get_there": "Often pairs well with a planning master's", "growth_outlook": "Stable"},
        {"title": "Supply Chain Analyst", "why_it_fits": "Economics of logistics and operations", "salary_range": "$60k-$80k", "day_in_the_life": "Optimize inventory and distribution decisions", "how_to_get_there": "Apply to retail/manufacturing analyst programs", "growth_outlook": "Growing"},
        {"title": "Compensation Analyst", "why_it_fits": "Applies labor economics directly", "salary_range": "$55k-$75k", "day_in_the_life": "Benchmark pay, model compensation structures", "how_to_get_there": "Apply to corporate HR/total rewards teams", "growth_outlook": "Stable"},
        {"title": "Underwriter", "why_it_fits": "Risk-pricing, analytical fit", "salary_range": "$55k-$75k", "day_in_the_life": "Assess risk, price policies or loans", "how_to_get_there": "Apply directly to insurers or lenders", "growth_outlook": "Stable"},
        {"title": "International Trade Analyst", "why_it_fits": "Applies macro/trade theory directly", "salary_range": "$55k-$80k", "day_in_the_life": "Analyze tariffs, trade flows, compliance", "how_to_get_there": "Apply to trade-focused firms or government", "growth_outlook": "Stable"},
        {"title": "Real Estate Analyst", "why_it_fits": "Applies micro/market analysis", "salary_range": "$60k-$85k", "day_in_the_life": "Underwrite deals, model property returns", "how_to_get_there": "Apply to REITs or brokerage firms", "growth_outlook": "Growing"},
        {"title": "Economic Consultant", "why_it_fits": "Applies econ theory to litigation/business", "salary_range": "$65k-$95k", "day_in_the_life": "Build economic models for legal/business cases", "how_to_get_there": "Apply to economic consulting firms", "growth_outlook": "Stable"},
        {"title": "Credit Risk Analyst", "why_it_fits": "Quantitative risk assessment", "salary_range": "$60k-$80k", "day_in_the_life": "Model borrower default risk", "how_to_get_there": "Apply to bank risk teams", "growth_outlook": "Growing"},
    ],
    "computer science": [
        {"title": "Software Engineer", "why_it_fits": "Core CS career path", "salary_range": "$75k-$110k", "day_in_the_life": "Write, test, and ship production code", "how_to_get_there": "Build projects, grind interview prep", "growth_outlook": "Growing fast"},
        {"title": "Data Engineer", "why_it_fits": "High-demand data infrastructure role", "salary_range": "$80k-$115k", "day_in_the_life": "Build pipelines that move and clean data", "how_to_get_there": "Learn SQL, Spark, cloud data tools", "growth_outlook": "Growing fast"},
        {"title": "Machine Learning Engineer", "why_it_fits": "Highest-growth CS specialty right now", "salary_range": "$95k-$140k", "day_in_the_life": "Build and deploy ML models into production", "how_to_get_there": "Build ML projects, learn PyTorch/TensorFlow", "growth_outlook": "Growing very fast"},
        {"title": "DevOps/Cloud Engineer", "why_it_fits": "Infrastructure-focused, in demand", "salary_range": "$85k-$120k", "day_in_the_life": "Manage deployment pipelines and cloud infra", "how_to_get_there": "Get AWS/Azure certifications", "growth_outlook": "Growing fast"},
        {"title": "Security Engineer", "why_it_fits": "High-demand, well-compensated specialty", "salary_range": "$90k-$130k", "day_in_the_life": "Find and fix security vulnerabilities", "how_to_get_there": "Build a security-focused portfolio, get certs", "growth_outlook": "Growing very fast"},
        {"title": "Mobile App Developer", "why_it_fits": "Consumer-facing CS specialty", "salary_range": "$75k-$105k", "day_in_the_life": "Build iOS/Android app features", "how_to_get_there": "Ship a real app to the App Store", "growth_outlook": "Stable, competitive"},
        {"title": "Product Manager (Technical)", "why_it_fits": "For CS majors who like strategy over pure code", "salary_range": "$85k-$120k", "day_in_the_life": "Define product specs, work with engineers", "how_to_get_there": "Often a step from engineering after 2-3 years", "growth_outlook": "Competitive"},
        {"title": "Game Developer", "why_it_fits": "For CS majors with a creative/gaming interest", "salary_range": "$65k-$95k", "day_in_the_life": "Build gameplay systems and engine features", "how_to_get_there": "Ship a portfolio game, apply to studios", "growth_outlook": "Stable, competitive"},
        {"title": "Site Reliability Engineer", "why_it_fits": "Systems-focused, well-paid", "salary_range": "$90k-$130k", "day_in_the_life": "Keep production systems fast and reliable", "how_to_get_there": "Build strong systems/networking fundamentals", "growth_outlook": "Growing fast"},
        {"title": "Data Scientist", "why_it_fits": "Blends CS with statistics", "salary_range": "$85k-$120k", "day_in_the_life": "Analyze data, build predictive models", "how_to_get_there": "Build a portfolio of real data projects", "growth_outlook": "Growing fast"},
        {"title": "Full-Stack Developer", "why_it_fits": "Broad, flexible CS career path", "salary_range": "$70k-$105k", "day_in_the_life": "Build both frontend and backend features", "how_to_get_there": "Build and deploy full projects end-to-end", "growth_outlook": "Growing"},
        {"title": "QA/Test Engineer", "why_it_fits": "Entry point into software with less grind", "salary_range": "$60k-$85k", "day_in_the_life": "Write automated tests, find bugs before users do", "how_to_get_there": "Learn test automation frameworks", "growth_outlook": "Stable"},
        {"title": "Solutions Architect", "why_it_fits": "For CS majors who like client-facing work", "salary_range": "$90k-$130k", "day_in_the_life": "Design technical solutions for enterprise clients", "how_to_get_there": "Usually a step up after a few years engineering", "growth_outlook": "Growing"},
        {"title": "Embedded Systems Engineer", "why_it_fits": "Hardware-adjacent CS specialty", "salary_range": "$75k-$105k", "day_in_the_life": "Write low-level code for physical devices", "how_to_get_there": "Build projects with microcontrollers", "growth_outlook": "Stable"},
        {"title": "IT Consultant", "why_it_fits": "CS knowledge applied to business problems", "salary_range": "$70k-$100k", "day_in_the_life": "Advise companies on technology decisions", "how_to_get_there": "Apply to tech consulting firms", "growth_outlook": "Stable"},
    ],
    "psychology": [
        {"title": "HR Generalist", "why_it_fits": "Applies people-focused psych training", "salary_range": "$50k-$70k", "day_in_the_life": "Handle hiring, employee relations, policy", "how_to_get_there": "Apply to corporate HR rotational programs", "growth_outlook": "Stable"},
        {"title": "Market Research Analyst", "why_it_fits": "Applies behavioral insight to consumer data", "salary_range": "$55k-$75k", "day_in_the_life": "Design surveys, analyze consumer behavior", "how_to_get_there": "Apply directly to research or marketing teams", "growth_outlook": "Stable"},
        {"title": "UX Researcher", "why_it_fits": "Directly applies psychology to product design", "salary_range": "$70k-$100k", "day_in_the_life": "Run user studies, translate findings into design", "how_to_get_there": "Build a portfolio of real research projects", "growth_outlook": "Growing"},
        {"title": "School Counselor", "why_it_fits": "Direct psych-to-career path", "salary_range": "$50k-$65k", "day_in_the_life": "Support students academically and emotionally", "how_to_get_there": "Requires a master's in school counseling", "growth_outlook": "Stable"},
        {"title": "Case Manager", "why_it_fits": "People-focused, helping-profession fit", "salary_range": "$40k-$55k", "day_in_the_life": "Connect clients with social services and support", "how_to_get_there": "Apply to social service agencies", "growth_outlook": "Growing"},
        {"title": "Recruiter", "why_it_fits": "Reads people well, fast-paced", "salary_range": "$45k-$70k+ (plus commission)", "day_in_the_life": "Source, interview, and place candidates", "how_to_get_there": "Apply to staffing firms or corporate TA teams", "growth_outlook": "Stable"},
        {"title": "Behavioral Analyst (ABA)", "why_it_fits": "Direct clinical application of psychology", "salary_range": "$45k-$65k", "day_in_the_life": "Work directly with clients on behavior plans", "how_to_get_there": "Get RBT certification, work toward BCBA", "growth_outlook": "Growing fast"},
        {"title": "Training & Development Specialist", "why_it_fits": "Applies learning psychology in workplaces", "salary_range": "$50k-$70k", "day_in_the_life": "Design and deliver employee training", "how_to_get_there": "Apply to corporate L&D teams", "growth_outlook": "Stable"},
        {"title": "Probation Officer", "why_it_fits": "Applies psych to behavior/rehabilitation", "salary_range": "$45k-$65k", "day_in_the_life": "Supervise and support people in the justice system", "how_to_get_there": "Apply to state/county probation departments", "growth_outlook": "Stable"},
        {"title": "Sales Representative", "why_it_fits": "Understanding people drives sales success", "salary_range": "$45k-$70k+ (plus commission)", "day_in_the_life": "Build relationships, close deals", "how_to_get_there": "Apply to B2B sales development programs", "growth_outlook": "Stable"},
        {"title": "Social Media Strategist", "why_it_fits": "Applies behavioral insight to content/audience", "salary_range": "$45k-$65k", "day_in_the_life": "Plan content, analyze audience engagement", "how_to_get_there": "Build a portfolio managing real accounts", "growth_outlook": "Growing"},
        {"title": "Nonprofit Program Coordinator", "why_it_fits": "Mission-driven, people-centered work", "salary_range": "$40k-$55k", "day_in_the_life": "Run programs that serve a community need", "how_to_get_there": "Apply directly to nonprofits", "growth_outlook": "Stable"},
        {"title": "Psychiatric Technician", "why_it_fits": "Direct clinical exposure without a grad degree yet", "salary_range": "$35k-$50k", "day_in_the_life": "Support patients in psychiatric care settings", "how_to_get_there": "Apply to hospitals/treatment centers", "growth_outlook": "Growing"},
        {"title": "User Experience (UX) Designer", "why_it_fits": "Psych insight applied to interface design", "salary_range": "$65k-$95k", "day_in_the_life": "Design interfaces informed by how people think", "how_to_get_there": "Build a design portfolio, learn Figma", "growth_outlook": "Growing"},
        {"title": "Compliance/Ethics Analyst", "why_it_fits": "Understanding behavior applied to policy", "salary_range": "$55k-$75k", "day_in_the_life": "Monitor and support ethical workplace practices", "how_to_get_there": "Apply to corporate compliance teams", "growth_outlook": "Stable"},
    ],
    "nursing": [
        {"title": "Registered Nurse", "why_it_fits": "The core, direct nursing career path", "salary_range": "$65k-$85k", "day_in_the_life": "Assess patients, administer care, coordinate with doctors", "how_to_get_there": "Pass the NCLEX after graduation", "growth_outlook": "Growing fast"},
        {"title": "ICU/Critical Care Nurse", "why_it_fits": "Higher-acuity specialty for confident RNs", "salary_range": "$70k-$95k", "day_in_the_life": "Monitor critically ill patients, manage complex cases", "how_to_get_there": "Gain 1-2 years RN experience first", "growth_outlook": "Growing"},
        {"title": "Emergency Room Nurse", "why_it_fits": "Fast-paced, high-variety nursing specialty", "salary_range": "$65k-$90k", "day_in_the_life": "Triage and treat urgent, unpredictable cases", "how_to_get_there": "Apply directly to ER departments", "growth_outlook": "Growing"},
        {"title": "Pediatric Nurse", "why_it_fits": "For nurses drawn to working with children", "salary_range": "$60k-$80k", "day_in_the_life": "Care for infants through teens in clinical settings", "how_to_get_there": "Seek pediatric unit placements", "growth_outlook": "Stable"},
        {"title": "Nurse Case Manager", "why_it_fits": "Less bedside, more care coordination", "salary_range": "$65k-$85k", "day_in_the_life": "Coordinate patient care across providers and insurers", "how_to_get_there": "Move into case management after RN experience", "growth_outlook": "Stable"},
        {"title": "Public Health Nurse", "why_it_fits": "Community-level rather than bedside care", "salary_range": "$55k-$75k", "day_in_the_life": "Run community health programs and screenings", "how_to_get_there": "Apply to public health departments", "growth_outlook": "Growing"},
        {"title": "Operating Room Nurse", "why_it_fits": "Precision-focused surgical specialty", "salary_range": "$70k-$95k", "day_in_the_life": "Assist surgeons, manage sterile procedures", "how_to_get_there": "Seek perioperative training programs", "growth_outlook": "Stable"},
        {"title": "Oncology Nurse", "why_it_fits": "Specialized, relationship-driven cancer care", "salary_range": "$65k-$90k", "day_in_the_life": "Administer treatment, support patients long-term", "how_to_get_there": "Gain oncology certification after RN licensure", "growth_outlook": "Growing"},
        {"title": "Psychiatric Nurse", "why_it_fits": "For nurses drawn to mental health care", "salary_range": "$65k-$85k", "day_in_the_life": "Support patients through mental health treatment", "how_to_get_there": "Seek psychiatric unit placements", "growth_outlook": "Growing"},
        {"title": "Home Health Nurse", "why_it_fits": "More autonomy, one-on-one patient care", "salary_range": "$60k-$80k", "day_in_the_life": "Provide in-home care to recovering or chronic patients", "how_to_get_there": "Apply to home health agencies", "growth_outlook": "Growing fast"},
        {"title": "School Nurse", "why_it_fits": "Predictable schedule, community-based care", "salary_range": "$50k-$65k", "day_in_the_life": "Manage student health needs during the school day", "how_to_get_there": "Apply directly to school districts", "growth_outlook": "Stable"},
        {"title": "Clinical Research Nurse", "why_it_fits": "Blends nursing with clinical trial work", "salary_range": "$65k-$90k", "day_in_the_life": "Monitor patients enrolled in clinical trials", "how_to_get_there": "Apply to hospital or pharma research teams", "growth_outlook": "Growing"},
        {"title": "Nurse Educator", "why_it_fits": "For RNs who want to teach the next generation", "salary_range": "$65k-$85k", "day_in_the_life": "Teach nursing students in classroom and clinical settings", "how_to_get_there": "Often requires a graduate nursing degree", "growth_outlook": "Growing"},
        {"title": "Travel Nurse", "why_it_fits": "Higher pay in exchange for short-term relocations", "salary_range": "$80k-$120k", "day_in_the_life": "Work short-term contracts at hospitals nationwide", "how_to_get_there": "Gain 1-2 years experience, then join a travel agency", "growth_outlook": "Growing"},
        {"title": "Nurse Practitioner", "why_it_fits": "Highest-autonomy nursing path, needs more schooling", "salary_range": "$95k-$125k", "day_in_the_life": "Diagnose and treat patients with prescribing authority", "how_to_get_there": "Requires an MSN or DNP after your RN", "growth_outlook": "Growing very fast"},
    ],
    "marketing": [
        {"title": "Marketing Coordinator", "why_it_fits": "The standard entry point into marketing", "salary_range": "$45k-$60k", "day_in_the_life": "Support campaigns, manage timelines and assets", "how_to_get_there": "Apply to marketing coordinator/associate roles", "growth_outlook": "Stable"},
        {"title": "Social Media Manager", "why_it_fits": "Direct, visible application of marketing skills", "salary_range": "$45k-$65k", "day_in_the_life": "Plan content, engage audiences, track engagement", "how_to_get_there": "Build a portfolio managing real accounts", "growth_outlook": "Growing"},
        {"title": "Digital Marketing Specialist", "why_it_fits": "High-demand, data-adjacent marketing track", "salary_range": "$50k-$70k", "day_in_the_life": "Run paid campaigns, track conversion metrics", "how_to_get_there": "Learn Google Ads and analytics tools", "growth_outlook": "Growing fast"},
        {"title": "Market Research Analyst", "why_it_fits": "For marketing students who like the data side", "salary_range": "$50k-$70k", "day_in_the_life": "Design surveys, analyze consumer trends", "how_to_get_there": "Apply directly to research or brand teams", "growth_outlook": "Stable"},
        {"title": "Brand Manager", "why_it_fits": "A common mid-career marketing destination", "salary_range": "$60k-$85k", "day_in_the_life": "Own a brand's positioning and campaign strategy", "how_to_get_there": "Usually a step up after 2-3 years in marketing", "growth_outlook": "Stable"},
        {"title": "Content Marketing Manager", "why_it_fits": "For marketers who like writing and storytelling", "salary_range": "$55k-$75k", "day_in_the_life": "Plan and produce content across channels", "how_to_get_there": "Build a portfolio of real content work", "growth_outlook": "Growing"},
        {"title": "Public Relations Specialist", "why_it_fits": "Closely adjacent to marketing, more media-facing", "salary_range": "$45k-$65k", "day_in_the_life": "Manage media relationships and brand reputation", "how_to_get_there": "Apply to PR agencies or in-house comms teams", "growth_outlook": "Stable"},
        {"title": "SEO/SEM Specialist", "why_it_fits": "A technical, high-demand marketing specialty", "salary_range": "$50k-$70k", "day_in_the_life": "Optimize content and ads for search visibility", "how_to_get_there": "Learn SEO tools and Google Ads certification", "growth_outlook": "Growing fast"},
        {"title": "Product Marketing Manager", "why_it_fits": "Blends marketing with product strategy", "salary_range": "$65k-$90k", "day_in_the_life": "Position products, enable sales teams", "how_to_get_there": "Often a step up from marketing or PM roles", "growth_outlook": "Growing"},
        {"title": "Marketing Analytics Manager", "why_it_fits": "For marketers who lean quantitative", "salary_range": "$60k-$85k", "day_in_the_life": "Measure campaign ROI, build reporting dashboards", "how_to_get_there": "Build strong Excel/SQL skills alongside marketing", "growth_outlook": "Growing"},
        {"title": "Advertising Account Executive", "why_it_fits": "Client-facing marketing/agency track", "salary_range": "$50k-$70k", "day_in_the_life": "Manage client campaigns and relationships", "how_to_get_there": "Apply to advertising agencies", "growth_outlook": "Stable"},
        {"title": "E-commerce Marketing Manager", "why_it_fits": "High-growth, retail-focused marketing specialty", "salary_range": "$55k-$80k", "day_in_the_life": "Drive online sales through campaigns and promotions", "how_to_get_there": "Apply to retail or DTC brand marketing teams", "growth_outlook": "Growing fast"},
        {"title": "Growth Marketing Manager", "why_it_fits": "For marketers who like experimentation and data", "salary_range": "$65k-$95k", "day_in_the_life": "Run rapid tests to drive user acquisition", "how_to_get_there": "Build a track record of measurable campaign wins", "growth_outlook": "Growing fast"},
        {"title": "Email Marketing Specialist", "why_it_fits": "A focused, measurable marketing channel", "salary_range": "$45k-$65k", "day_in_the_life": "Build campaigns, segment audiences, test messaging", "how_to_get_there": "Learn a major email platform like Klaviyo or HubSpot", "growth_outlook": "Stable"},
        {"title": "Influencer/Partnerships Manager", "why_it_fits": "For marketers drawn to relationship building", "salary_range": "$50k-$70k", "day_in_the_life": "Recruit and manage brand partnerships", "how_to_get_there": "Build relationships in a specific industry niche", "growth_outlook": "Growing"},
    ],
    "biology": [
        {"title": "Lab Technician", "why_it_fits": "The most common entry-level biology role", "salary_range": "$40k-$55k", "day_in_the_life": "Run samples, maintain lab equipment, log results", "how_to_get_there": "Apply directly to university or company labs", "growth_outlook": "Stable"},
        {"title": "Research Assistant", "why_it_fits": "Direct extension of undergraduate lab work", "salary_range": "$38k-$52k", "day_in_the_life": "Support faculty or industry research projects", "how_to_get_there": "Build relationships with professors during school", "growth_outlook": "Stable"},
        {"title": "Clinical Research Coordinator", "why_it_fits": "Applies biology to real clinical trials", "salary_range": "$48k-$65k", "day_in_the_life": "Manage trial logistics and patient enrollment", "how_to_get_there": "Apply to hospital or pharma research teams", "growth_outlook": "Growing"},
        {"title": "Environmental Scientist", "why_it_fits": "For biology majors drawn to ecology/conservation", "salary_range": "$50k-$70k", "day_in_the_life": "Collect field data, assess environmental impact", "how_to_get_there": "Apply to environmental consulting or government roles", "growth_outlook": "Growing"},
        {"title": "Biotech Quality Control Analyst", "why_it_fits": "Applies lab skills in a manufacturing setting", "salary_range": "$50k-$68k", "day_in_the_life": "Test products against safety and quality standards", "how_to_get_there": "Apply to biotech or pharma manufacturing", "growth_outlook": "Growing"},
        {"title": "Pharmaceutical Sales Representative", "why_it_fits": "For biology majors who like people over the bench", "salary_range": "$55k-$80k", "day_in_the_life": "Educate providers on pharmaceutical products", "how_to_get_there": "Apply directly to pharma sales teams", "growth_outlook": "Stable"},
        {"title": "Science Teacher", "why_it_fits": "A stable, direct application of a biology degree", "salary_range": "$42k-$60k", "day_in_the_life": "Teach biology or general science to students", "how_to_get_there": "May require a teaching certification", "growth_outlook": "Stable"},
        {"title": "Wildlife Biologist", "why_it_fits": "For biology majors drawn to field work", "salary_range": "$42k-$60k", "day_in_the_life": "Study animal populations and habitats in the field", "how_to_get_there": "Apply to state wildlife agencies or nonprofits", "growth_outlook": "Stable"},
        {"title": "Food Scientist", "why_it_fits": "Applies biology/chemistry to food production", "salary_range": "$50k-$70k", "day_in_the_life": "Test and develop food products for safety and quality", "how_to_get_there": "Apply to food manufacturing companies", "growth_outlook": "Growing"},
        {"title": "Genetic Counseling Assistant", "why_it_fits": "A stepping stone into genetics-focused healthcare", "salary_range": "$45k-$60k", "day_in_the_life": "Support genetic counselors with patient case prep", "how_to_get_there": "Apply to hospital genetics departments", "growth_outlook": "Growing"},
        {"title": "Forensic Science Technician", "why_it_fits": "Applies lab skills to criminal investigations", "salary_range": "$48k-$65k", "day_in_the_life": "Analyze evidence samples for law enforcement", "how_to_get_there": "Apply to crime labs or law enforcement agencies", "growth_outlook": "Growing"},
        {"title": "Public Health Analyst", "why_it_fits": "Applies biology to population-level health data", "salary_range": "$48k-$68k", "day_in_the_life": "Analyze health trends and program outcomes", "how_to_get_there": "Apply to public health departments or nonprofits", "growth_outlook": "Growing"},
        {"title": "Science Writer", "why_it_fits": "For biology majors who like communicating science", "salary_range": "$45k-$65k", "day_in_the_life": "Translate research into articles for broader audiences", "how_to_get_there": "Build a portfolio of science writing samples", "growth_outlook": "Stable"},
        {"title": "Biology Lab Manager", "why_it_fits": "A step up from lab technician with more ownership", "salary_range": "$50k-$70k", "day_in_the_life": "Oversee lab operations, supplies, and staff", "how_to_get_there": "Usually follows a few years as a lab technician", "growth_outlook": "Stable"},
        {"title": "Medical Sales Representative", "why_it_fits": "For biology majors who like sales over research", "salary_range": "$60k-$90k", "day_in_the_life": "Sell medical devices or products to providers", "how_to_get_there": "Apply directly to medical device companies", "growth_outlook": "Growing"},
    ],
    "engineering": [
        {"title": "Mechanical Engineer", "why_it_fits": "One of the most common engineering career paths", "salary_range": "$65k-$85k", "day_in_the_life": "Design and test mechanical systems and components", "how_to_get_there": "Apply to engineering rotational programs", "growth_outlook": "Stable"},
        {"title": "Civil Engineer", "why_it_fits": "For engineers drawn to infrastructure and construction", "salary_range": "$60k-$80k", "day_in_the_life": "Design and oversee infrastructure projects", "how_to_get_there": "Apply to civil/construction engineering firms", "growth_outlook": "Stable"},
        {"title": "Electrical Engineer", "why_it_fits": "For engineers drawn to circuits and power systems", "salary_range": "$65k-$90k", "day_in_the_life": "Design and test electrical systems and components", "how_to_get_there": "Apply to electrical engineering teams", "growth_outlook": "Growing"},
        {"title": "Manufacturing/Process Engineer", "why_it_fits": "Applies engineering to production efficiency", "salary_range": "$65k-$85k", "day_in_the_life": "Optimize production lines and processes", "how_to_get_there": "Apply to manufacturing companies directly", "growth_outlook": "Stable"},
        {"title": "Quality Engineer", "why_it_fits": "For engineers who like process and standards work", "salary_range": "$60k-$80k", "day_in_the_life": "Ensure products meet quality and safety standards", "how_to_get_there": "Apply to manufacturing or industrial companies", "growth_outlook": "Stable"},
        {"title": "Project Engineer", "why_it_fits": "Blends technical work with project coordination", "salary_range": "$60k-$85k", "day_in_the_life": "Coordinate timelines, budgets, and technical teams", "how_to_get_there": "Often a step up after a few years as an engineer", "growth_outlook": "Stable"},
        {"title": "Systems Engineer", "why_it_fits": "For engineers who like big-picture technical design", "salary_range": "$70k-$95k", "day_in_the_life": "Design how complex systems and components interact", "how_to_get_there": "Apply to systems engineering teams", "growth_outlook": "Growing"},
        {"title": "Design Engineer", "why_it_fits": "Hands-on technical design and prototyping work", "salary_range": "$65k-$85k", "day_in_the_life": "Design and prototype new products or components", "how_to_get_there": "Build a portfolio of real design projects", "growth_outlook": "Stable"},
        {"title": "Structural Engineer", "why_it_fits": "A specialized civil engineering track", "salary_range": "$62k-$85k", "day_in_the_life": "Analyze and design load-bearing structures", "how_to_get_there": "Often requires PE licensure over time", "growth_outlook": "Stable"},
        {"title": "Sales Engineer", "why_it_fits": "For engineers who like people over the bench", "salary_range": "$65k-$95k", "day_in_the_life": "Explain technical products to prospective clients", "how_to_get_there": "Apply to technical sales teams", "growth_outlook": "Growing"},
        {"title": "Environmental Engineer", "why_it_fits": "Applies engineering to sustainability challenges", "salary_range": "$60k-$80k", "day_in_the_life": "Design systems to reduce environmental impact", "how_to_get_there": "Apply to environmental engineering firms", "growth_outlook": "Growing"},
        {"title": "Robotics Engineer", "why_it_fits": "For engineers drawn to automation and robotics", "salary_range": "$70k-$100k", "day_in_the_life": "Design and program robotic systems", "how_to_get_there": "Build robotics projects during school", "growth_outlook": "Growing fast"},
        {"title": "R&D Engineer", "why_it_fits": "For engineers who like inventing new solutions", "salary_range": "$70k-$95k", "day_in_the_life": "Prototype and test new product concepts", "how_to_get_there": "Apply to company R&D departments", "growth_outlook": "Growing"},
        {"title": "Field Engineer", "why_it_fits": "For engineers who prefer on-site over office work", "salary_range": "$60k-$85k", "day_in_the_life": "Install, test, and troubleshoot equipment on-site", "how_to_get_there": "Apply to field service engineering roles", "growth_outlook": "Stable"},
        {"title": "Engineering Project Manager", "why_it_fits": "A management path leveraging technical background", "salary_range": "$75k-$100k", "day_in_the_life": "Lead cross-functional engineering project teams", "how_to_get_there": "Usually a step up after several years engineering", "growth_outlook": "Growing"},
    ],
    "communication": [
        {"title": "Public Relations Specialist", "why_it_fits": "A core, direct application of communication skills", "salary_range": "$45k-$65k", "day_in_the_life": "Manage media relationships and brand reputation", "how_to_get_there": "Apply to PR agencies or in-house comms teams", "growth_outlook": "Stable"},
        {"title": "Social Media Manager", "why_it_fits": "Direct, visible application of communication skills", "salary_range": "$45k-$65k", "day_in_the_life": "Plan content, engage audiences, track engagement", "how_to_get_there": "Build a portfolio managing real accounts", "growth_outlook": "Growing"},
        {"title": "Communications Coordinator", "why_it_fits": "The standard entry point into comms roles", "salary_range": "$42k-$60k", "day_in_the_life": "Draft internal and external messaging", "how_to_get_there": "Apply to corporate or nonprofit comms teams", "growth_outlook": "Stable"},
        {"title": "Content Creator/Writer", "why_it_fits": "For communication majors who like writing", "salary_range": "$42k-$62k", "day_in_the_life": "Write and produce content across platforms", "how_to_get_there": "Build a portfolio of published writing samples", "growth_outlook": "Growing"},
        {"title": "Corporate Communications Specialist", "why_it_fits": "Applies comms skills inside a company", "salary_range": "$50k-$70k", "day_in_the_life": "Manage internal messaging and executive communications", "how_to_get_there": "Apply to corporate communications teams", "growth_outlook": "Stable"},
        {"title": "Marketing Coordinator", "why_it_fits": "Closely adjacent field, heavy skill overlap", "salary_range": "$45k-$60k", "day_in_the_life": "Support campaigns, manage timelines and assets", "how_to_get_there": "Apply to marketing coordinator roles", "growth_outlook": "Stable"},
        {"title": "Event Coordinator", "why_it_fits": "For communication majors who like live events", "salary_range": "$42k-$58k", "day_in_the_life": "Plan and execute corporate or public events", "how_to_get_there": "Apply to event planning agencies or in-house teams", "growth_outlook": "Stable"},
        {"title": "Broadcast/Media Producer", "why_it_fits": "For comm majors drawn to video/audio production", "salary_range": "$45k-$65k", "day_in_the_life": "Produce segments for broadcast or digital media", "how_to_get_there": "Build a demo reel during school", "growth_outlook": "Stable"},
        {"title": "Human Resources Coordinator", "why_it_fits": "Applies communication skills to people operations", "salary_range": "$45k-$62k", "day_in_the_life": "Support hiring, onboarding, and employee communication", "how_to_get_there": "Apply to corporate HR teams", "growth_outlook": "Stable"},
        {"title": "Journalist/Reporter", "why_it_fits": "For comm majors most drawn to reporting", "salary_range": "$38k-$55k", "day_in_the_life": "Research, write, and publish news stories", "how_to_get_there": "Build clips through a student paper or internship", "growth_outlook": "Stable"},
        {"title": "Media Relations Specialist", "why_it_fits": "A specialized, press-facing communication role", "salary_range": "$48k-$68k", "day_in_the_life": "Pitch stories and manage press inquiries", "how_to_get_there": "Apply to PR agencies or press offices", "growth_outlook": "Stable"},
        {"title": "Internal Communications Manager", "why_it_fits": "A mid-career comms specialty inside companies", "salary_range": "$55k-$75k", "day_in_the_life": "Manage employee-facing messaging and culture comms", "how_to_get_there": "Usually a step up after a few years in comms", "growth_outlook": "Growing"},
        {"title": "Nonprofit Communications Manager", "why_it_fits": "For comm majors drawn to mission-driven work", "salary_range": "$42k-$60k", "day_in_the_life": "Manage messaging and outreach for a cause", "how_to_get_there": "Apply directly to nonprofit organizations", "growth_outlook": "Stable"},
        {"title": "Digital Content Strategist", "why_it_fits": "Blends comms with digital/data thinking", "salary_range": "$50k-$70k", "day_in_the_life": "Plan content strategy across digital channels", "how_to_get_there": "Build a portfolio of measurable content work", "growth_outlook": "Growing"},
        {"title": "Speechwriter", "why_it_fits": "A specialized, high-skill writing niche", "salary_range": "$50k-$72k", "day_in_the_life": "Write speeches and talking points for executives", "how_to_get_there": "Build a strong persuasive writing portfolio", "growth_outlook": "Stable"},
    ],
}


def normalize_major(name: str) -> str:
    """Match a free-text major name to a databank key — handles common
    real-world variations (abbreviations, 'and' vs '&', extra words)."""
    n = (name or "").lower().strip()
    aliases = {
        "cs": "computer science", "comp sci": "computer science", "computer sci": "computer science",
        "econ": "economics", "fin": "finance",
        "psych": "psychology",
    }
    n = aliases.get(n, n)
    for key in CAREER_DATABANK:
        if key in n or n in key:
            return key
    return n


# ----------------------------------------------------------------------------
# Common switches databank — the most genuinely common, well-known switch
# destinations from a given major, based on real adjacent-field/coursework
# overlap patterns, not a personalized-to-you AI guess. Static and instant:
# zero AI call for the browse list itself, which is what actually caused the
# wait when clicking through several of the 10 in a row.
# ----------------------------------------------------------------------------
COMMON_SWITCHES = {
    "finance": [
        {"name": "Economics", "fit_percentage": 88, "one_liner": "Heaviest course overlap of any switch from Finance", "salary_range": "$60k-$85k"},
        {"name": "Management Information Systems", "fit_percentage": 82, "one_liner": "Same business core, adds real tech/data skills", "salary_range": "$65k-$95k"},
        {"name": "Accounting", "fit_percentage": 80, "one_liner": "Shares core business courses, more structured career path", "salary_range": "$55k-$75k"},
        {"name": "Business Analytics", "fit_percentage": 78, "one_liner": "Keeps the quant edge, adds data/coding skills", "salary_range": "$65k-$90k"},
        {"name": "Marketing", "fit_percentage": 68, "one_liner": "Same college, much more creative/people-facing", "salary_range": "$50k-$70k"},
        {"name": "Actuarial Science", "fit_percentage": 65, "one_liner": "For the most quantitatively inclined Finance students", "salary_range": "$65k-$85k"},
        {"name": "Real Estate", "fit_percentage": 62, "one_liner": "A specialized, less crowded finance-adjacent path", "salary_range": "$55k-$80k"},
        {"name": "International Business", "fit_percentage": 58, "one_liner": "For students drawn to global markets specifically", "salary_range": "$55k-$75k"},
    ],
    "economics": [
        {"name": "Finance", "fit_percentage": 87, "one_liner": "The single most common switch from Economics", "salary_range": "$60k-$80k"},
        {"name": "Data Science", "fit_percentage": 80, "one_liner": "For students who liked the stats/modeling side most", "salary_range": "$75k-$105k"},
        {"name": "Business Analytics", "fit_percentage": 76, "one_liner": "Applies econ's quant training to real business data", "salary_range": "$65k-$90k"},
        {"name": "Political Science", "fit_percentage": 65, "one_liner": "For students drawn to the policy side of Econ", "salary_range": "$45k-$65k"},
        {"name": "Mathematics", "fit_percentage": 63, "one_liner": "For the most theory-driven Econ students", "salary_range": "$60k-$85k"},
        {"name": "Statistics", "fit_percentage": 62, "one_liner": "Deepens the quantitative side specifically", "salary_range": "$65k-$90k"},
        {"name": "Public Policy", "fit_percentage": 60, "one_liner": "A direct, applied path for policy-minded students", "salary_range": "$50k-$70k"},
        {"name": "International Relations", "fit_percentage": 55, "one_liner": "For students most interested in global economics", "salary_range": "$45k-$65k"},
    ],
    "computer science": [
        {"name": "Data Science", "fit_percentage": 85, "one_liner": "The most common CS switch — huge course overlap", "salary_range": "$75k-$110k"},
        {"name": "Information Systems", "fit_percentage": 78, "one_liner": "Less theory-heavy, more directly business-applied", "salary_range": "$65k-$95k"},
        {"name": "Computer Engineering", "fit_percentage": 76, "one_liner": "For students who want more hardware/systems depth", "salary_range": "$75k-$105k"},
        {"name": "Mathematics", "fit_percentage": 68, "one_liner": "For the most theory-driven CS students", "salary_range": "$60k-$85k"},
        {"name": "Business Analytics", "fit_percentage": 66, "one_liner": "Trades some depth in CS for business context", "salary_range": "$65k-$90k"},
        {"name": "Statistics", "fit_percentage": 64, "one_liner": "Deepens the data/modeling side of CS specifically", "salary_range": "$65k-$90k"},
        {"name": "Cybersecurity", "fit_percentage": 62, "one_liner": "A focused, high-demand CS specialty track", "salary_range": "$70k-$100k"},
        {"name": "Digital Media/Design", "fit_percentage": 45, "one_liner": "For CS students who realized they want the creative side", "salary_range": "$50k-$75k"},
    ],
    "psychology": [
        {"name": "Sociology", "fit_percentage": 80, "one_liner": "Heaviest course overlap of any switch from Psychology", "salary_range": "$45k-$65k"},
        {"name": "Human Resources/Management", "fit_percentage": 75, "one_liner": "Applies people-focused psych training to business", "salary_range": "$50k-$70k"},
        {"name": "Social Work", "fit_percentage": 72, "one_liner": "A direct, helping-profession application of psychology", "salary_range": "$40k-$55k"},
        {"name": "Communication Studies", "fit_percentage": 68, "one_liner": "For students most interested in interpersonal dynamics", "salary_range": "$45k-$65k"},
        {"name": "Criminal Justice", "fit_percentage": 62, "one_liner": "Applies behavioral psych to the justice system", "salary_range": "$45k-$65k"},
        {"name": "Nursing/Health Sciences", "fit_percentage": 55, "one_liner": "For students drawn to psychology's clinical side", "salary_range": "$60k-$80k"},
        {"name": "Education", "fit_percentage": 54, "one_liner": "Applies developmental psychology directly", "salary_range": "$42k-$60k"},
        {"name": "Business Administration", "fit_percentage": 50, "one_liner": "For students who want a more generalist business path", "salary_range": "$50k-$70k"},
    ],
    "nursing": [
        {"name": "Health Sciences", "fit_percentage": 82, "one_liner": "Keeps the healthcare focus, less clinical intensity", "salary_range": "$50k-$70k"},
        {"name": "Public Health", "fit_percentage": 75, "one_liner": "Population-level health work instead of bedside care", "salary_range": "$50k-$70k"},
        {"name": "Health Administration/Management", "fit_percentage": 68, "one_liner": "Healthcare operations instead of direct patient care", "salary_range": "$55k-$78k"},
        {"name": "Biology", "fit_percentage": 60, "one_liner": "A common reset into pre-health without clinical hour requirements", "salary_range": "$40k-$60k"},
        {"name": "Kinesiology/Exercise Science", "fit_percentage": 62, "one_liner": "Health-adjacent, without nursing's clinical intensity", "salary_range": "$42k-$62k"},
        {"name": "Psychology", "fit_percentage": 58, "one_liner": "For students drawn to the patient-interaction side", "salary_range": "$45k-$65k"},
        {"name": "Social Work", "fit_percentage": 55, "one_liner": "Shifts from medical care to social/behavioral support", "salary_range": "$42k-$58k"},
        {"name": "Nutrition/Dietetics", "fit_percentage": 52, "one_liner": "A specific health-adjacent specialty, less clinical", "salary_range": "$48k-$68k"},
    ],
    "marketing": [
        {"name": "Communication Studies", "fit_percentage": 78, "one_liner": "Closely related, broader communication focus", "salary_range": "$45k-$65k"},
        {"name": "Public Relations", "fit_percentage": 75, "one_liner": "Very adjacent, more media/reputation-focused", "salary_range": "$45k-$65k"},
        {"name": "Business Analytics", "fit_percentage": 68, "one_liner": "Adds quantitative rigor to marketing instincts", "salary_range": "$65k-$90k"},
        {"name": "Management", "fit_percentage": 65, "one_liner": "A broader business generalist path", "salary_range": "$50k-$72k"},
        {"name": "Psychology", "fit_percentage": 58, "one_liner": "For students interested in consumer behavior specifically", "salary_range": "$45k-$65k"},
        {"name": "Graphic Design", "fit_percentage": 55, "one_liner": "For students drawn to the creative/visual side", "salary_range": "$42k-$62k"},
        {"name": "Journalism", "fit_percentage": 52, "one_liner": "For students who like the content/storytelling side", "salary_range": "$38k-$55k"},
        {"name": "International Business", "fit_percentage": 50, "one_liner": "For students interested in global brand work", "salary_range": "$55k-$75k"},
    ],
    "biology": [
        {"name": "Public Health", "fit_percentage": 78, "one_liner": "Keeps the science focus with a broader population lens", "salary_range": "$48k-$68k"},
        {"name": "Environmental Science", "fit_percentage": 75, "one_liner": "A closely related natural science field", "salary_range": "$50k-$70k"},
        {"name": "Health Sciences", "fit_percentage": 72, "one_liner": "A common landing spot for pre-health generalists", "salary_range": "$48k-$68k"},
        {"name": "Biochemistry", "fit_percentage": 70, "one_liner": "Deepens the molecular/chemistry side of biology", "salary_range": "$50k-$72k"},
        {"name": "Nursing", "fit_percentage": 65, "one_liner": "A common pivot toward direct clinical care", "salary_range": "$65k-$85k"},
        {"name": "Nutrition/Dietetics", "fit_percentage": 58, "one_liner": "Applies biology to a specific health specialty", "salary_range": "$48k-$68k"},
        {"name": "Kinesiology/Exercise Science", "fit_percentage": 56, "one_liner": "Applies biology to human movement and health", "salary_range": "$42k-$62k"},
        {"name": "Psychology", "fit_percentage": 55, "one_liner": "For students drawn to the behavioral side of biology", "salary_range": "$45k-$65k"},
    ],
    "engineering": [
        {"name": "Industrial Engineering", "fit_percentage": 72, "one_liner": "A closely related, more process/systems-focused track", "salary_range": "$65k-$85k"},
        {"name": "Computer Science", "fit_percentage": 68, "one_liner": "For students who found they preferred the software side", "salary_range": "$75k-$110k"},
        {"name": "Construction Management", "fit_percentage": 65, "one_liner": "Applies engineering knowledge to project/site management", "salary_range": "$58k-$80k"},
        {"name": "Business Administration", "fit_percentage": 62, "one_liner": "For students pivoting toward management generalist paths", "salary_range": "$50k-$72k"},
        {"name": "Physics", "fit_percentage": 60, "one_liner": "For the most theory-driven engineering students", "salary_range": "$55k-$78k"},
        {"name": "Data Science", "fit_percentage": 58, "one_liner": "Applies engineering's quant rigor to data work", "salary_range": "$75k-$105k"},
        {"name": "Project Management", "fit_percentage": 55, "one_liner": "A generalist pivot leveraging technical background", "salary_range": "$60k-$85k"},
        {"name": "Mathematics", "fit_percentage": 52, "one_liner": "For the most theoretical, proof-driven students", "salary_range": "$55k-$78k"},
    ],
    "communication": [
        {"name": "Marketing", "fit_percentage": 82, "one_liner": "The single most common switch — huge skill overlap", "salary_range": "$45k-$65k"},
        {"name": "Public Relations", "fit_percentage": 80, "one_liner": "Very closely related, often the same department", "salary_range": "$45k-$65k"},
        {"name": "Journalism", "fit_percentage": 70, "one_liner": "For students drawn to the reporting/content side", "salary_range": "$38k-$55k"},
        {"name": "Human Resources/Management", "fit_percentage": 60, "one_liner": "Applies communication skills to organizational contexts", "salary_range": "$45k-$65k"},
        {"name": "Business Administration", "fit_percentage": 58, "one_liner": "A generalist business pivot", "salary_range": "$50k-$70k"},
        {"name": "Psychology", "fit_percentage": 58, "one_liner": "For students interested in interpersonal/persuasion theory", "salary_range": "$45k-$65k"},
        {"name": "Political Science", "fit_percentage": 55, "one_liner": "For students interested in public communication/policy", "salary_range": "$42k-$62k"},
        {"name": "Film/Media Studies", "fit_percentage": 52, "one_liner": "For students drawn to the production/media side", "salary_range": "$40k-$60k"},
    ],
}


CAREERS_SCHEMA = """Respond ONLY with valid JSON, no markdown:
{
  "careers": [
    {
      "title": "Job title",
      "why_it_fits": "Max 12 words connecting this to their specific interests/values/major",
      "salary_range": "$X-$Y, entry-level",
      "day_in_the_life": "Max 15 words, concrete and specific, not generic",
      "how_to_get_there": "Max 12 words — the realistic first step from where they are now",
      "growth_outlook": "2-4 words on demand/growth, e.g. 'Growing fast' or 'Stable, competitive'"
    }
  ]
}
Generate exactly 18 careers. They should span a real range — some closely tied to their
current/likely major, some more exploratory based on interests they mentioned, some that
connect two interests together in a way they may not have considered. Vary salary ranges
honestly — not everything should be high-paying. No duplicates. No generic filler titles.
Every field is a FRAGMENT, not a full sentence — short, scannable, no filler words like
"this role" or "in this field". CRITICAL: never use a double-quote character (") inside any
string value — it breaks JSON parsing. Use single quotes ('like this') if you need to quote
a phrase, or better, just rephrase to avoid quoting at all."""


def _validate_careers(parsed: dict) -> tuple[bool, str]:
    n = len(parsed.get("careers", []))
    if n < 12:  # allow some slack below the requested 18, but not a near-empty response
        return False, f"Got only {n} careers, expected around 18 (incomplete response)"
    return True, ""


async def generate_careers(answers: dict, unl: bool) -> dict:
    # Instant, zero-AI-call path for common majors — a real speed win, with
    # an honest tradeoff: this list isn't personalized to THIS student's
    # specific interests/values the way the AI-generated version is, since
    # it's the same curated list for every student in that major. Still 18
    # real, well-reasoned careers either way.
    databank_key = normalize_major(answers.get("major", ""))
    if databank_key in CAREER_DATABANK:
        return {"careers": CAREER_DATABANK[databank_key], "_source": "databank"}

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

    result = await call_ai_with_retry(prompt, _validate_careers, max_tokens=6000)
    result["_source"] = "ai"
    return result


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
      "one_liner": "Max 12 words on why this fits their specific interests/values",
      "salary_range": "$X-$Y, entry-level"
    }
  ]
}
Generate exactly 10 majors, ranked by fit_percentage descending. Include their current major
if it genuinely belongs in a top-10 fit list — don't force it in artificially if it doesn't.
Vary fit_percentage honestly (not all 80+). No duplicates. No generic filler names.
one_liner is a FRAGMENT, not a full sentence — short and scannable.
CRITICAL: never use a double-quote character (") inside any string value — use single quotes
('like this') instead, or rephrase to avoid quoting entirely."""

MAJOR_SEARCH_SCHEMA = """Respond ONLY with valid JSON, no markdown:
{
  "name": "the exact major name searched for",
  "fit_percentage": 68,
  "one_liner": "Max 15 words on why this does or doesn't fit their specific interests/values",
  "salary_range": "$X-$Y, entry-level"
}
Be honest — if this major is a poor fit for their stated interests/values, say so plainly and
give an honest, lower fit_percentage rather than padding it. one_liner is short and scannable,
not a full paragraph. CRITICAL: never use a double-quote character (") inside any string value."""


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
    # Instant, zero-AI-call path — the most genuinely common switch
    # destinations for this major, not a personalized guess. This is what
    # actually removes the wait, since the AI call was the bottleneck when
    # clicking through several of the 10 in a row.
    databank_key = normalize_major(answers.get("major", ""))
    if databank_key in COMMON_SWITCHES:
        return {"majors": COMMON_SWITCHES[databank_key], "_source": "databank"}

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

    result = await call_ai_with_retry(prompt, _validate_majors_list, max_tokens=3000)
    result["_source"] = "ai"
    return result


async def generate_major_search(answers: dict, search_major: str, unl: bool) -> dict:
    # Instant path — if this exact switch is already in the common-switches
    # databank for their current major, use that real data instead of a new
    # AI call. Falls back to a live AI evaluation for anything not covered.
    databank_key = normalize_major(answers.get("major", ""))
    if databank_key in COMMON_SWITCHES:
        match = next((m for m in COMMON_SWITCHES[databank_key]
                      if m["name"].lower() == search_major.strip().lower()), None)
        if match:
            return {**match, "_source": "databank"}

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

    result = await call_ai_with_retry(prompt, _validate_major_search, max_tokens=800)
    result["_source"] = "ai"
    return result


# ----------------------------------------------------------------------------
# Sample schedule — a concrete, illustrative "here's what your next semesters
# could look like" for a specific major, built from the credits/time already
# computed for that path. Deliberately does NOT include professors, rooms, or
# specific sections — those change every term and aren't something this app
# can reliably know. Just course identifiers (e.g. "FINA 363"), clearly
# framed as a sample sequence, not a guaranteed real schedule.
# ----------------------------------------------------------------------------
SCHEDULE_SCHEMA = """Respond ONLY with valid JSON, no markdown:
{
  "terms": [
    {
      "term_label": "Semester 1",
      "courses": ["FINA 363", "ECON 212", "MGMT 200", "General Elective"]
    }
  ]
}
Each term should have 4-5 courses (a realistic full-time course load). Use REAL course
identifiers where you know them for this school/major (e.g. "FINA 363" not "Finance Course 1"),
falling back to a realistic-sounding placeholder only when genuinely uncertain of the exact
code. Sequence courses sensibly — prerequisites and intro courses before advanced ones.
This is a SAMPLE/illustrative plan, not a guaranteed real schedule (real course offerings and
availability vary by term) — do not include professor names, room numbers, or specific
sections; just course identifiers. CRITICAL: never use a double-quote character (") inside
any string value."""


def _validate_schedule(parsed: dict) -> tuple[bool, str]:
    terms = parsed.get("terms", [])
    if not terms or not isinstance(terms, list):
        return False, "Missing or empty terms in schedule"
    if any(not t.get("courses") for t in terms):
        return False, "A term has no courses listed"
    return True, ""


async def generate_schedule(answers: dict, target_major: str, additional_credits_needed,
                             catalog: dict, unl: bool) -> dict:
    # Roughly how many terms this needs: ~15 credits/term is a typical
    # full-time pace. Always at least 1 term so there's something to show.
    try:
        credits = float(additional_credits_needed) if additional_credits_needed is not None else 30
    except (TypeError, ValueError):
        credits = 30
    n_terms = max(1, min(8, round(credits / 15) or 1))

    catalog_block = (
        f"\nVERIFIED CATALOG DATA — ground real course codes in this:\n{catalog['content']}\n"
        if catalog.get("content") else ""
    )
    unl_block = (
        "This is a University of Nebraska-Lincoln student — use real UNL course codes "
        "where you know them (e.g. FINA 361, ECON 212, CSCE 156)." if unl else ""
    )

    prompt = f"""You are MajorMove's schedule simulator. Build a SAMPLE, illustrative
term-by-term course sequence for a student switching to {target_major}.

Student:
- School: {answers.get('school')}
- Current year: {answers.get('year')}
- Target major: {target_major}
- Approximate additional credits needed: {credits}
- Build approximately {n_terms} term(s) of courses to cover that
{catalog_block}{unl_block}

{SCHEDULE_SCHEMA}"""

    return await call_ai_with_retry(prompt, _validate_schedule, max_tokens=1500)


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


@app.post("/schedule")
async def schedule(
    school: str = Form(...),
    year: str = Form(...),
    target_major: str = Form(...),
    additional_credits_needed: str = Form("30"),
    anon_id: str = Form(None),
    user: Optional[dict] = Depends(current_user),
):
    """A sample, illustrative term-by-term course sequence for a specific
    major — the concrete 'here's what it could actually look like' view.
    Lightweight: no transcript images, reuses catalog data already cached
    for this school/major if available."""
    answers = {"school": school, "year": year}
    catalog = await fetch_catalog(school, target_major)
    log_event("schedule_started", anon_id=anon_id,
              user_id=user["id"] if user else None, school=school,
              props={"target_major": target_major})

    result = await generate_schedule(answers, target_major, additional_credits_needed,
                                      catalog, is_unl(school))
    log_event("schedule_completed", anon_id=anon_id,
              user_id=user["id"] if user else None, school=school,
              props={"target_major": target_major, "term_count": len(result.get("terms", []))})
    return result


# ----------------------------------------------------------------------------
# Sheldon — the in-app academic advisor chat. Deliberately scoped, not a
# general-purpose assistant: grounded in THIS student's own already-generated
# analysis (their majors, paths, interests), and explicitly redirects
# off-topic questions rather than trying to answer anything at all. Reuses
# the same proven call_ai_with_retry helper — conversation history is
# formatted into one text block rather than a true multi-turn message list,
# since that's simpler and doesn't require touching well-tested infrastructure.
# ----------------------------------------------------------------------------
ADVISOR_SCHEMA = """Respond ONLY with valid JSON, no markdown:
{"reply": "your response to the student"}
CRITICAL: never use a double-quote character (") inside reply — use single quotes instead."""


def _validate_advisor(parsed: dict) -> tuple[bool, str]:
    if not parsed.get("reply"):
        return False, "Missing reply"
    return True, ""


async def generate_advisor_reply(context: dict, history: list, question: str) -> dict:
    history_text = "\n".join(
        f"{'Student' if h['role']=='user' else 'Sheldon'}: {h['content']}" for h in history[-10:]
    )
    prompt = f"""You are Sheldon, MajorMove's in-app academic advisor. You are helping ONE specific
student understand the analysis MajorMove already generated for them — you are not a general-purpose
assistant.

What you know about this student:
- Current major: {context.get('current_major')}
- Alternative majors MajorMove suggested: {', '.join(context.get('path_names', []))}
- Their stated interests: {', '.join(context.get('interests', []))}
- Their stated career values: {', '.join(context.get('values', []))}
- Their school: {context.get('school')}

Rules:
- Stay strictly scoped to helping with THIS student's majors, courses, careers, and this analysis.
  If asked something unrelated (general trivia, other topics, anything outside academic/career
  advising), politely say that's outside what you can help with here and redirect to what you can.
- Be honest, not falsely reassuring — this is MajorMove's whole ethos. Don't invent specific facts
  (exact scholarship dollar amounts, guaranteed course availability) you don't actually know.
- Keep replies conversational and SHORT — a few sentences, not an essay. This is a chat, not a report.

Conversation so far:
{history_text}

Student's new message: {question}

{ADVISOR_SCHEMA}"""

    return await call_ai_with_retry(prompt, _validate_advisor, max_tokens=500)


@app.post("/advisor/chat")
async def advisor_chat(
    school: str = Form(...),
    current_major: str = Form(...),
    path_names: str = Form(""),
    interests: str = Form(""),
    values: str = Form(""),
    history_json: str = Form("[]"),
    question: str = Form(...),
    anon_id: str = Form(None),
    user: Optional[dict] = Depends(current_user),
):
    """Sheldon's chat endpoint — grounded in the student's own analysis,
    not a generic assistant. history_json is the prior conversation turns
    as [{"role": "user"|"assistant", "content": "..."}]."""
    if not question.strip():
        raise HTTPException(400, "question is required")
    try:
        history = json.loads(history_json) if history_json else []
    except json.JSONDecodeError:
        history = []

    context = {
        "school": school, "current_major": current_major,
        "path_names": [p for p in path_names.split(",") if p],
        "interests": [i for i in interests.split(",") if i],
        "values": [v for v in values.split(",") if v],
    }
    log_event("advisor_chat", anon_id=anon_id,
              user_id=user["id"] if user else None, school=school)

    return await generate_advisor_reply(context, history, question.strip())


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
    """Founder dashboard — the numbers that actually matter, not just totals.
    Retention as a RATE (not a raw count) is the single most buyer-relevant
    number here; totals alone can't distinguish 12 returners out of 15 users
    from 12 out of 1,000."""
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
        returning = conn.execute("""
            SELECT COUNT(*) c FROM (
                SELECT email FROM roadmaps WHERE email IS NOT NULL
                GROUP BY email HAVING COUNT(DISTINCT substr(created_at,1,10)) >= 2
            )
        """).fetchone()["c"]

        # Real transcript engagement — pulled from each roadmap's stored
        # diagnostics, not a guess. This is the "keep the transcripts"
        # number: how many students actually uploaded a real transcript
        # versus skipping it, which is a genuine depth-of-engagement signal
        # distinct from a raw signup count.
        all_payloads = conn.execute("SELECT payload, created_at FROM roadmaps").fetchall()
        real_transcript_count = 0
        for row in all_payloads:
            try:
                p = json.loads(row["payload"])
                if p.get("_transcript_file_uploaded") or p.get("_transcript_text_received") \
                   or (p.get("_transcript_pages_received") or 0) > 0:
                    real_transcript_count += 1
            except (json.JSONDecodeError, TypeError):
                continue

        # Growth trend — this week vs the week before, using created_at
        # parsed in Python rather than SQL date functions, since those
        # differ between SQLite and Postgres and this keeps it portable.
        now = datetime.utcnow()
        this_week_cutoff = now - timedelta(days=7)
        last_week_cutoff = now - timedelta(days=14)
        roadmaps_this_week = 0
        roadmaps_last_week = 0
        for row in all_payloads:
            try:
                created = datetime.fromisoformat(row["created_at"])
            except (ValueError, TypeError):
                continue
            if created >= this_week_cutoff:
                roadmaps_this_week += 1
            elif created >= last_week_cutoff:
                roadmaps_last_week += 1

    def pct(numerator, denominator):
        return round((numerator / denominator) * 100, 1) if denominator else 0.0

    return {
        "total_users": total_users,
        "total_roadmaps": total_roadmaps,
        "total_emails_collected": total_emails,
        "returning_users": returning,
        "retention_rate_pct": pct(returning, total_emails),
        "real_transcript_count": real_transcript_count,
        "transcript_upload_rate_pct": pct(real_transcript_count, total_roadmaps),
        "distinct_schools_count": len(schools),
        "roadmaps_this_week": roadmaps_this_week,
        "roadmaps_last_week": roadmaps_last_week,
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
