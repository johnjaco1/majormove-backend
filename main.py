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

# Optional PDF extraction
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
DB_PATH = os.environ.get("DB_PATH", "majormove.db")
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
# Database (SQLite for MVP; swap DB_PATH/queries for Postgres in production)
# ----------------------------------------------------------------------------
@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as conn:
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
def extract_pdf_text(data: bytes) -> str:
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
      "extra_time": "0 semesters / 1 semester / 2 semesters — with a short reason",
      "honest_take": "one honest, specific sentence about real outcomes",
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

ABSOLUTE RULE — this breaks the entire response if violated, follow it with zero exceptions:
The double-quote character (") may ONLY appear as JSON structure (wrapping keys and string values).
It must NEVER appear inside the text of any string value, for ANY reason — not for emphasis, not to
quote a phrase, not for scare quotes, not for anything.
WRONG (breaks parsing): "honest_take": "This is a "great fit" if you like data."
WRONG (breaks parsing): "why_fit": "Explore this "seriously" before committing."
RIGHT: "honest_take": "This is a great fit if you like data."
RIGHT: "why_fit": "Seriously consider exploring this before committing."
Simply do not use quotation marks of any kind inside your sentences. Rephrase instead of quoting."""


async def generate_analysis(answers: dict, catalog: dict, transcript_text: str,
                             transcript_image_b64: Optional[str], image_media_type: Optional[str]) -> dict:
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

    prompt = f"""You are MajorMove, an AI academic advisor whose purpose is to help a college student
make a clearer, better decision about their major — and, in aggregate, to help their university
retain and graduate more students. Be warm, honest, specific, never generic.

Student:
- School: {answers.get('school')}
- Year: {answers.get('year')}
- Current major: {answers.get('major')}
- Interests: {', '.join(answers.get('interests', []))}
- Career values: {', '.join(answers.get('values', []))}
- Financial: {answers.get('financial')}
{catalog_block}{transcript_block}
{unl_block}

{ANALYSIS_SCHEMA}"""

    if transcript_image_b64:
        content = [
            {"type": "image", "source": {"type": "base64",
             "media_type": image_media_type or "image/jpeg", "data": transcript_image_b64}},
            {"type": "text", "text": prompt + "\n\nA transcript image is attached — read completed courses/credits from it."},
        ]
    else:
        content = prompt

    last_error = None
    last_debug = None
    for attempt_num in range(2):  # try once, then retry once more on any failure
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY,
                         "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": ANTHROPIC_MODEL, "max_tokens": 8000,
                      "messages": [{"role": "user", "content": content}]},
            )
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

        # Parsed successfully, but the response is only valid if it actually
        # has the full required shape — a truncated/early-stopped response
        # can sometimes still parse as valid JSON if it happens to be cut
        # off at a spot that closes cleanly, while still being incomplete.
        if len(parsed.get("paths", [])) != 4:
            last_error = f"Got {len(parsed.get('paths', []))} paths, expected 4 (incomplete response)"
            last_debug = f"stop_reason: {data.get('stop_reason')}, usage: {data.get('usage')}"
            continue  # retry — parsed fine but incomplete

        return parsed  # success

    raise HTTPException(502, f"Could not get a complete analysis after 2 attempts. "
                              f"Last error: {last_error}. Debug: {last_debug}")

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

# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "pypdf": HAS_PYPDF, "serper": bool(SERPER_API_KEY), "ai": bool(ANTHROPIC_API_KEY)}


@app.post("/auth/signup")
async def signup(req: SignupReq):
    salt = secrets.token_hex(16)
    pw = hash_password(req.password, salt)
    try:
        with db() as conn:
            cur = conn.execute(
                "INSERT INTO users (email, password_hash, salt, school, created_at) VALUES (?,?,?,?,?)",
                (req.email.lower(), pw, salt, req.school, datetime.utcnow().isoformat()),
            )
            uid = cur.lastrowid
    except sqlite3.IntegrityError:
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
    transcript_file: Union[UploadFile, str, None] = File(None),
    user: Optional[dict] = Depends(current_user),
):
    answers = {
        "school": school, "year": year, "major": major,
        "interests": [i for i in interests.split(",") if i],
        "values": [v for v in values.split(",") if v],
        "financial": financial,
    }

    # Transcript: PDF → text; image → pass to model
    # (Swagger's "Try it out" UI sometimes sends an empty string instead of
    # omitting the file entirely — treat anything that isn't a real UploadFile as "no file")
    t_text = transcript_text or ""
    t_image_b64, t_media = None, None
    if isinstance(transcript_file, UploadFile):
        raw = await transcript_file.read()
        ctype = transcript_file.content_type or ""
        if ctype == "application/pdf" or transcript_file.filename.lower().endswith(".pdf"):
            extracted = extract_pdf_text(raw)
            t_text = (t_text + "\n" + extracted).strip()
        elif ctype.startswith("image/"):
            import base64
            t_image_b64 = base64.b64encode(raw).decode()
            t_media = ctype

    catalog = await fetch_catalog(school, major)
    resolved_email = email or (user["email"] if user else None)
    log_event("analysis_started", anon_id=anon_id,
              user_id=user["id"] if user else None, school=school,
              props={"major": major, "has_transcript": bool(t_text or t_image_b64),
                     "transcript_kind": "text" if t_text else ("image" if t_image_b64 else "none"),
                     "email_provided": bool(resolved_email)})

    try:
        result = await generate_analysis(answers, catalog, t_text, t_image_b64, t_media)
    except json.JSONDecodeError:
        raise HTTPException(502, "Could not parse AI response")

    result["_catalog_source"] = catalog.get("source_url")
    result["_catalog_verified"] = catalog.get("verified", False)

    with db() as conn:
        conn.execute(
            "INSERT INTO roadmaps (user_id, email, school, year, major, payload, created_at) VALUES (?,?,?,?,?,?,?)",
            (user["id"] if user else None, resolved_email, school, year, major,
             json.dumps(result), datetime.utcnow().isoformat()),
        )
    log_event("analysis_completed", anon_id=anon_id,
              user_id=user["id"] if user else None, school=school)
    return result


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
