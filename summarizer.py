"""
====================================================
  MODULE 3: LLM Summarization Engine — Google Gemini
  Smart Email Summarizer Project
====================================================
  Handles:
   - Connects to Google Gemini API (gemini-2.5-flash)
   - Structured JSON output with response_mime_type
   - Extracts: summary, priority, action items, deadlines
   - Retry logic with exponential backoff
   - Batch processes all threads from Module 2
   - Saves structured results for Module 4 (Dashboard)
====================================================
  Setup:
    pip install google-genai python-dotenv

  Get free API key → https://aistudio.google.com/
  Then either:
    export GEMINI_API_KEY=AIza...
  Or create .env file:
    GEMINI_API_KEY=AIza...
====================================================
  Free tier limits (as of 2026):
    gemini-2.5-flash → 15 req/min, 1M tokens/day
    gemini-2.5-flash-lite → 30 req/min (faster, lighter)
====================================================
"""

import os
import json
import time
import re
from datetime import datetime
from dotenv import load_dotenv

from google import genai
from google.genai import types
from storage import StorageManager
from processing import extract_thread_id 



# ─────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────

load_dotenv()   # reads GEMINI_API_KEY from .env file

# Model options:
#   "gemini-2.5-flash"      → best quality, free tier ✅ (recommended)
#   "gemini-2.5-flash-lite" → faster + higher rate limits, free tier ✅
MODEL          = "gemini-2.5-flash"
MAX_TOKENS     = 2048   # 1024 was too low — complex thread JSON can be truncated mid-object
TEMPERATURE    = 0.2    # low = more consistent/structured output
MAX_BODY_CHARS = 4000   # truncate very long emails before sending
MAX_RETRIES    = 3
RETRY_DELAY    = 2      # seconds (doubles on each retry)



# ── at the top of summarizer.py, after CONFIGURATION section ──

SKIP_KEYWORDS = ["password", "otp", "bank account", "credit card", "aadhar"]

def is_sensitive(email: dict) -> bool:
    """Returns True if email body contains sensitive keywords."""
    body = (email.get("clean_text") or email.get("body", "")).lower()
    return any(word in body for word in SKIP_KEYWORDS)
#  STEP 1: INITIALIZE GEMINI CLIENT



def get_client():
    """
    Initializes and returns the Google Gemini API client.
    Reads GEMINI_API_KEY from environment or .env file.

    Returns:
        genai.Client: Gemini API client

    Raises:
        EnvironmentError: If API key is not set
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "\n GEMINI_API_KEY not found!\n"
            "   1. Get a free key at: https://aistudio.google.com/\n"
            "   2. Set it in terminal:  export GEMINI_API_KEY=AIza...\n"
            "      Or in .env file:     GEMINI_API_KEY=AIza...\n"
        )

    client = genai.Client(api_key=api_key)
    return client


#  STEP 2: PROMPT TEMPLATES


SYSTEM_PROMPT = """You are an expert email assistant that analyzes emails and threads.
Extract key information and return it as clean, structured JSON only.
Never include markdown, backticks, or explanation text — pure JSON only."""


def build_single_email_prompt(email: dict) -> str:
    """
    Builds the summarization prompt for a single email.

    Args:
        email (dict): Cleaned email dict from Module 2

    Returns:
        str: Full prompt string for Gemini
    """
    subject   = email.get("subject",   "(No Subject)")
    sender    = email.get("sender",    "Unknown")
    timestamp = email.get("timestamp", "")
    body      = email.get("clean_text") or email.get("body", "")

    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + f"\n\n[... truncated at {MAX_BODY_CHARS} chars ...]"

    return f"""Analyze this email and return ONLY a JSON object with this exact schema:

{{
  "summary": "2-3 sentence summary of what this email is about",
  "key_points": ["point 1", "point 2", "point 3"],
  "action_items": [
    {{
      "task": "what needs to be done",
      "owner": "who should do it (use 'recipient' if unclear)",
      "deadline": "deadline date if mentioned, else null",
      "priority_hint": "urgent/high/medium/low"
    }}
  ],
  "deadlines": [
    {{
      "description": "what the deadline is for",
      "date": "the date or time mentioned",
      "is_explicit": true
    }}
  ],
  "priority": "URGENT | HIGH | MEDIUM | LOW",
  "priority_reason": "one sentence explaining the priority",
  "category": "work | personal | finance | newsletter | notification | support | other",
  "sentiment": "positive | neutral | negative | urgent",
  "requires_reply": true,
  "estimated_read_time_seconds": 30
}}

Priority rules:
- URGENT → needs attention within hours (deadline today/tomorrow, emergency)
- HIGH   → needs attention within 1-2 days (important tasks, soon deadlines)
- MEDIUM → needs attention this week (non-urgent tasks, FYI emails)
- LOW    → no immediate action (newsletters, receipts, automated notifications)

Use empty arrays [] for action_items or deadlines if none exist.
Set requires_reply to false for newsletters, receipts, automated notifications.

EMAIL:
Subject : {subject}
From    : {sender}
Date    : {timestamp}

Body:
{body}"""


def build_thread_prompt(thread: dict) -> str:
    """
    Builds the summarization prompt for an email thread (multiple messages).

    Args:
        thread (dict): Thread dict from Module 2 with combined_text

    Returns:
        str: Full prompt string for Gemini
    """
    subject      = thread.get("subject",      "(No Subject)")
    participants = thread.get("participants",  [])
    email_count  = thread.get("email_count",  1)
    date_range   = thread.get("date_range",   {})
    combined     = thread.get("combined_text", "")

    if len(combined) > MAX_BODY_CHARS:
        combined = combined[:MAX_BODY_CHARS] + f"\n\n[... truncated at {MAX_BODY_CHARS} chars ...]"

    participants_str = ", ".join(participants) if participants else "Unknown"
    date_str         = f"{date_range.get('earliest','')} → {date_range.get('latest','')}"

    return f"""Analyze this email thread ({email_count} messages) and return ONLY a JSON object:

{{
  "summary": "2-3 sentence summary of the full conversation and current status",
  "thread_conclusion": "what was decided or the current state (one sentence)",
  "key_points": ["point 1", "point 2", "point 3"],
  "action_items": [
    {{
      "task": "what needs to be done",
      "owner": "name or email of who should do it",
      "deadline": "deadline date if mentioned, else null",
      "priority_hint": "urgent/high/medium/low"
    }}
  ],
  "deadlines": [
    {{
      "description": "what the deadline is for",
      "date": "the date or time mentioned",
      "is_explicit": true
    }}
  ],
  "open_questions": ["any unresolved questions in the thread"],
  "priority": "URGENT | HIGH | MEDIUM | LOW",
  "priority_reason": "one sentence explaining the priority",
  "category": "work | personal | finance | newsletter | notification | support | other",
  "sentiment": "positive | neutral | negative | urgent",
  "requires_reply": true,
  "participants_roles": {{
    "initiator": "who started the thread",
    "decision_maker": "who seems to be making decisions (or null)"
  }}
}}

Summarize the ENTIRE conversation, not just the latest message.
Use empty arrays for action_items, deadlines, open_questions if none exist.

EMAIL THREAD:
Subject      : {subject}
Participants : {participants_str}
Date range   : {date_str}
Messages     : {email_count}

Thread Content:
{combined}"""


# ─────────────────────────────────────────────────
#  STEP 3: CALL GEMINI API (with retry + backoff)
# ─────────────────────────────────────────────────

def call_gemini(client: genai.Client, prompt: str, retries: int = MAX_RETRIES) -> str:
    """
    Calls the Gemini API with retry logic and exponential backoff.

    Uses response_mime_type='application/json' to force structured JSON output
    directly from Gemini — no post-processing needed for format.

    Args:
        client:   Gemini API client
        prompt:   The user prompt string
        retries:  Number of retry attempts on failure

    Returns:
        str: Raw JSON string from Gemini

    Raises:
        Exception: After all retries exhausted
    """
    config = types.GenerateContentConfig(
        system_instruction = SYSTEM_PROMPT,
        max_output_tokens  = MAX_TOKENS,
        temperature        = TEMPERATURE,
        response_mime_type = "application/json",   # ← forces pure JSON output
    )

    delay = RETRY_DELAY

    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model    = MODEL,
                contents = prompt,
                config   = config,
            )
            return response.text.strip()

        except Exception as e:
            error_str = str(e).lower()

            # Safety block — don't retry, return a safe fallback immediately
            if "safety" in error_str or "blocked" in error_str:
                print(f" Content blocked by safety filter. Skipping.")
                return json.dumps({"summary": "Content blocked by safety filter.",
                                   "key_points": [], "action_items": [], "deadlines": [],
                                   "priority": "LOW", "priority_reason": "Safety filter triggered",
                                   "category": "other", "sentiment": "neutral",
                                   "requires_reply": False})

            # Rate limit errors — retry with exponential backoff
            if "429" in error_str or "quota" in error_str or "rate" in error_str:
                if attempt < retries:
                    wait = delay * (2 ** (attempt - 1))   # exponential: 2s, 4s, 8s
                    print(f" Rate limit hit. Waiting {wait}s (retry {attempt}/{retries})...")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Gemini API rate limit — failed after {retries} attempts: {e}") from e

            # Server errors (5xx) — retry with linear backoff
            if any(code in error_str for code in ["500", "502", "503", "overloaded"]):
                if attempt < retries:
                    wait = delay * attempt
                    print(f" Server error. Retry {attempt}/{retries} in {wait}s...")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Gemini API server error — failed after {retries} attempts: {e}") from e

            # All other errors — raise immediately on final attempt, else wait and retry
            if attempt == retries:
                raise RuntimeError(f"Gemini API failed after {retries} attempts: {e}") from e

            time.sleep(delay)

#  STEP 4: PARSE JSON RESPONSE

def parse_json_response(raw_text: str) -> dict:
    """
    Safely parses Gemini's JSON response.

    Even with response_mime_type='application/json', we occasionally see malformed
    JSON (usually an unterminated string). This function tries multiple
    increasingly-robust strategies to recover.


    Args:
        raw_text (str): Raw response from Gemini

    Returns:
        dict: Parsed result

    Raises:
        ValueError: If JSON cannot be parsed
    """
    text = raw_text.strip()

    # Strip markdown fences in case they appear despite mime type
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$",          "", text)

    # Remove trailing commas before } or ] (common LLM mistake)
    text = re.sub(r",\s*([}\]])", r"\1", text)

    try:
        return json.loads(text)

    except json.JSONDecodeError as e:
        # Try finding a JSON block inside the text
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"Could not parse Gemini response as JSON.\n"
            f"Error: {e}\n"
            f"Raw (first 400 chars):\n{raw_text[:400]}"
        )

#  STEP 5: SUMMARIZE SINGLE EMAIL

def summarize_email(client: genai.Client, email: dict) -> dict:
    """
    Summarizes a single email using Gemini.

    Args:
        client:  Gemini API client
        email:   Cleaned email dict from Module 2

    Returns:
        dict: Email dict with 'summary_result' attached
    """
    prompt = build_single_email_prompt(email)
    raw    = call_gemini(client, prompt)
    result = parse_json_response(raw)

    email["summary_result"] = result
    email["summarized_at"]  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    email["model_used"]     = MODEL

    return email


#  STEP 6: SUMMARIZE EMAIL THREAD


def summarize_thread(client: genai.Client, thread: dict) -> dict:
    """
    Summarizes a multi-message email thread using Gemini.

    Args:
        client:   Gemini API client
        thread:   Thread dict from Module 2 with combined_text

    Returns:
        dict: Thread dict with 'summary_result' attached
    """
    prompt = build_thread_prompt(thread)
    raw    = call_gemini(client, prompt)
    result = parse_json_response(raw)

    thread["summary_result"] = result
    thread["summarized_at"]  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    thread["model_used"]     = MODEL

    return thread

#  STEP 7: BATCH PROCESS ALL THREADS

def summarize_all(client: genai.Client, threads: list, delay_between: float = 1.0) -> list:
    """
    Batch processes all threads through the Gemini summarization pipeline.

    Note: Free tier allows 15 req/min for gemini-2.5-flash.
    The default 1s delay keeps you well within that limit.

    Args:
        client:          Gemini API client
        threads:         List of thread dicts from Module 2
        delay_between:   Seconds to wait between API calls (default 1s)

    Returns:
        list: Threads with summary_result attached to each
    """
    print(f" Summarizing {len(threads)} thread(s) with {MODEL}...\n")

    results  = []
    failed   = []
    skipped  = []                                        # ← track skipped sensitive emails
    stats    = {"URGENT": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0,
                "total_action_items": 0, "total_deadlines": 0}

    for i, thread in enumerate(threads, 1):
        subject = thread.get("subject", "(No Subject)")
        count   = thread.get("email_count", 1)
        label   = "Thread" if count > 1 else "Email"

        print(f"  [{i}/{len(threads)}] {label} ({count} msg): {subject[:52]}")

        try:
            if count > 1:
                # ── Sensitive check for threads (any email in thread triggers skip) ──
                sensitive_emails = [
                    e for e in thread.get("emails", []) if is_sensitive(e)
                ]
                if sensitive_emails:
                    print(f"           ⚠️  Skipping — contains sensitive content "
                          f"({len(sensitive_emails)} email(s) flagged)")
                    thread["summary_result"] = None
                    thread["skipped"]        = True
                    thread["skip_reason"]    = "sensitive_content"
                    skipped.append(thread)
                    results.append(thread)
                    continue                              # ← skip Gemini call entirely

                summarized = summarize_thread(client, thread)

            else:
                emails = thread.get("emails", [])
                if emails:
                    # ── Sensitive check for single email ──
                    if is_sensitive(emails[0]):
                        print(f"           ⚠️  Skipping — sensitive content detected")
                        thread["summary_result"] = None
                        thread["skipped"]        = True
                        thread["skip_reason"]    = "sensitive_content"
                        skipped.append(thread)
                        results.append(thread)
                        continue                          # ← skip Gemini call entirely

                    summarized_email = summarize_email(client, emails[0])
                    thread["summary_result"] = summarized_email["summary_result"]
                    thread["summarized_at"]  = summarized_email["summarized_at"]
                    thread["model_used"]     = MODEL
                    summarized = thread

                else:
                    raise ValueError(
                        f"Inconsistent data: thread has email_count={count} "
                        f"but the 'emails' list is empty."
                    )

            sr       = summarized.get("summary_result", {})
            priority = sr.get("priority", "MEDIUM").upper()

            if priority in stats:
                stats[priority] += 1
            stats["total_action_items"] += len(sr.get("action_items", []))
            stats["total_deadlines"]    += len(sr.get("deadlines", []))

            results.append(summarized)

            snippet = sr.get("summary", "")[:72]
            print(f"           ✓ [{priority:6s}] {snippet}...")

        except Exception as e:
            print(f"           ✗ FAILED: {e}")
            thread["summary_result"] = None
            thread["error"]          = str(e)
            failed.append(thread)
            results.append(thread)

        # Stay within free tier rate limits
        if i < len(threads):
            time.sleep(delay_between)

    print(f"""
    Summarization Complete:
   Processed      : {len(threads)} threads
   Succeeded      : {len(threads) - len(failed) - len(skipped)}
   Skipped        : {len(skipped)} (sensitive content)
   Failed         : {len(failed)}
   ──────────────────────────────
   🔴 URGENT      : {stats['URGENT']}
   🟠 HIGH        : {stats['HIGH']}
   🟡 MEDIUM      : {stats['MEDIUM']}
   🟢 LOW         : {stats['LOW']}
   ──────────────────────────────
   📋 Action items: {stats['total_action_items']}
   📅 Deadlines   : {stats['total_deadlines']}
   ⚠️  Sensitive   : {len(skipped)}
""")

    return results

# ─────────────────────────────────────────────────
#  STEP 8: SAVE RESULTS
# ─────────────────────────────────────────────────

def save_summaries(summarized_threads: list, output_dir: str = "summaries") -> str:
    """
    Saves summarized results to a JSON file.

    Args:
        summarized_threads: List of threads with summary_result
        output_dir:         Directory to save output

    Returns:
        str: Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"summaries_{ts}.json")

    # Strip large nested data to keep file readable
    clean = []
    for t in summarized_threads:
        entry = {k: v for k, v in t.items() if k not in ("emails", "combined_text")}
        clean.append(entry)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved summaries → {filepath}")
    return filepath


# ─────────────────────────────────────────────────
#  STEP 9: PERSIST TO SQLITE (Module 4 — storage.py)
# ─────────────────────────────────────────────────

def persist_to_db(summarized_threads: list) -> None:
    """
    Persists summarized results to the SQLite database via storage.py.

    Fixes two issues:
      1. storage.py was never called from the pipeline (dead module).
      2. Gemini returns uppercase priorities ("URGENT") but the storage
         CHECK constraint requires lowercase ("urgent") — lowercased here.

    Args:
        summarized_threads: List of threads with summary_result attached
    """
    storage = StorageManager()
    saved  = 0
    failed = 0

    for thread in summarized_threads:
        sr = thread.get("summary_result")
        if not sr:
            continue

        # Bug 2 fix: lowercase priority before writing to SQLite
        priority_raw = sr.get("priority", "medium")
        priority     = priority_raw.lower()          # "URGENT" → "urgent" etc.

        action_items = [
            item.get("task", "")
            for item in sr.get("action_items", [])
            if item.get("task")
        ]

        tags = [sr.get("category", "other")] if sr.get("category") else []

        # Normalize sender key (stable grouping key)
        participants = thread.get("participants", ["Unknown"])
        sender_display = ", ".join(participants) if participants else "Unknown"

        # Use the first email-like address as the sender key; fall back to display.
        sender_key = None
        if participants:
            # participants already look like emails (from processing.py), but keep this defensive
            for p in participants:
                p = (p or "").strip().lower()
                if re.search(r"[\w.+-]+@[\w.-]+", p):
                    sender_key = p
                    break
        if not sender_key:
            sender_key = sender_display.strip().lower()

        ok = storage.save_summary(
            email_id     = thread.get("thread_id", ""),
            sender       = sender_display,
            sender_key   = sender_key,
            subject      = thread.get("subject", "(No Subject)"),
            summary      = sr.get("summary", ""),

            action_items = action_items or None,
            priority     = priority,
            tags         = tags or None,
            timestamp    = thread.get("date_range", {}).get("latest", ""),
        )

        if ok:
            saved += 1
            # Mark every individual email in this thread as processed
            for email in thread.get("emails", []):
                storage.mark_email_processed(
                    email_id   = email.get("id", ""),
                    message_id = email.get("message_id", ""),
                    sender     = email.get("sender", ""),
                    subject    = email.get("subject", ""),
                )
        else:
            failed += 1

    print(f"\n💾 SQLite: {saved} thread(s) saved, {failed} skipped (duplicates)")


# ─────────────────────────────────────────────────
#  PRINT HELPERS
# ─────────────────────────────────────────────────

PRIORITY_ICONS = {"URGENT": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
CATEGORY_ICONS = {
    "work": "💼", "personal": "👤", "finance": "💰",
    "newsletter": "📰", "notification": "🔔", "support": "🛠️", "other": "📩",
}


def print_summary(thread: dict):
    """Pretty-prints the summarized result for one thread."""
    divider = "─" * 62
    sr      = thread.get("summary_result", {})

    if not sr:
        print(f"{divider}")
        print(f"❌ {thread.get('subject','Unknown')} — failed: {thread.get('error','?')}")
        print(f"{divider}\n")
        return

    priority = sr.get("priority", "MEDIUM").upper()
    category = sr.get("category", "other").lower()
    p_icon   = PRIORITY_ICONS.get(priority, "⚪")
    c_icon   = CATEGORY_ICONS.get(category, "📩")

    print(divider)
    print(f"{p_icon} [{priority:6s}]  {c_icon}  {thread.get('subject','(No Subject)')}")
    print(f"   From   : {', '.join(thread.get('participants', ['Unknown']))}")
    print(f"   Msgs   : {thread.get('email_count',1)}   Date: {thread.get('date_range',{}).get('latest','')}")
    print(f"   Model  : {thread.get('model_used', MODEL)}")

    print(f"\n📄 Summary:\n   {sr.get('summary','')}")

    if sr.get("thread_conclusion"):
        print(f"\n🏁 Conclusion:\n   {sr['thread_conclusion']}")

    if sr.get("key_points"):
        print(f"\n📌 Key Points:")
        for pt in sr["key_points"]:
            print(f"   • {pt}")

    if sr.get("action_items"):
        print(f"\n✅ Action Items:")
        for item in sr["action_items"]:
            dl = f"  ⏰ Due: {item['deadline']}" if item.get("deadline") else ""
            print(f"   [{item.get('priority_hint','?').upper():6s}] {item['task']}")
            print(f"            → {item.get('owner','?')}{dl}")

    if sr.get("deadlines"):
        print(f"\n📅 Deadlines:")
        for d in sr["deadlines"]:
            tag = "explicit" if d.get("is_explicit") else "inferred"
            print(f"   {d['date']:20s} — {d['description']}  ({tag})")

    if sr.get("open_questions"):
        print(f"\n❓ Open Questions:")
        for q in sr["open_questions"]:
            print(f"   • {q}")

    reply = "💬 Reply needed" if sr.get("requires_reply") else "👀 No reply needed"
    print(f"\n   {reply}  |  Sentiment: {sr.get('sentiment','neutral')}")
    print(f"   Priority reason: {sr.get('priority_reason','')}")
    print(f"{divider}\n")


# ─────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  SMART EMAIL SUMMARIZER — Module 3: Gemini Summarization")
    print("=" * 62 + "\n")
        
    # ── 1. Init Gemini client ──
    client = get_client()
    print(f"✅ Gemini client ready  (model: {MODEL})\n")

    # ── 2. Load threads from Module 2 ──
    PREPROCESSED_DIR = "preprocessed"
    threads = []

    if os.path.exists(PREPROCESSED_DIR):
        thread_files = sorted([
            f for f in os.listdir(PREPROCESSED_DIR)
            if f.startswith("threads_") and f.endswith(".json")
        ], reverse=True)

        email_files = sorted([
            f for f in os.listdir(PREPROCESSED_DIR)
            if f.startswith("processed_emails_") and f.endswith(".json")
        ], reverse=True)

        # ── Load emails indexed by id for O(1) lookup ──
        emails_by_id: dict = {}
        if email_files:
            email_path = os.path.join(PREPROCESSED_DIR, email_files[0])
            print(f"Loading emails: {email_path}")
            with open(email_path, "r", encoding="utf-8") as f:
                email_list = json.load(f)
            emails_by_id = {e["id"]: e for e in email_list if e.get("id")}

        if thread_files:
           
            thread_path = os.path.join(PREPROCESSED_DIR, thread_files[0])
            print(f"Loading threads: {thread_path}\n")
            with open(thread_path, "r", encoding="utf-8") as f:
                threads_raw = json.load(f)

            for thread in threads_raw:
                thread_id = thread.get("thread_id", "")

                # Re-attach matching emails by thread_id (exact match, not subject fuzzy)
                matching_emails = [
                    e for e in emails_by_id.values()
                    if e.get("threadId") == thread_id
                    or e.get("id") == thread_id
                    or extract_thread_id(e) == thread_id
                ]
                matching_emails.sort(key=lambda e: e.get("timestamp", ""))
                thread["emails"] = matching_emails

                # combined_text is already saved in threads_*.json by processing.py —
                # only rebuild it if missing (e.g. older file format)
                if not thread.get("combined_text") and matching_emails:
                    parts = []
                    for j, em in enumerate(matching_emails, 1):
                        body = em.get("clean_text") or em.get("body", "")
                        if body:
                            parts.append(
                                f"[Message {j} | From: {em.get('sender', '')} "
                                f"| {em.get('timestamp', '')}]\n{body}"
                            )
                    thread["combined_text"] = "\n\n".join(parts)

                # email_count from disk is authoritative; only override if missing
                thread.setdefault("email_count", max(len(matching_emails), 1))

            threads = threads_raw

        elif email_files:
            # ── Fallback: no thread file, build one thread per email ──
            for email in email_list:
                threads.append({
                    "thread_id":     email.get("id", ""),
                    "subject":       email.get("subject", ""),
                    "participants":  list({email.get("sender", ""), email.get("recipient", "")} - {""}),
                    "email_count":   1,
                    "date_range":    {
                        "earliest": email.get("timestamp", ""),
                        "latest":   email.get("timestamp", ""),
                    },
                    "combined_text": email.get("clean_text") or email.get("body", ""),
                    "emails":        [email],
                })

    # ── Fallback sample data ──
    if not threads:
        print("⚠️  No preprocessed data found. Using sample data for testing.\n")
        threads = [
            {
                "thread_id":    "sample_001",
                "subject":      "Q3 Report Submission",
                "participants": ["manager@company.com", "harshita@gmail.com"],
                "email_count":  2,
                "date_range":   {"earliest": "2025-06-20 09:00:00",
                                 "latest":   "2025-06-20 11:00:00"},
                "combined_text": (
                    "[Message 1 | From: manager@company.com | 2025-06-20 09:00]\n"
                    "Hi Harshita, the Q3 report must be submitted by June 28th. "
                    "Finance needs it for the board presentation on June 30th. "
                    "Please share a draft by June 26th for review.\n\n"
                    "[Message 2 | From: harshita@gmail.com | 2025-06-20 11:00]\n"
                    "Understood, I'll share a draft by June 26th. "
                    "Could you please share the latest revenue data from finance? "
                    "I need the Q2 actuals to complete section 3."
                ),
                "emails": [],
            },
            {
                "thread_id":    "sample_002",
                "subject":      "Team Sync — Tomorrow 3PM IST",
                "participants": ["lead@company.com", "harshita@gmail.com"],
                "email_count":  1,
                "date_range":   {"earliest": "2025-06-21 08:00:00",
                                 "latest":   "2025-06-21 08:00:00"},
                "combined_text": (
                    "[Message 1 | From: lead@company.com | 2025-06-21 08:00]\n"
                    "Hi Team, reminder about our sync tomorrow at 3PM IST. "
                    "Please come prepared with sprint updates and blockers. "
                    "Agenda: sprint review, Q3 planning kickoff, and infra migration update. "
                    "Join link: meet.google.com/xyz-abc-123"
                ),
                "emails": [],
            },
            {
                "thread_id":    "sample_003",
                "subject":      "Invoice #INV-892 — Payment Overdue",
                "participants": ["billing@vendor.com", "harshita@gmail.com"],
                "email_count":  1,
                "date_range":   {"earliest": "2025-06-22 07:00:00",
                                 "latest":   "2025-06-22 07:00:00"},
                "combined_text": (
                    "[Message 1 | From: billing@vendor.com | 2025-06-22 07:00]\n"
                    "Dear Customer, Invoice #INV-892 for ₹18,500 was due on June 15th "
                    "and is now 7 days overdue. A late fee of ₹500 will apply after June 25th. "
                    "Please process payment at your earliest convenience. "
                    "Pay online: vendor.com/pay or bank transfer details attached."
                ),
                "emails": [],
            },
        ]

    # ── 3. Summarize all threads ──
    summarized = summarize_all(client, threads)

    # ── 4. Print results ──
    print(f"\n{'=' * 62}")
    print(f"  RESULTS — {len(summarized)} Thread(s)")
    print(f"{'=' * 62}\n")

    for thread in summarized:
        print_summary(thread)

    # ── 5. Save to JSON ──
    save_summaries(summarized)

    # ── 6. Persist to SQLite (Module 4 — storage.py) ──
    persist_to_db(summarized)

    print("\n✅ Module 3 + 4 complete! SQLite DB populated, ready for dashboard.")


if __name__ == "__main__":
    main()