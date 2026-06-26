"""
====================================================
  MODULE 2: Email Preprocessing
  Smart Email Summarizer Project
====================================================
  Handles:
   - Strip HTML tags (from raw body if not done yet)
   - Remove email signatures
   - Remove quoted reply chains
   - Remove forwarded message headers
   - Normalize whitespace / encoding artifacts
   - Detect & group email threads
   - Prepare clean text for LLM summarization
====================================================
"""

import re
import json
import os
from datetime import datetime
from collections import defaultdict


# ─────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────

# Common signature markers (case-insensitive)
SIGNATURE_MARKERS = [
    r"^[-–—]{2,}\s*$",                          # -- or --- or ——
    r"^thanks[,.]?\s*$",
    r"^thank you[,.]?\s*$",
    r"^regards[,.]?\s*$",
    r"^best regards[,.]?\s*$",
    r"^warm regards[,.]?\s*$",
    r"^sincerely[,.]?\s*$",
    r"^cheers[,.]?\s*$",
    r"^with thanks[,.]?\s*$",
    r"^yours (truly|sincerely|faithfully)[,.]?\s*$",
    r"^sent from my (iphone|ipad|android|samsung|pixel)",
    r"^get outlook for (ios|android)",
    r"^this email (and any|was sent)",
    r"^confidentiality notice",
    r"^disclaimer[:\s]",
    r"^unsubscribe",
    r"^you('re| are) receiving this",
    r"^\[image:.*\]$",                           # embedded image placeholders
]

# Quoted reply markers (lines that start a reply block)
QUOTE_MARKERS = [
    r"^on .{5,100} wrote:\s*$",                 # "On Mon, Jun 20 2024, John wrote:"
    r"^-{3,}\s*original message\s*-{3,}",       # --- Original Message ---
    r"^-{3,}\s*forwarded message\s*-{3,}",      # --- Forwarded Message ---
    r"^from:\s*.+",                              # "From: someone@email.com" (in reply block)
    r"^sent:\s*.+",                              # "Sent: Monday, June 20"
    r"^to:\s*.+",
    r"^subject:\s*.+",
    r"^date:\s*.+",
    r"^>{1,}",                                   # > quoted line (email clients)
    r"^\[quoted text hidden\]",
    r"^begin forwarded message[:\s]",
]

# Encoding artifacts to clean
ENCODING_ARTIFACTS = [
    (r"=\r?\n",         ""),          # quoted-printable soft line breaks
    (r"=([A-F0-9]{2})", ""),          # quoted-printable encoded chars
    (r"\u00a0",         " "),         # non-breaking space → regular space
    (r"\u200b",         ""),          # zero-width space
    (r"\u2019",         "'"),         # curly apostrophe
    (r"\u2018",         "'"),
    (r"\u201c",         '"'),         # curly quotes
    (r"\u201d",         '"'),
    (r"\r\n",           "\n"),        # Windows CRLF → Unix LF
    (r"\r",             "\n"),
]




def strip_html_tags(text):
    """
    Safety pass: removes any leftover HTML tags from body text.
    Module 1 should already handle this, but run again as a safety net.

    Args:
        text (str): Email body text

    Returns:
        str: Text with HTML tags removed
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove HTML entities like &nbsp; &amp; &lt;
    text = re.sub(r"&[a-zA-Z]{2,6};", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    return text

#  STEP 2: FIX ENCODING ARTIFACTS

def fix_encoding(text):
    """
    Cleans up encoding artifacts from email transfer formats.
    Handles quoted-printable, unicode special chars, CRLF.

    Args:
        text (str): Raw email text

    Returns:
        str: Cleaned text
    """
    for pattern, replacement in ENCODING_ARTIFACTS:
        text = re.sub(pattern, replacement, text)
    return text

#  STEP 3: REMOVE QUOTED REPLIES


def remove_quoted_replies(text):
    """
    Removes quoted reply chains from email body.
    Keeps only the most recent message content.

    Strategies:
    - Detects "On [date], [person] wrote:" patterns
    - Detects "> quoted text" blocks
    - Detects "--- Original Message ---" dividers

    Args:
        text (str): Email body text

    Returns:
        tuple: (cleaned_text, was_quoted: bool)
    """
    lines         = text.splitlines()
    clean_lines   = []
    found_quote   = False
    quote_depth   = 0          # how many consecutive quote lines seen

    compiled_markers = [
        re.compile(marker, re.IGNORECASE) for marker in QUOTE_MARKERS
    ]

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check if this line is a quote marker
        is_quote_line = any(m.match(line) for m in compiled_markers)

        if is_quote_line:
            found_quote = True
            quote_depth += 1

            # Special case: "On [date], X wrote:" often spans 2 lines
            # e.g. "On Mon, Jun 20 2024 at 3:00 PM,\nJohn Doe <john@x.com> wrote:"
            # Check if it's an incomplete "On... wrote:" split across lines
            if re.match(r"^on .{5,80}$", line, re.IGNORECASE) and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if re.search(r"wrote:\s*$", next_line, re.IGNORECASE):
                    i += 2      # skip both lines
                    continue

            # Stop collecting lines — everything below is quoted
            break

        else:
            quote_depth = 0
            clean_lines.append(lines[i])

        i += 1

    cleaned = "\n".join(clean_lines)
    return cleaned, found_quote


# ─────────────────────────────────────────────────
#  STEP 4: REMOVE EMAIL SIGNATURES
# ─────────────────────────────────────────────────

def remove_signature(text):
    """
    Detects and removes email signatures from the body.

    Strategy: Scan from the BOTTOM of the email upward.
    A "signature block" usually appears in the last 15–20 lines
    and starts with a common marker (thanks, regards, --, etc.)

    Args:
        text (str): Email body text

    Returns:
        tuple: (cleaned_text, signature_found: bool)
    """
    lines = text.splitlines()
    compiled_markers = [
        re.compile(marker, re.IGNORECASE) for marker in SIGNATURE_MARKERS
    ]

    # Only scan the last N lines for a signature
    SCAN_LAST_N_LINES = 20
    scan_start = max(0, len(lines) - SCAN_LAST_N_LINES)
    sig_start  = None

    for i in range(scan_start, len(lines)):
        line = lines[i].strip()
        if any(m.match(line) for m in compiled_markers):
            sig_start = i
            break

    if sig_start is not None:
        return "\n".join(lines[:sig_start]), True

    return text, False


# ─────────────────────────────────────────────────
#  STEP 5: REMOVE FORWARDED HEADERS
# ─────────────────────────────────────────────────

def remove_forwarded_headers(text):
    """
    Removes forwarded message header blocks like:
    ---------- Forwarded message ---------
    From: ...
    Date: ...
    Subject: ...
    To: ...

    Keeps the forwarded body content itself.

    Args:
        text (str): Email body text

    Returns:
        tuple: (cleaned_text, was_forwarded: bool)
    """
    # Pattern: forwarded header block (From/Date/Subject/To on consecutive lines)
    fwd_header_pattern = re.compile(
        r"(-{3,}\s*(forwarded|original)\s*(message|mail)\s*-{3,}"
        r"[\s\S]{0,300}?)"
        r"(from:.*\n)?(date:.*\n)?(subject:.*\n)?(to:.*\n)?",
        re.IGNORECASE | re.MULTILINE
    )

    cleaned, count = fwd_header_pattern.subn("", text)
    return cleaned.strip(), count > 0


# ─────────────────────────────────────────────────
#  STEP 6: NORMALIZE WHITESPACE
# ─────────────────────────────────────────────────

def normalize_whitespace(text):
    """
    Final cleanup pass:
    - Collapse 3+ blank lines to max 2
    - Strip leading/trailing whitespace per line
    - Remove lines that are just punctuation or dashes

    Args:
        text (str): Email body text

    Returns:
        str: Normalized text
    """
    lines = text.splitlines()

    cleaned = []
    blank_count = 0

    for line in lines:
        stripped = line.strip()

        # Skip lines that are only dashes, dots, underscores (decorative separators)
        if re.match(r"^[-=_.]{3,}$", stripped):
            continue

        if stripped == "":
            blank_count += 1
            if blank_count <= 1:   # allow at most 1 consecutive blank line
                cleaned.append("")
        else:
            blank_count = 0
            cleaned.append(stripped)

    return "\n".join(cleaned).strip()


# ─────────────────────────────────────────────────
#  STEP 7: FULL PIPELINE — CLEAN ONE EMAIL
# ─────────────────────────────────────────────────

def clean_email_body(raw_body):
    """
    Runs the full preprocessing pipeline on a single email body.

    Pipeline order:
        1. Strip leftover HTML
        2. Fix encoding artifacts
        3. Remove forwarded headers
        4. Remove quoted replies
        5. Remove signature
        6. Normalize whitespace

    Args:
        raw_body (str): Raw email body from Module 1

    Returns:
        dict: {
            "clean_text":       str,   ← ready for LLM
            "had_html":         bool,
            "had_quoted":       bool,
            "had_signature":    bool,
            "was_forwarded":    bool,
            "char_count_before": int,
            "char_count_after":  int,
        }
    """
    original_len = len(raw_body)

    # ── 1. Strip HTML ──
    had_html = bool(re.search(r"<[a-zA-Z]", raw_body))
    text = strip_html_tags(raw_body)

    # ── 2. Fix encoding ──
    text = fix_encoding(text)

    # ── 3. Remove forwarded headers ──
    text, was_forwarded = remove_forwarded_headers(text)

    # ── 4. Remove quoted replies ──
    text, had_quoted = remove_quoted_replies(text)

    # ── 5. Remove signature ──
    text, had_signature = remove_signature(text)

    # ── 6. Normalize whitespace ──
    text = normalize_whitespace(text)

    return {
        "clean_text":        text,
        "had_html":          had_html,
        "had_quoted":        had_quoted,
        "had_signature":     had_signature,
        "was_forwarded":     was_forwarded,
        "char_count_before": original_len,
        "char_count_after":  len(text),
        "reduction_pct":     round((1 - len(text) / max(original_len, 1)) * 100, 1),
    }

#  STEP 8: THREAD GROUPING

def extract_thread_id(email):
    """
    Determines a thread group key for an email.

    Strategy (in priority order):
    1. Use Gmail's threadId if available (most reliable)
    2. Normalize subject line (strip Re:, Fwd:, whitespace)
       and group by (normalized_subject + participants)

    Args:
        email (dict): Parsed email from Module 1

    Returns:
        str: Thread group key
    """
    # Gmail provides threadId — use it directly if available
    if "threadId" in email and email["threadId"]:
        return email["threadId"]

    # Fallback: normalize subject
    subject = email.get("subject", "")

    # Remove Re:, Fwd:, Fw:, [EXTERNAL] etc. prefixes
    subject_clean = re.sub(
        r"^(re|fwd?|fw|回复|转发|antw|aw|sv|vs|enc|ref)[\s:：]+",
        "",
        subject,
        flags=re.IGNORECASE
    ).strip().lower()

    # Build participant set (sender + recipient, normalized)
    participants = set()
    for field in ["sender", "recipient"]:
        raw = email.get(field, "")
        # Extract email addresses from "Name <email>" format
        emails_found = re.findall(r"[\w.+-]+@[\w.-]+", raw)
        participants.update(e.lower() for e in emails_found)

    participant_key = "_".join(sorted(participants))
    return f"{subject_clean}|{participant_key}"


def group_into_threads(emails):
    """
    Groups a list of emails into conversation threads.
    Sorts each thread chronologically (oldest first).

    Args:
        emails (list[dict]): List of parsed emails from Module 1

    Returns:
        list[dict]: List of thread objects:
            {
                "thread_id":    str,
                "subject":      str,      ← representative subject
                "participants": list[str],
                "email_count":  int,
                "date_range":   { "earliest": str, "latest": str },
                "emails":       list[dict],   ← sorted by timestamp
                "combined_text": str,         ← all clean bodies joined
            }
    """
    threads = defaultdict(list)

    for email in emails:
        thread_id = extract_thread_id(email)
        threads[thread_id].append(email)

    result = []

    for thread_id, thread_emails in threads.items():
        # Sort emails in thread by timestamp (oldest first)
        def parse_ts(e):
            try:
                return datetime.strptime(e.get("timestamp", ""), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return datetime.min

        thread_emails.sort(key=parse_ts)

        # Collect all participants
        participants = set()
        for e in thread_emails:
            for field in ["sender", "recipient"]:
                raw = e.get(field, "")
                found = re.findall(r"[\w.+-]+@[\w.-]+", raw)
                participants.update(f.lower() for f in found)

        # Use the first (oldest) email's subject as thread subject
        subject = thread_emails[0].get("subject", "(No Subject)")
        subject = re.sub(r"^(re|fwd?|fw)[\s:：]+", "", subject, flags=re.IGNORECASE).strip()

        # Timestamps for date range
        timestamps = [e.get("timestamp", "") for e in thread_emails if e.get("timestamp")]

        # Combine clean bodies with clear separators for LLM context
        combined_parts = []
        for i, e in enumerate(thread_emails, 1):
            clean  = e.get("clean_text", e.get("body", "")).strip()
            sender = e.get("sender", "Unknown")
            ts     = e.get("timestamp", "")
            if clean:
                combined_parts.append(
                    f"[Message {i} | From: {sender} | {ts}]\n{clean}"
                )

        combined_text = "\n\n".join(combined_parts)

        result.append({
            "thread_id":     thread_id,
            "subject":       subject,
            "participants":  sorted(participants),
            "email_count":   len(thread_emails),
            "date_range": {
                "earliest": min(timestamps) if timestamps else "",
                "latest":   max(timestamps) if timestamps else "",
            },
            "emails":        thread_emails,
            "combined_text": combined_text,
        })

    # Sort threads by latest email (most recent thread first)
    result.sort(
        key=lambda t: t["date_range"]["latest"],
        reverse=True
    )

    return result


# ─────────────────────────────────────────────────
#  STEP 9: PROCESS ALL EMAILS (main batch runner)
# ─────────────────────────────────────────────────

def preprocess_emails(emails):
    """
    Runs full preprocessing on a list of emails from Module 1.

    Steps:
        1. Clean each email body
        2. Attach clean_text back to email dict
        3. Group into threads
        4. Return threads ready for LLM summarization

    Args:
        emails (list[dict]): Raw emails from Module 1

    Returns:
        tuple: (processed_emails, threads)
    """
    print(f"🧹 Preprocessing {len(emails)} emails...\n")

    processed = []
    stats = {
        "had_html":       0,
        "had_quoted":     0,
        "had_signature":  0,
        "was_forwarded":  0,
        "total_reduced":  0,
    }

    for i, email in enumerate(emails, 1):
        raw_body = email.get("body", "")
        result   = clean_email_body(raw_body)

        # Attach cleaned fields back to email
        email["clean_text"]        = result["clean_text"]
        email["had_html"]          = result["had_html"]
        email["had_quoted"]        = result["had_quoted"]
        email["had_signature"]     = result["had_signature"]
        email["was_forwarded"]     = result["was_forwarded"]
        email["body_reduction_pct"] = result["reduction_pct"]

        # Collect stats
        for key in ["had_html", "had_quoted", "had_signature", "was_forwarded"]:
            if result[key]:
                stats[key] += 1
        stats["total_reduced"] += result["reduction_pct"]

        print(f"  [{i}/{len(emails)}] ✓ {email.get('subject','')[:50]}")
        print(f"           Chars: {result['char_count_before']} → {result['char_count_after']}"
              f"  ({result['reduction_pct']}% reduced)"
              f"  | HTML:{result['had_html']}"
              f"  Quoted:{result['had_quoted']}"
              f"  Sig:{result['had_signature']}")

        processed.append(email)

    # Print summary stats
    avg_reduction = stats["total_reduced"] / max(len(emails), 1)
    print(f"""
📊 Preprocessing Summary:
   Total emails     : {len(emails)}
   Had HTML         : {stats['had_html']}
   Had quoted reply : {stats['had_quoted']}
   Had signature    : {stats['had_signature']}
   Were forwarded   : {stats['was_forwarded']}
   Avg text reduced : {avg_reduction:.1f}%
""")

    # Group into threads
    print("Grouping emails into threads...")
    threads = group_into_threads(processed)
    print(f"   Found {len(threads)} thread(s) from {len(emails)} email(s)\n")

    return processed, threads




def save_preprocessed(processed_emails, threads, output_dir="preprocessed"):
    """
    Saves preprocessed emails and grouped threads to JSON.

    Args:
        processed_emails (list[dict])
        threads (list[dict])
        output_dir (str)

    Returns:
        tuple: (emails_path, threads_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    emails_path  = os.path.join(output_dir, f"processed_emails_{ts}.json")
    threads_path = os.path.join(output_dir, f"threads_{ts}.json")

    # Save processed emails (without full email list inside threads to avoid duplication)
    with open(emails_path, "w", encoding="utf-8") as f:
        json.dump(processed_emails, f, indent=2, ensure_ascii=False)

    # For threads JSON, exclude nested "emails" list to keep it readable
    threads_summary = []
    for t in threads:
        summary = {k: v for k, v in t.items() if k != "emails"}
        threads_summary.append(summary)

    with open(threads_path, "w", encoding="utf-8") as f:
        json.dump(threads_summary, f, indent=2, ensure_ascii=False)

    print(f"Saved processed emails → {emails_path}")
    print(f"Saved threads          → {threads_path}")
    return emails_path, threads_path


# ─────────────────────────────────────────────────
#  PRINT HELPERS
# ─────────────────────────────────────────────────

def print_thread(thread, preview_chars=300):
    """Pretty-print a single grouped thread."""
    divider = "─" * 60

    print(divider)
    print(f"Thread   : {thread['subject']}")
    print(f"Messages : {thread['email_count']}")
    print(f"People   : {', '.join(thread['participants'])}")
    print(f"Dates    : {thread['date_range']['earliest']}  →  {thread['date_range']['latest']}")

    preview = thread["combined_text"][:preview_chars]
    print(f"\n Combined Preview:\n{preview}")
    if len(thread["combined_text"]) > preview_chars:
        print(f"   ... [{len(thread['combined_text']) - preview_chars} more characters]")

    print(divider + "\n")

#  MAIN — Test with sample data or real emails
# ─────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SMART EMAIL SUMMARIZER — Module 2: Preprocessing")
    print("=" * 60 + "\n")

    # ── Option A: Load from Module 1 output ──
    EMAILS_JSON = "fetched_emails"  # folder from Module 1

    emails = []
    if os.path.exists(EMAILS_JSON):
        json_files = sorted([
            f for f in os.listdir(EMAILS_JSON) if f.endswith(".json")
        ], reverse=True)

        if json_files:
            latest = os.path.join(EMAILS_JSON, json_files[0])
            print(f"Loading emails from: {latest}\n")
            with open(latest, "r", encoding="utf-8") as f:
                emails = json.load(f)

    # ── Option B: Use sample emails for testing ──
    if not emails:
        print("No fetched emails found. Using sample test data.\n")
        emails = [
            {
                "id": "001",
                "subject": "Re: Project Deadline Update",
                "sender": "manager@company.com",
                "recipient": "harshita@gmail.com",
                "timestamp": "2025-06-20 10:30:00",
                "body": """Hi Harshita,

Yes, the deadline has been pushed to June 30th. Please make sure the
report is ready by then.

Thanks,
Raj

On Mon, Jun 20, 2025 at 9:00 AM Harshita <harshita@gmail.com> wrote:
> Hi Raj,
> Can you confirm the deadline for the project?
> Thanks

--
Raj Kumar
Senior Manager | Company Inc.
raj@company.com | +91 98765 43210
""",
            },
            {
                "id": "002",
                "subject": "Re: Project Deadline Update",
                "sender": "harshita@gmail.com",
                "recipient": "manager@company.com",
                "timestamp": "2025-06-20 11:00:00",
                "body": """Thanks Raj, noted!

I'll have it done by the 28th to give buffer time.

Harshita

On Mon, Jun 20, 2025 at 10:30 AM Raj Kumar <manager@company.com> wrote:
> Yes, the deadline has been pushed to June 30th.
> Please make sure the report is ready by then.
> Thanks, Raj
""",
            },
            {
                "id": "003",
                "subject": "Meeting Tomorrow at 3PM",
                "sender": "team@company.com",
                "recipient": "harshita@gmail.com",
                "timestamp": "2025-06-21 08:00:00",
                "body": """<html><body>
<p>Hi Team,</p>
<p>Just a reminder that we have a <b>team sync tomorrow at 3PM IST</b>.</p>
<p>Please review the <a href="#">Q3 roadmap doc</a> beforehand.</p>
<br>
<p>Best regards,<br>Team Lead</p>
<p style="color:gray;font-size:11px">This email was sent from our internal system.
Confidentiality notice: this message is for the intended recipient only.</p>
</body></html>""",
            },
        ]

    # ── Run preprocessing ──
    processed_emails, threads = preprocess_emails(emails)

    # ── Print threads ──
    print(f"\n{'=' * 60}")
    print(f"  GROUPED THREADS ({len(threads)} total)")
    print(f"{'=' * 60}\n")

    for thread in threads:
        print_thread(thread, preview_chars=400)

    # ── Save output ──
    save_preprocessed(processed_emails, threads)

    print("\n Module 2 complete! Clean threads ready for Module 3 (LLM Summarization)")


if __name__ == "__main__":
    main()