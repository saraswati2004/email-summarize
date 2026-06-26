import os
import base64
import json
import sys
import re
from datetime import datetime
from email import message_from_bytes
from bs4 import BeautifulSoup

from google.auth.transport.requests import Request 
from google.oauth2.credentials import Credentials
from google.auth.exceptions import RefreshError
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import atexit
#  CONFIGURATION

SCOPES           = ["https://www.googleapis.com/auth/gmail.readonly"]
CREDENTIALS_FILE = "credentials.json"
TOKEN_DIR        = "tokens"          # folder to store one token per account
MAX_EMAILS       = 10
EMAIL_FILTER     = "newer_than:7d"


def list_saved_accounts():
    """
    Returns all Gmail accounts that already have a saved token.
    """
    if not os.path.exists(TOKEN_DIR):
        return []
    return [
        f.replace("token_", "").replace(".json", "")
        for f in os.listdir(TOKEN_DIR)
        if f.startswith("token_") and f.endswith(".json")
    ]

#  STEP 1: AUTHENTICATE

def authenticate_gmail(email_hint=None):
    """
    Authenticates a Gmail account and saves its token separately.

    - If email_hint is given (e.g. "user1@gmail.com"), loads that account's token.
    - If token doesn't exist, opens browser to log in and saves a NEW token file.
    - Each account gets its own file: tokens/token_user1@gmail.com.json

    Args:
        email_hint (str): Gmail address to authenticate. None = ask user to pick.

    Returns:
        tuple: (service, email_address)
    """
    os.makedirs(TOKEN_DIR, exist_ok=True)

    # ── Show saved accounts and let user pick ──
    saved = list_saved_accounts()

    if not email_hint:
        if saved:
            print("\n Saved accounts:")
            for i, acc in enumerate(saved, 1):
                print(f"   [{i}] {acc}")
            print(f"   [N] Add a new account\n")

            choice = input("Pick an account number or N to add new: ").strip()

            if choice.lower() == "n":
                email_hint = None   # will trigger new login below
            elif choice.isdigit() and 1 <= int(choice) <= len(saved):
                email_hint = saved[int(choice) - 1]
            else:
                print("Invalid choice. Adding new account.")
                email_hint = None
        else:
            print("No saved accounts found. Starting new login...")

    # ── Load existing token for this account ──
    token_file = os.path.join(TOKEN_DIR, f"token_{email_hint}.json") if email_hint else None
    creds = None

    if token_file and os.path.exists(token_file):
        try:
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)
            print(f"Loaded saved token for: {email_hint}")
        except Exception as e:
            print(f"Error loading token: {e}. Re-authenticating...")
            creds = None

    # ── Refresh if expired ──
    if creds and creds.expired and creds.refresh_token:
        try:
            print("Refreshing expired token...")
            creds.refresh(Request())
        except RefreshError:
            print("Token expired and can't refresh. Please log in again.")
            creds = None

    # ── New login if no valid token ──
    if not creds or not creds.valid:
        if not os.path.exists(CREDENTIALS_FILE):
            print(f"Error: '{CREDENTIALS_FILE}' not found.", file=sys.stderr)
            return None, None

        print("Opening browser for Google login...")
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)

        # ── Find out which account just logged in ──
        import googleapiclient.discovery as disc
        temp_service = disc.build("gmail", "v1", credentials=creds)
        profile      = temp_service.users().getProfile(userId="me").execute()
        email_hint   = profile.get("emailAddress", "unknown")

        # ── Save token with account name ──
        token_file = os.path.join(TOKEN_DIR, f"token_{email_hint}.json")
        with open(token_file, "w") as f:
            f.write(creds.to_json())
        print(f"Token saved for: {email_hint}")

    service = build("gmail", "v1", credentials=creds)
    print(f"Authenticated as: {email_hint}\n")
    return service, email_hint

#  STEP 2: HTML → PLAIN TEXT STRIPPER

def strip_html(html_content):
    """
    Converts HTML email body to clean plain text.
    Removes scripts, styles, links, extra whitespace.

    Args:
        html_content (str): Raw HTML string

    Returns:
        str: Clean plain text
    """
    soup = BeautifulSoup(html_content, "lxml")

    # Remove invisible/junk elements
    for tag in soup(["script", "style", "head", "meta", "link", "img"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines()]
    clean_lines = [line for line in lines if line]       # remove empty lines
    clean_text = "\n".join(clean_lines)

    # Remove leftover HTML entities like &amp; &nbsp;
    clean_text = re.sub(r"&[a-z]+;", " ", clean_text)
    clean_text = re.sub(r"\s{2,}", " ", clean_text)      # collapse extra spaces

    return clean_text.strip()


#  STEP 3: PARSE EMAIL PARTS (body + attachments)
 

def parse_email_parts(payload):
    """
    Recursively parses MIME parts of an email.
    Extracts plain text, HTML fallback, and attachment metadata.

    Args:
        payload (dict): Gmail message payload

    Returns:
        dict: { "body": str, "attachments": list }
    """
    body        = ""
    attachments = []
    parts       = payload.get("parts", [])
    mime_type   = payload.get("mimeType", "")

    def decode_data(data):
        """Decode base64url Gmail data."""
        return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")

    # Single-part email (no nested parts)
    if not parts:
        data = payload.get("body", {}).get("data", "")
        if data:
            raw = decode_data(data)
            body = strip_html(raw) if "html" in mime_type else raw
        return {"body": body, "attachments": attachments}

    # Multi-part email — recurse through all parts
    for part in parts:
        part_mime = part.get("mimeType", "")
        filename  = part.get("filename", "")
        part_data = part.get("body", {}).get("data", "")

        # ── Attachment detected ──
        if filename:
            attachment_id = part.get("body", {}).get("attachmentId", "")
            attachments.append({
                "filename":      filename,
                "mimeType":      part_mime,
                "attachmentId":  attachment_id,
                "size_bytes":    part.get("body", {}).get("size", 0)
            })

        # ── Plain text body ── (preferred)
        elif part_mime == "text/plain" and part_data:
            body = decode_data(part_data)

        # ── HTML body ── (fallback if no plain text)
        elif part_mime == "text/html" and part_data and not body:
            body = strip_html(decode_data(part_data))

        # ── Nested multipart (e.g. multipart/alternative inside multipart/mixed) ──
        elif "multipart" in part_mime:
            nested = parse_email_parts(part)
            if nested["body"] and not body:
                body = nested["body"]
            attachments.extend(nested["attachments"])

    return {"body": body, "attachments": attachments}


# ─────────────────────────────────────────────────
#  STEP 4: PARSE SINGLE EMAIL MESSAGE
# ─────────────────────────────────────────────────

def parse_email(service, msg_id):
    """
    Fetches and parses a single email by message ID.

    Args:
        service: Gmail API service object
        msg_id (str): Gmail message ID

    Returns:
        dict: Structured email data
    """
    message = service.users().messages().get(
        userId="me",
        id=msg_id,
        format="full"
    ).execute()

    headers   = message.get("payload", {}).get("headers", [])
    header_map = {h["name"].lower(): h["value"] for h in headers}

    # ── Extract headers ──
    subject   = header_map.get("subject",  "(No Subject)")
    sender    = header_map.get("from",     "(Unknown Sender)")
    recipient = header_map.get("to",       "(Unknown)")
    date_str  = header_map.get("date",     "")
    msg_id_hdr = header_map.get("message-id", "")

    # ── Parse timestamp ──
    try:
        # Gmail also gives internal timestamp in milliseconds
        timestamp_ms = int(message.get("internalDate", 0))
        timestamp    = datetime.fromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        timestamp = date_str  # fallback to header date string

    # ── Parse body + attachments ──
    parsed = parse_email_parts(message.get("payload", {}))

    # ── Labels (INBOX, UNREAD, IMPORTANT, etc.) ──
    labels = message.get("labelIds", [])

    return {
        "id":           msg_id,
        "threadId":     message.get("threadId", ""),   # ← ADD THIS LINE
        "message_id":   msg_id_hdr,
        "subject":      subject,
        "sender":       sender,
        "recipient":    recipient,
        "timestamp":    timestamp,
        "date_raw":     date_str,
        "body":         parsed["body"],
        "attachments":  parsed["attachments"],
        "labels":       labels,
        "snippet":      message.get("snippet", ""),
    }



# ─────────────────────────────────────────────────
#  STEP 5: FETCH MULTIPLE EMAILS
# ─────────────────────────────────────────────────

def fetch_emails(service, query=EMAIL_FILTER, max_results=MAX_EMAILS):
    print(f"Fetching emails with query: '{query}' (max: {max_results})\n")

    all_messages = []
    page_token = None

    # Keep fetching pages until we have enough or run out
    while len(all_messages) < max_results:
        batch_size = min(max_results - len(all_messages), 100)  # Gmail max per page is 100

        request_kwargs = {
            "userId": "me",
            "q": query,
            "maxResults": batch_size
        }

        if page_token:
            request_kwargs["pageToken"] = page_token  # ← attach token for next page

        result = service.users().messages().list(**request_kwargs).execute()

        messages = result.get("messages", [])
        all_messages.extend(messages)

        page_token = result.get("nextPageToken")  # ← grab token for next page

        print(f"  Fetched {len(all_messages)} so far...")

        if not page_token:
            break  # no more pages left

    if not all_messages:
        print("No emails found matching the query.")
        return []

    print(f"Found {len(all_messages)} email(s). Parsing...\n")

    emails = []
    for i, msg in enumerate(all_messages, 1):
        try:
            email_data = parse_email(service, msg["id"])
            emails.append(email_data)
            print(f"  [{i}/{len(all_messages)}] ✓ Parsed: {email_data['subject'][:60]}")
        except Exception as e:
            print(f"  [{i}/{len(all_messages)}] ✗ Error parsing message {msg['id']}: {e}")

    return emails

def print_email(email, body_preview_chars=300):
    """
    Pretty-prints a parsed email to the console.

    Args:
        email (dict):           Parsed email object
        body_preview_chars:     How many chars of body to show
    """
    divider = "─" * 60

    print(divider)
    print(f" Subject  : {email['subject']}")
    print(f" From     : {email['sender']}")
    print(f" To       : {email['recipient']}")
    print(f" Time     : {email['timestamp']}")
    print(f"  Labels   : {', '.join(email['labels'])}")

    # Attachments
    if email["attachments"]:
        print(f"\n📎 Attachments ({len(email['attachments'])}):")
        for att in email["attachments"]:
            size_kb = att["size_bytes"] / 1024
            print(f"   • {att['filename']} [{att['mimeType']}] ({size_kb:.1f} KB)")

    # Body preview
    body = email["body"].strip()
    if body:
        preview = body[:body_preview_chars]
        print(f"\n Body Preview:\n{preview}")
        if len(body) > body_preview_chars:
            print(f"   ... [{len(body) - body_preview_chars} more characters]")
    else:
        print(f"\n Snippet: {email['snippet']}")

    print(divider + "\n")


# ─────────────────────────────────────────────────
#  STEP 7: SAVE TO JSON
# ─────────────────────────────────────────────────

def save_emails_to_json(emails, output_dir="fetched_emails"):
    os.makedirs(output_dir, exist_ok=True)
    filename = datetime.now().strftime("emails_%Y%m%d_%H%M%S.json")
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(emails, f, indent=2, ensure_ascii=False)

    print(f"\n Saved {len(emails)} emails → {filepath}")
    return filepath


# ─────────────────────────────────────────────────
#  MAIN — Run this file directly to test
# ─────────────────────────────────────────────────

def main():
    # Authenticate — user picks which account
    service, active_email = authenticate_gmail()

    if not service:
        print("Authentication failed.", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching emails for: {active_email}\n")

    emails = fetch_emails(service, query=EMAIL_FILTER, max_results=MAX_EMAILS)

    if not emails:
        return

    for email in emails:
        print_email(email, body_preview_chars=400)

    save_emails_to_json(emails)
if __name__ == "__main__":
    main()