"""
====================================================
  MODULE 5: API Server — FastAPI
  Smart Email Summarizer Project
====================================================
  Exposes the full pipeline (mail.py → processing.py →
  summarizer.py → storage.py) over HTTP so index.html
  (or anything else) can read results and trigger runs.

  Endpoints (all consumed by index.html):
    GET  /summaries          -> list of saved summaries
    GET  /summaries/{id}     -> single summary by email_id
    GET  /stats              -> dashboard stats / analytics
    POST /summarize          -> runs the full pipeline
    GET  /health             -> simple liveness check

  Extra (not required by index.html, but useful):
    GET  /summaries/priority/{priority}
    GET  /summaries/sender/{sender_key}
    GET  /summaries/search?q=...
    GET  /summaries/export.json
    GET  /summaries/export.csv
    POST /summarize/account   (?email=someone@gmail.com)
====================================================
  Run:
    pip install -r requirements.txt
    uvicorn api:app --reload --port 8000

  Then open index.html in a browser (Settings -> API URL
  should already default to http://localhost:8000).
====================================================
"""

import os
import asyncio
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from storage import StorageManager
import mail
import processing
import summarizer

# ─────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("mailbrain")

# ─────────────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────────────

app = FastAPI(
    title="MailBrain API",
    description="Smart Email Summarizer — Gmail fetch, clean, summarize (Gemini), store, serve.",
    version="1.0.0",
)

# index.html is opened as a static file (file:// or any origin) -- allow all origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage = StorageManager()

# Optional simple API-key gate. If API_KEY env var is unset, auth is disabled
# (index.html's Settings drawer has an "API Key" field for this).
API_KEY = os.environ.get("MAILBRAIN_API_KEY")


def check_api_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


# Tracks pipeline run status so we don't kick off two runs at once
# and so callers can poll if they want to.
pipeline_state = {"running": False, "last_result": None, "last_error": None}


# ─────────────────────────────────────────────────
#  RESPONSE MODELS
# ─────────────────────────────────────────────────

class SummarizeResponse(BaseModel):
    status: str
    detail: str


class StatsResponse(BaseModel):
    total_summaries: int
    by_priority: dict
    with_action_items: int
    top_senders_by_sender_key: dict


# ─────────────────────────────────────────────────
#  HEALTH
# ─────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "pipeline_running": pipeline_state["running"]}


# ─────────────────────────────────────────────────
#  SUMMARIES — read endpoints
# ─────────────────────────────────────────────────

@app.get("/summaries")
def get_summaries(
    limit: Optional[int] = Query(default=20, ge=1, le=1000, description="Max rows (use a high number for 'all')"),
):
    """
    Returns the most recent summaries, newest first.
    This is what index.html calls on load: /summaries?limit=200
    """
    try:
        return storage.get_recent_summaries(limit=limit)
    except Exception as e:
        log.exception("Failed to fetch summaries")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summaries/{email_id}")
def get_summary(email_id: str):
    result = storage.get_summary_by_id(email_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"No summary found for email_id '{email_id}'")
    return result


@app.get("/summaries/priority/{priority}")
def get_summaries_by_priority(priority: str):
    priority = priority.lower()
    if priority not in {"low", "medium", "high", "urgent"}:
        raise HTTPException(status_code=400, detail="priority must be one of: low, medium, high, urgent")
    return storage.get_summaries_by_priority(priority)


@app.get("/summaries/sender/{sender_key}")
def get_summaries_by_sender(sender_key: str):
    return storage.get_summaries_by_sender_key(sender_key.lower())


@app.get("/summaries/with-actions")
def get_summaries_with_actions():
    return storage.get_summaries_with_action_items()


@app.get("/summaries/search")
def search_summaries(q: str = Query(..., min_length=1)):
    return storage.search_summaries(q)


# ─────────────────────────────────────────────────
#  STATS
# ─────────────────────────────────────────────────

@app.get("/stats", response_model=StatsResponse)
def get_stats(top_n: int = Query(default=10, ge=1, le=50)):
    try:
        return storage.get_sender_analytics(top_n=top_n)
    except Exception as e:
        log.exception("Failed to compute stats")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────
#  EXPORTS
# ─────────────────────────────────────────────────

@app.get("/summaries/export.json")
def export_json():
    path = storage.export_to_json()
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=500, detail="Export failed")
    return FileResponse(path, filename=os.path.basename(path), media_type="application/json")


@app.get("/summaries/export.csv")
def export_csv():
    path = storage.export_to_csv()
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=500, detail="Export failed")
    return FileResponse(path, filename=os.path.basename(path), media_type="text/csv")


# ─────────────────────────────────────────────────
#  PIPELINE — runs mail.py -> processing.py -> summarizer.py -> storage.py
# ─────────────────────────────────────────────────

def _run_pipeline_sync(email_hint: Optional[str] = None) -> dict:
    """
    Runs the full pipeline synchronously (blocking). Executed in a worker
    thread by the async endpoint below so it doesn't block the event loop.
    """
    log.info("Pipeline started (account=%s)", email_hint or "default/saved")

    # 1) Authenticate + fetch from Gmail
    service, active_email = mail.authenticate_gmail(email_hint=email_hint)
    if not service:
        raise RuntimeError(
            "Gmail authentication failed. Make sure credentials.json exists "
            "next to api.py and that you've logged in at least once."
        )

    emails = mail.fetch_emails(service, query=mail.EMAIL_FILTER, max_results=mail.MAX_EMAILS)
    if not emails:
        return {"account": active_email, "fetched": 0, "summarized": 0, "saved": 0}

    # 2) Clean + thread
    processed_emails, threads = processing.preprocess_emails(emails)

    # 3) Summarize via Gemini
    client = summarizer.get_client()
    summarized = summarizer.summarize_all(client, threads)

    # 4) Persist to SQLite
    summarizer.persist_to_db(summarized)

    succeeded = sum(1 for t in summarized if t.get("summary_result"))

    log.info(
        "Pipeline complete (account=%s): fetched=%d threads=%d summarized=%d",
        active_email, len(emails), len(threads), succeeded,
    )

    return {
        "account": active_email,
        "fetched": len(emails),
        "threads": len(threads),
        "summarized": succeeded,
    }


@app.post("/summarize", response_model=SummarizeResponse)
async def run_pipeline(background_tasks: BackgroundTasks):
    """
    Triggers the full pipeline: fetch new emails -> clean -> summarize -> store.
    This is what index.html's "Run Pipeline" button calls.

    Runs in the background and returns immediately; poll GET /health or
    GET /summaries afterward to see results, matching index.html's behavior
    of refreshing ~1.5s after this call returns.
    """
    if pipeline_state["running"]:
        raise HTTPException(status_code=409, detail="Pipeline is already running")

    async def _runner():
        pipeline_state["running"] = True
        pipeline_state["last_error"] = None
        try:
            result = await asyncio.to_thread(_run_pipeline_sync, None)
            pipeline_state["last_result"] = result
        except Exception as e:
            log.exception("Pipeline failed")
            pipeline_state["last_error"] = str(e)
        finally:
            pipeline_state["running"] = False

    background_tasks.add_task(_runner)
    return {"status": "started", "detail": "Pipeline running in background. Check /health or /summaries shortly."}


@app.post("/summarize/account", response_model=SummarizeResponse)
async def run_pipeline_for_account(background_tasks: BackgroundTasks, email: str = Query(...)):
    """Same as /summarize but for a specific saved Gmail account."""
    if pipeline_state["running"]:
        raise HTTPException(status_code=409, detail="Pipeline is already running")

    async def _runner():
        pipeline_state["running"] = True
        pipeline_state["last_error"] = None
        try:
            result = await asyncio.to_thread(_run_pipeline_sync, email)
            pipeline_state["last_result"] = result
        except Exception as e:
            log.exception("Pipeline failed")
            pipeline_state["last_error"] = str(e)
        finally:
            pipeline_state["running"] = False

    background_tasks.add_task(_runner)
    return {"status": "started", "detail": f"Pipeline running for {email} in background."}


@app.get("/summarize/status")
def pipeline_status():
    return pipeline_state


# ─────────────────────────────────────────────────
#  ACCOUNTS
# ─────────────────────────────────────────────────

@app.get("/accounts")
def list_accounts():
    """Lists Gmail accounts with a saved OAuth token."""
    return {"accounts": mail.list_saved_accounts()}


# ─────────────────────────────────────────────────
#  MAINTENANCE
# ─────────────────────────────────────────────────

@app.post("/maintenance/backup")
def backup_db():
    path = storage.backup_database()
    if not path:
        raise HTTPException(status_code=500, detail="Backup failed")
    return {"status": "ok", "backup_file": path}


@app.delete("/maintenance/old")
def delete_old(days: int = Query(default=90, ge=1)):
    deleted = storage.delete_old_summaries(days=days)
    return {"status": "ok", "deleted": deleted}


@app.delete("/maintenance/clear-all")
def clear_all():
    ok = storage.clear_all_data()
    if not ok:
        raise HTTPException(status_code=500, detail="Clear failed")
    return {"status": "ok"}


# ─────────────────────────────────────────────────
#  LOCAL DEV ENTRYPOINT
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)