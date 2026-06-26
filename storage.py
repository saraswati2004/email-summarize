"""
====================================================
  MODULE 4: Storage Layer — SQLite Database
  Smart Email Summarizer Project
====================================================
  Handles:
   - Initialize SQLite database schema
   - Store processed email summaries
   - Track processed emails (avoid reprocessing)
   - Query summarized emails by various filters
   - Export summaries to JSON/CSV
   - Manage database lifecycle (create, backup, clean)
====================================================
  Setup:
    pip install sqlite3 (built-in with Python)

  Database file: summaries.db (created automatically)
====================================================
"""
import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────

DB_BACKUP_DIR = "backups"

# Use a stable DB path relative to this file to avoid CWD mismatches
DB_FILE = str((Path(__file__).resolve().parent / "summaries.db"))



# ─────────────────────────────────────────────────
#  DATABASE CONNECTION & INITIALIZATION
# ─────────────────────────────────────────────────

class StorageManager:
    """Manages all database operations for email summaries."""

    def __init__(self, db_file: str = DB_FILE):
        """
        Initialize storage manager and create database if needed.

        Args:
            db_file (str): Path to SQLite database file
        """
        self.db_file = db_file
        self.init_database()

    def get_connection(self):
        conn = sqlite3.connect(self.db_file)
        conn.execute("PRAGMA key='your-password-here'")  # encrypts DB
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self):
        """
        Initialize database schema if tables don't exist.
        Creates: summaries, processed_emails
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # ── Table 1: Email Summaries ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_id TEXT UNIQUE NOT NULL,
                sender TEXT NOT NULL,
                sender_key TEXT,
                subject TEXT NOT NULL,
                summary TEXT NOT NULL,

                action_items TEXT,

                priority TEXT DEFAULT 'medium',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                tags TEXT,
                CONSTRAINT chk_priority CHECK (priority IN ('low', 'medium', 'high', 'urgent'))
            )
        """)

        # ── Table 2: Processed Emails (for deduplication) ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_id TEXT UNIQUE NOT NULL,
                message_id TEXT,
                sender TEXT NOT NULL,
                subject TEXT NOT NULL,
                processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'completed'
            )
        """)

        # ── Create indexes for faster queries ──
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_email_id ON summaries(email_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sender ON summaries(sender)")
        # sender_key index (sender_key column may not exist in older DBs)
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sender_key ON summaries(sender_key)")
        except sqlite3.OperationalError:
            pass

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_priority ON summaries(priority)")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON summaries(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_email_id ON processed_emails(email_id)")

        conn.commit()
        conn.close()
        print(f"✅ Database initialized: {self.db_file}")

    # ─────────────────────────────────────────────────
    #  INSERT OPERATIONS
    # ─────────────────────────────────────────────────

    def save_summary(
        self,
        email_id: str,
        sender: str,
        sender_key: Optional[str],
        subject: str,
        summary: str,

        action_items: Optional[List[str]] = None,
        priority: str = "medium",
        tags: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
    ) -> bool:
        """
        Save a processed email summary to the database.

        Args:
            email_id (str):              Unique Gmail message ID
            sender (str):                Sender's email address
            subject (str):               Email subject
            summary (str):               Generated summary text
            action_items (list, optional): List of action items extracted
            priority (str):              Priority level: low, medium, high, urgent
            tags (list, optional):       Custom tags for filtering
            timestamp (str, optional):   Original email timestamp

        Returns:
            bool: True if saved successfully, False if duplicate
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            action_items_json = json.dumps(action_items) if action_items else None
            tags_json = json.dumps(tags) if tags else None

            cursor.execute("""
                INSERT INTO summaries (
                    email_id, sender, sender_key, subject, summary,
                    action_items, priority, tags, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                email_id, sender, sender_key, subject, summary,
                action_items_json, priority, tags_json, timestamp
            ))


            conn.commit()
            conn.close()
            print(f"✅ Saved summary for: {subject[:50]}")
            return True

        except sqlite3.IntegrityError:
            print(f"⚠️  Email already processed: {email_id}")
            return False
        except Exception as e:
            print(f"❌ Error saving summary: {e}")
            return False

    def mark_email_processed(
        self,
        email_id: str,
        message_id: str,
        sender: str,
        subject: str,
        status: str = "completed"
    ) -> bool:
        """
        Mark an email as processed to avoid reprocessing.

        Args:
            email_id (str):    Gmail message ID
            message_id (str):  RFC Message-ID header
            sender (str):      Sender's email
            subject (str):     Email subject
            status (str):      Processing status (completed, failed, pending)

        Returns:
            bool: True if marked successfully
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR IGNORE INTO processed_emails (
                    email_id, message_id, sender, subject, status
                )
                VALUES (?, ?, ?, ?, ?)
            """, (email_id, message_id, sender, subject, status))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"❌ Error marking email processed: {e}")
            return False

    # ─────────────────────────────────────────────────
    #  QUERY OPERATIONS
    # ─────────────────────────────────────────────────

    def is_email_processed(self, email_id: str) -> bool:
        """
        Check if an email has already been processed.

        Args:
            email_id (str): Gmail message ID

        Returns:
            bool: True if already processed, False otherwise
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM processed_emails WHERE email_id = ?", (email_id,))
            result = cursor.fetchone()
            conn.close()

            return result is not None
        except Exception as e:
            print(f"❌ Error checking if email processed: {e}")
            return False

    def get_summary_by_id(self, email_id: str) -> Optional[Dict]:
        """
        Retrieve a summary by email ID.

        Args:
            email_id (str): Gmail message ID

        Returns:
            dict: Summary record or None if not found
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, email_id, sender, sender_key, subject, summary,
                       action_items, priority, timestamp, processed_at, tags
                FROM summaries
                WHERE email_id = ?
            """, (email_id,))



            row = cursor.fetchone()
            conn.close()

            if row:
                return self._row_to_dict(row)
            return None

        except Exception as e:
            print(f"❌ Error retrieving summary: {e}")
            return None

    def get_summaries_by_priority(self, priority: str) -> List[Dict]:
        """
        Retrieve all summaries of a specific priority.

        Args:
            priority (str): Priority level (low, medium, high, urgent)

        Returns:
            list[dict]: List of summaries matching the priority
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, email_id, sender, sender_key, subject, summary,
                       action_items, priority, timestamp, processed_at, tags
                FROM summaries
                WHERE priority = ?

                ORDER BY processed_at DESC
            """, (priority,))

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_dict(row) for row in rows]

        except Exception as e:
            print(f"❌ Error retrieving summaries by priority: {e}")
            return []

    def get_summaries_by_sender_key(self, sender_key: str) -> List[Dict]:

        """
        Retrieve all summaries from a specific sender key.

        Args:
            sender_key (str): Normalized sender email key (lowercase email address)


        Returns:
            list[dict]: List of summaries from that sender
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, email_id, sender, sender_key, subject, summary,
                       action_items, priority, timestamp, processed_at, tags
                FROM summaries
                WHERE sender_key = ?
                ORDER BY processed_at DESC
            """, (sender_key,))


            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_dict(row) for row in rows]

        except Exception as e:
            print(f"❌ Error retrieving summaries by sender: {e}")
            return []

    def get_summaries_with_action_items(self) -> List[Dict]:
        """
        Retrieve all summaries that have action items.

        Returns:
            list[dict]: List of summaries with non-empty action items
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, email_id, sender, subject, summary,
                       action_items, priority, timestamp, processed_at, tags
                FROM summaries
                WHERE action_items IS NOT NULL AND action_items != '[]'
                ORDER BY priority DESC, processed_at DESC
            """)

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_dict(row) for row in rows]

        except Exception as e:
            print(f"❌ Error retrieving summaries with action items: {e}")
            return []

    def get_recent_summaries(self, limit: Optional[int] = 20) -> List[Dict]:
        """
        Retrieve the most recent summaries.

        Args:
            limit (int | None): Maximum number of records to return. Pass None for all records.

        Returns:
            list[dict]: Recent summaries ordered by timestamp (newest first)
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            if limit is None:
                cursor.execute("""
                    SELECT id, email_id, sender, subject, summary,
                           action_items, priority, timestamp, processed_at, tags
                    FROM summaries
                    ORDER BY processed_at DESC
                """)
            else:
                cursor.execute("""
                    SELECT id, email_id, sender, subject, summary,
                           action_items, priority, timestamp, processed_at, tags
                    FROM summaries
                    ORDER BY processed_at DESC
                    LIMIT ?
                """, (limit,))

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_dict(row) for row in rows]

        except Exception as e:
            print(f" Error retrieving recent summaries: {e}")
            return []

    def search_summaries(self, keyword: str) -> List[Dict]:
        """
        Search summaries by keyword in subject or summary text.

        Args:
            keyword (str): Search term

        Returns:
            list[dict]: Matching summaries
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            search_term = f"%{keyword}%"
            cursor.execute("""
                SELECT id, email_id, sender, subject, summary,
                       action_items, priority, timestamp, processed_at, tags
                FROM summaries
                WHERE subject LIKE ? OR summary LIKE ?
                ORDER BY processed_at DESC
            """, (search_term, search_term))

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_dict(row) for row in rows]

        except Exception as e:
            print(f"❌ Error searching summaries: {e}")
            return []

    def get_stats(self) -> Dict:
        """Backwards-compatible alias for stats/analytics."""
        return self.get_sender_analytics(top_n=10)

    def get_sender_analytics(self, top_n: int = 10) -> Dict:


        """
        Get database statistics.

        Returns:
            dict: Statistics including total summaries, by priority, by sender, etc.
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Total summaries
            cursor.execute("SELECT COUNT(*) as count FROM summaries")
            total = cursor.fetchone()["count"]

            # By priority
            cursor.execute("""
                SELECT priority, COUNT(*) as count
                FROM summaries
                GROUP BY priority
            """)
            by_priority = {row["priority"]: row["count"] for row in cursor.fetchall()}

            # With action items
            cursor.execute("""
                SELECT COUNT(*) as count FROM summaries
                WHERE action_items IS NOT NULL AND action_items != '[]'
            """)
            with_actions = cursor.fetchone()["count"]

            # Top senders (by normalized sender_key)
            cursor.execute("""
                SELECT sender_key, COUNT(*) as count
                FROM summaries
                GROUP BY sender_key
                ORDER BY count DESC
                LIMIT ?
            """, (top_n,))
            top_senders = {row["sender_key"]: row["count"] for row in cursor.fetchall()}

            conn.close()

            return {
                "total_summaries": total,
                "by_priority": by_priority,
                "with_action_items": with_actions,
                "top_senders_by_sender_key": top_senders,
            }


        except Exception as e:
            print(f" Error getting stats: {e}")
            return {}

    # ─────────────────────────────────────────────────
    #  EXPORT OPERATIONS
    # ─────────────────────────────────────────────────

    def export_to_json(self, output_file: Optional[str] = None) -> str:
        """
        Export all summaries to a JSON file.

        Args:
            output_file (str, optional): Output file path. Auto-generates if not provided.

        Returns:
            str: Path to the exported file
        """
        try:
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"summaries_export_{timestamp}.json"

            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, email_id, sender, subject, summary,
                       action_items, priority, timestamp, processed_at, tags
                FROM summaries
                ORDER BY processed_at DESC
            """)
            rows = cursor.fetchall()
            conn.close()

            summaries = [self._row_to_dict(row) for row in rows]

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(summaries, f, indent=2, ensure_ascii=False, default=str)

            print(f"✅ Exported {len(summaries)} summaries → {output_file}")
            return output_file

        except Exception as e:
            print(f"❌ Error exporting to JSON: {e}")
            return ""

    def export_to_csv(self, output_file: Optional[str] = None) -> str:
        """
        Export all summaries to a CSV file.

        Args:
            output_file (str, optional): Output file path. Auto-generates if not provided.

        Returns:
            str: Path to the exported file
        """
        try:
            import csv

            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"summaries_export_{timestamp}.csv"

            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT email_id, sender, subject, summary,
                       action_items, priority, timestamp, processed_at, tags
                FROM summaries
                ORDER BY processed_at DESC
            """)
            rows = cursor.fetchall()
            conn.close()

            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Email ID", "Sender", "Subject", "Summary",
                    "Action Items", "Priority", "Original Timestamp", "Processed At", "Tags"
                ])

                for row in rows:
                    writer.writerow([
                        row["email_id"],
                        row["sender"],
                        row["subject"],
                        row["summary"],
                        row["action_items"],
                        row["priority"],
                        row["timestamp"],
                        row["processed_at"],
                        row["tags"],
                    ])

            print(f" Exported {len(rows)} summaries → {output_file}")
            return output_file

        except Exception as e:
            print(f" Error exporting to CSV: {e}")
            return ""

    #  MAINTENANCE OPERATIONS
 

    def backup_database(self) -> str:
        """
        Create a timestamped backup of the database.

        Returns:
            str: Path to the backup file
        """
        try:
            Path(DB_BACKUP_DIR).mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(DB_BACKUP_DIR, f"summaries_backup_{timestamp}.db")

            conn = self.get_connection()
            backup_conn = sqlite3.connect(backup_file)
            conn.backup(backup_conn)
            backup_conn.close()
            conn.close()

            print(f" Database backed up → {backup_file}")
            return backup_file

        except Exception as e:
            print(f" Error backing up database: {e}")
            return ""

    def delete_old_summaries(self, days: int = 90) -> int:
        """
        Delete summaries older than a specified number of days.

        Args:
            days (int): Age threshold in days

        Returns:
            int: Number of deleted records
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM summaries
                WHERE processed_at < datetime('now', '-' || ? || ' days')
            """, (days,))

            deleted = cursor.rowcount
            conn.commit()
            conn.close()

            print(f" Deleted {deleted} summaries older than {days} days")
            return deleted

        except Exception as e:
            print(f" Error deleting old summaries: {e}")
            return 0

    def clear_all_data(self) -> bool:
        """
        Clear all summaries and processed emails (use with caution!).

        Returns:
            bool: True if cleared successfully
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM summaries")
            cursor.execute("DELETE FROM processed_emails")
            conn.commit()
            conn.close()

            print("  All data cleared from database")
            return True

        except Exception as e:
            print(f" Error clearing data: {e}")
            return False

   
    #  HELPER METHODS

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict:
        """
        Convert a sqlite3.Row to a dictionary, parsing JSON fields.

        Args:
            row (sqlite3.Row): Database row

        Returns:
            dict: Row data as dictionary
        """
        data = dict(row)

        # Parse JSON fields
        if data.get("action_items"):
            try:
                data["action_items"] = json.loads(data["action_items"])
            except (json.JSONDecodeError, TypeError):
                data["action_items"] = []

        if data.get("tags"):
            try:
                data["tags"] = json.loads(data["tags"])
            except (json.JSONDecodeError, TypeError):
                data["tags"] = []

        return data

#  DEMONSTRATION & TESTING

def demo():
    """
    Demonstrate storage module functionality.
    """
    print("=" * 60)
    print("  STORAGE MODULE — Demo & Testing")
    print("=" * 60 + "\n")

    # Initialize storage
    storage = StorageManager()

    # Example 1: Save a sample summary
    print("1️  Saving sample summaries...")
    storage.save_summary(
        email_id="msg_001",
        sender="boss@company.com",
        sender_key="boss@company.com",
        subject="Q4 Budget Review - Urgent Response Needed",

        summary="Need to review and approve the Q4 budget by end of week. Key concerns: marketing spend is 20% over estimate.",
        action_items=["Review budget spreadsheet", "Approve or revise line items", "Send feedback to finance"],
        priority="high",
        tags=["finance", "quarterly", "budget"],
        timestamp="2026-06-22 10:00:00"
    )

    storage.save_summary(
        email_id="msg_002",
        sender="client@example.com",
        sender_key="client@example.com",
        subject="Project Status Update",

        summary="Project is on track. Phase 1 delivered on schedule. Waiting for approval to proceed to Phase 2.",
        action_items=["Schedule Phase 2 kickoff meeting"],
        priority="medium",
        tags=["project", "client"],
        timestamp="2026-06-22 09:15:00"
    )

    storage.save_summary(
        email_id="msg_003",
        sender="team@company.com",
        sender_key="team@company.com",
        subject="Meeting Notes: Weekly Standup",

        summary="Team completed sprint tasks. Next sprint starts Monday with new feature requests.",
        action_items=None,
        priority="low",
        tags=["standup", "internal"],
        timestamp="2026-06-21 16:30:00"
    )

    # Example 2: Check if email is already processed
    print("\n Checking if emails are already processed...")
    is_processed = storage.is_email_processed("msg_001")
    print(f"   Is msg_001 processed? {is_processed}")

    # Example 3: Retrieve summaries by priority
    print("\n  Retrieving high-priority summaries...")
    high_priority = storage.get_summaries_by_priority("high")
    for summary in high_priority:
        print(f"   • {summary['subject']} (from {summary['sender']})")

    # Example 4: Get summaries with action items
    print("\n  Getting summaries with action items...")
    action_items = storage.get_summaries_with_action_items()
    for summary in action_items:
        print(f"   • {summary['subject']}")
        print(f"     Actions: {summary['action_items']}")

    # Example 5: Search summaries
    print("\n  Searching for 'budget'...")
    results = storage.search_summaries("budget")
    for result in results:
        print(f"   • {result['subject']}")

    # Example 6: Get statistics
    print("\n  Database statistics...")
    stats = storage.get_stats()
    print(f"   Total summaries: {stats.get('total_summaries')}")
    print(f"   By priority: {stats.get('by_priority')}")
    print(f"   With action items: {stats.get('with_action_items')}")
    print(f"   Top senders: {stats.get('top_senders')}")

    # Example 7: Export data
    print("\n Exporting to JSON...")
    storage.export_to_json()

    print("\n Demo completed!")


if __name__ == "__main__":
    demo()