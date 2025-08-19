# routine.py
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from notion_client import Client
from dotenv import load_dotenv

load_dotenv()  # safe no-op if .env doesn't exist

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
TASKS_DB_ID  = os.getenv("TASKS_DB_ID")
DAILY_DB_ID  = os.getenv("DAILY_DB_ID")
WEEKLY_DB_ID = os.getenv("WEEKLY_DB_ID")
TZ           = os.getenv("TIMEZONE", "America/Los_Angeles")
WEEK_START   = int(os.getenv("WEEK_START", "1"))  # Monday=1 ... Sunday=7

if not all([NOTION_TOKEN, TASKS_DB_ID, DAILY_DB_ID, WEEKLY_DB_ID]):
    raise SystemExit("Missing one of NOTION_TOKEN, TASKS_DB_ID, DAILY_DB_ID, WEEKLY_DB_ID.")

notion = Client(auth=NOTION_TOKEN)

# ----------------- Helpers -----------------
def _sleep_backoff(i):
    # basic exponential-ish backoff for rate limits
    time.sleep(0.2 + 0.1 * i)

def query_all(database_id: str, **kwargs):
    """Query entire database with pagination."""
    results = []
    start_cursor = None
    page = 0
    while True:
        page += 1
        resp = notion.databases.query(
            **({"database_id": database_id, "start_cursor": start_cursor} | kwargs)
        )
        results.extend(resp.get("results", []))
        if not resp.get("has_more"):
            break
        start_cursor = resp.get("next_cursor")
        _sleep_backoff(page)
    return results

def get_checkbox(page: dict, prop: str) -> bool:
    p = page["properties"].get(prop)
    if not p:
        return False
    if p["type"] == "checkbox":
        return bool(p["checkbox"])
    if p["type"] == "formula" and p["formula"]["type"] == "boolean":
        return bool(p["formula"]["boolean"])
    return False

def today_iso() -> str:
    return datetime.now(ZoneInfo(TZ)).date().isoformat()

def week_bounds(dt=None):
    """Return (start_date, end_date) as dates in local TZ. WEEK_START: Monday=1 ... Sunday=7."""
    if dt is None:
        dt = datetime.now(ZoneInfo(TZ)).date()
    # Python weekday: Monday=0 ... Sunday=6
    py_weekday = dt.weekday()  # 0..6
    desired = WEEK_START - 1   # 0..6
    delta = (py_weekday - desired) % 7
    start = dt - timedelta(days=delta)
    end   = start + timedelta(days=6)
    return start, end

def upsert_daily_summary(date_iso: str, completed_ids: list[str], total: int):
    # Check if already exists
    existing = notion.databases.query(
        database_id=DAILY_DB_ID,
        filter={"property": "Date", "date": {"equals": date_iso}},
    )
    if existing.get("results"):
        return existing["results"][0]["id"]

    percent = (len(completed_ids) / total) if total else 0.0
    res = notion.pages.create(
        parent={"database_id": DAILY_DB_ID},
        properties={
            "Date": {"date": {"start": date_iso}},
            "Completed Tasks": {"relation": [{"id": pid} for pid in completed_ids]},
            "Completed Count": {"number": len(completed_ids)},
            "Total Tasks": {"number": total},
            # If your DB uses a Formula property for Percent, you can omit this:
            "Percent": {"number": percent},
        },
    )
    return res["id"]

def upsert_weekly_summary(week_start_iso: str, week_end_iso: str, completed_ids: list[str], total: int):
    existing = notion.databases.query(
        database_id=WEEKLY_DB_ID,
        filter={"property": "Week Start", "date": {"equals": week_start_iso}},
    )
    if existing.get("results"):
        return existing["results"][0]["id"]

    percent = (len(completed_ids) / total) if total else 0.0
    res = notion.pages.create(
        parent={"database_id": WEEKLY_DB_ID},
        properties={
            "Week Start": {"date": {"start": week_start_iso}},
            "Week End": {"date": {"start": week_end_iso}},
            "Completed Tasks": {"relation": [{"id": pid} for pid in completed_ids]},
            "Completed Count": {"number": len(completed_ids)},
            "Total Tasks": {"number": total},
            "Percent": {"number": percent},
        },
    )
    return res["id"]

def reset_checkboxes(pages: list[dict], prop_name="Completed"):
    # Update only ones that are true to reduce API calls
    for i, p in enumerate(pages):
        if get_checkbox(p, prop_name):
            notion.pages.update(
                page_id=p["id"],
                properties={prop_name: {"checkbox": False}},
            )
            _sleep_backoff(i)

# ----------------- Jobs -----------------
def run_daily():
    date_iso = today_iso()

    # Get all Daily tasks
    tasks = query_all(
        TASKS_DB_ID,
        filter={"property": "Cadence", "select": {"equals": "Daily"}},
    )

    # Effective completion: Completed or Auto Done (if present)
    completed = [
        p for p in tasks
        if get_checkbox(p, "Completed") or get_checkbox(p, "Auto Done")
    ]

    upsert_daily_summary(date_iso, [p["id"] for p in completed], len(tasks))

    # Reset
    reset_checkboxes(tasks)

    print(f"[daily] {date_iso} logged {len(completed)}/{len(tasks)} and reset.")

def run_weekly():
    ws, we = week_bounds()
    week_start_iso = ws.isoformat()
    week_end_iso   = we.isoformat()

    tasks = query_all(
        TASKS_DB_ID,
        filter={"property": "Cadence", "select": {"equals": "Weekly"}},
    )

    completed = [
        p for p in tasks
        if get_checkbox(p, "Completed") or get_checkbox(p, "Auto Done")
    ]

    upsert_weekly_summary(week_start_iso, week_end_iso, [p["id"] for p in completed], len(tasks))

    # Reset
    reset_checkboxes(tasks)

    print(f"[weekly] {week_start_iso}–{week_end_iso} logged {len(completed)}/{len(tasks)} and reset.")

# ----------------- CLI -----------------
if __name__ == "__main__":
    import sys
    mode = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
    if mode == "daily":
        run_daily()
    elif mode == "weekly":
        run_weekly()
    else:
        run_daily()
        run_weekly()
