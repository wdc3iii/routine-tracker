# routine.py
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from notion_client import Client
from dotenv import load_dotenv
from typing import Dict, List, Tuple
from functools import lru_cache
from notion_client.errors import APIResponseError

load_dotenv()  # safe no-op if .env doesn't exist

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
DAILY_DB_ID  = os.getenv("DAILY_DB_ID")
WEEKLY_DB_ID = os.getenv("WEEKLY_DB_ID")
DAILY_DB_NAME = os.getenv("DAILY_DB_NAME", "Daily Checklist Completion")
WEEKLY_DB_NAME = os.getenv("WEEKLY_DB_NAME", "Weekly Checklist Completion")
ROUTINE_PAGE_ID = os.getenv("ROUTINE_PAGE_ID")
DAILY_SECTION_NAME = os.getenv("DAILY_SECTION_NAME", "Daily Checklist")
WEEKLY_SECTION_NAME = os.getenv("WEEKLY_SECTION_NAME", "Weekly Checklist")
TZ           = os.getenv("TIMEZONE", "America/Los_Angeles")
WEEK_START   = int(os.getenv("WEEK_START", "1"))  # Monday=1 ... Sunday=7
DESCEND_TYPES = {
    "toggle", "callout",
    "column_list", "column",
    "bulleted_list_item", "numbered_list_item",
    "to_do",
    "heading_1", "heading_2", "heading_3",
}
if not all([NOTION_TOKEN, DAILY_DB_ID, WEEKLY_DB_ID, DAILY_DB_NAME, ROUTINE_PAGE_ID]):
    raise SystemExit("Missing NOTION_TOKEN, DAILY_DB_ID, WEEKLY_DB_ID, DAILY_DB_NAME, or ROUTINE_PAGE_ID.")

notion = Client(auth=NOTION_TOKEN)

# ----------------- Helpers -----------------
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

def yesterday_iso():
    return (datetime.now(ZoneInfo(TZ)).date() - timedelta(days=1)).isoformat()


# ---- Block helpers ----
@lru_cache(maxsize=1024)
def list_children(block_id: str) -> List[dict]:
    results, start_cursor = [], None
    while True:
        resp = notion.blocks.children.list(block_id=block_id, start_cursor=start_cursor)
        results.extend(resp.get("results", []))
        if not resp.get("has_more"):
            break
        start_cursor = resp.get("next_cursor")
    return results


# ---- Fast children cache & selective descent ---
def gather_todos_selective(root_block_id: str, max_depth: int = 1) -> list[dict]:
    todos, stack = [], [(root_block_id, 0)]
    while stack:
        bid, depth = stack.pop()
        for child in list_children(bid):
            t = child.get("type")
            if t == "to_do":
                todos.append(child)
            if child.get("has_children") and t in DESCEND_TYPES and depth < max_depth:
                stack.append((child["id"], depth + 1))
    return todos

def plain_text_from_richtext(rich: List[dict]) -> str:
    return "".join([t.get("plain_text", "") for t in rich or []]).strip()

def block_title_text(b: dict) -> str:
    t = b.get("type")
    if t in ("heading_1", "heading_2", "heading_3", "toggle", "callout", "paragraph"):
        return plain_text_from_richtext(b[t].get("rich_text", []))
    if t == "child_database":
        return b["child_database"].get("title", "") or ""
    if t == "child_page":
        return b["child_page"].get("title", "") or ""
    return ""

def status_map_from_todos(todos: List[dict]) -> Dict[str, bool]:
    """
    Map {label: checked}. If duplicate labels exist, last one wins.
    """
    m: Dict[str, bool] = {}
    for b in todos:
        text = plain_text_from_richtext(b["to_do"].get("rich_text", []))
        checked = bool(b["to_do"].get("checked", False))
        if text:
            m[text] = checked
    return m

def block_matches_name(b: dict, name: str) -> bool:
    return block_title_text(b).strip().lower() == name.strip().lower()

def parent_id_of(block: dict) -> str | None:
    p = block.get("parent", {})
    # parent can be page_id, block_id, database_id, etc.; we just need block/page id key
    return p.get("block_id") or p.get("page_id")

def find_block_anywhere(page_id: str, section_name: str) -> dict | None:
    want = section_name.strip().lower()
    stack = list_children(page_id)[::-1]
    while stack:
        b = stack.pop()
        if block_title_text(b).strip().lower() == want:
            return b
        if b.get("has_children") and b.get("type") in DESCEND_TYPES:
            stack.extend(list_children(b["id"])[::-1])
    return None

def list_siblings(parent_id: str) -> list[dict]:
    """Return all direct children of a parent block/page (siblings under the same parent)."""
    return list_children(parent_id)

SECTION_CONTAINERS = {
    "toggle", "callout",
    "column_list", "column",
}

def todos_for_section_fast(page_id: str, section_name: str, stop_names: list[str]) -> list[dict]:
    """
    Anchor at the section title anywhere on the page.
    If that title is inside a container (e.g., callout), scan *within that container*
    starting right after the title block. Otherwise, scan siblings at the page/block level.
    Also include selective descent into likely containers for nested to_dos.
    """
    start_block = find_block_anywhere(page_id, section_name)
    if not start_block:
        return []

    # Decide the anchor parent to scan within:
    anchor_parent_id = parent_id_of(start_block)  # container or page
    anchor_type = None
    if anchor_parent_id:
        # Peek at the parent block if it exists (pages won’t show up as a block here)
        parents_children = list_children(anchor_parent_id)
        # Find index of our start block among its siblings in that parent
        try:
            start_idx = next(i for i, b in enumerate(parents_children) if b["id"] == start_block["id"])
        except StopIteration:
            start_idx = None
    else:
        parents_children, start_idx = [], None

    todos: list[dict] = []

    # 1) If the title block itself has children, include nested to_dos under it.
    if start_block.get("has_children"):
        todos.extend(gather_todos_selective(start_block["id"], max_depth=3))

    # 2) Scan *subsequent siblings in the same parent* starting right after the title.
    if parents_children and start_idx is not None:
        for j in range(start_idx + 1, len(parents_children)):
            sib = parents_children[j]
            # stop if we hit a new section heading/toggle/callout that matches a stop name
            title = block_title_text(sib).strip()
            if title and title in stop_names:
                break

            t = sib.get("type")
            if t == "to_do":
                todos.append(sib)

            # selective descent for nested containers under siblings
            if sib.get("has_children") and t in DESCEND_TYPES:
                todos.extend(gather_todos_selective(sib["id"], max_depth=3))

    return todos


def reset_todos(todos: List[dict]):
    for b in todos:
        if not b["to_do"].get("checked"):
            continue
        tries, delay = 0, 0.02
        while True:
            try:
                notion.blocks.update(block_id=b["id"], to_do={"checked": False})
                time.sleep(0.02)  # tiny courtesy pause; raise if you see 429s
                break
            except APIResponseError as e:
                if e.code == "rate_limited" and tries < 5:
                    time.sleep(delay); delay *= 2; tries += 1
                    continue
                else:
                    raise


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

# ----------------- Notion DB Schema Helpers -----------------
_schema_cache: dict[str, dict] = {}

def retrieve_database_schema(database_id: str) -> dict:
    if database_id in _schema_cache:
        return _schema_cache[database_id]
    schema = notion.databases.retrieve(database_id=database_id)
    _schema_cache[database_id] = schema
    return schema

def ensure_checkbox_columns_for_tasks(database_id: str, task_names: list[str]):
    if not task_names:
        return
    schema = retrieve_database_schema(database_id)
    existing = set(schema.get("properties", {}).keys())
    to_add = [n for n in task_names if n and n not in existing]
    if to_add:
        notion.databases.update(database_id=database_id,
                                properties={n: {"checkbox": {}} for n in to_add})
        _schema_cache.pop(database_id, None)  # invalidate


def get_existing_daily_checkbox_columns(database_id: str) -> list[str]:
    schema = retrieve_database_schema(database_id)
    cols = []
    for name, meta in schema.get("properties", {}).items():
        if meta.get("type") == "checkbox":
            cols.append(name)
    return cols

def get_existing_weekly_checkbox_columns(database_id: str) -> list[str]:
    schema = retrieve_database_schema(database_id)
    cols = []
    for name, meta in schema.get("properties", {}).items():
        if meta.get("type") == "checkbox":
            cols.append(name)
    return cols

def get_title_prop_name(database_id: str) -> str | None:
    schema = retrieve_database_schema(database_id)
    for name, meta in schema.get("properties", {}).items():
        if meta.get("type") == "title":
            return name
    return None

def ensure_daily_base_props():
    schema = retrieve_database_schema(DAILY_DB_ID)
    props = schema.get("properties", {})
    patch = {"properties": {}}
    if "Date" not in props:
        patch["properties"]["Date"] = {"date": {}}
    if patch["properties"]:
        notion.databases.update(database_id=DAILY_DB_ID, **patch)

def ensure_weekly_base_props():
    schema = retrieve_database_schema(WEEKLY_DB_ID)
    props = schema.get("properties", {})
    patch = {"properties": {}}
    if "Week Start" not in props:
        patch["properties"]["Week Start"] = {"date": {}}
    if "Week End" not in props:
        patch["properties"]["Week End"] = {"date": {}}
    if patch["properties"]:
        notion.databases.update(database_id=WEEKLY_DB_ID, **patch)


def upsert_daily_summary_wide(date_iso: str, task_status: dict[str, bool]):
    ensure_daily_base_props()
    ensure_checkbox_columns_for_tasks(DAILY_DB_ID, list(task_status.keys()))

    page_props = {"Date": {"date": {"start": date_iso}}}
    for name, done in task_status.items():
        page_props[name] = {"checkbox": bool(done)}

    existing = notion.databases.query(
        database_id=DAILY_DB_ID,
        filter={"property": "Date", "date": {"equals": date_iso}},
    )

    title_prop = get_title_prop_name(DAILY_DB_ID)
    if title_prop:
        page_props[title_prop] = {"title": [{"type": "text", "text": {"content": date_iso}}]}

    if existing.get("results"):
        notion.pages.update(page_id=existing["results"][0]["id"], properties=page_props)
    else:
        notion.pages.create(parent={"database_id": DAILY_DB_ID}, properties=page_props)


def upsert_weekly_summary_wide(week_start_iso: str, week_end_iso: str, task_status: dict[str, bool]):
    ensure_weekly_base_props()
    ensure_checkbox_columns_for_tasks(WEEKLY_DB_ID, list(task_status.keys()))

    page_props = {
        "Week Start": {"date": {"start": week_start_iso}},
        "Week End":   {"date": {"start": week_end_iso}},
    }
    for name, done in task_status.items():
        page_props[name] = {"checkbox": bool(done)}

    existing = notion.databases.query(
        database_id=WEEKLY_DB_ID,  # ✅ weekly DB here
        filter={"property": "Week Start", "date": {"equals": week_start_iso}},
    )

    title_prop = get_title_prop_name(WEEKLY_DB_ID)
    if title_prop:
        page_props[title_prop] = {
            "title": [{"type": "text", "text": {"content": f"{week_start_iso} – {week_end_iso}"}}]
        }

    if existing.get("results"):
        notion.pages.update(page_id=existing["results"][0]["id"], properties=page_props)
    else:
        notion.pages.create(parent={"database_id": WEEKLY_DB_ID}, properties=page_props)



# ----------------- Jobs -----------------
def run_daily():
    import time
    t0 = time.perf_counter_ns()
    date_iso = yesterday_iso()
    t1 = time.perf_counter_ns()
    print(f"ISO Date Time: {((t1 - t0) / 1_000_000):.2f} ms")

    daily_todos = todos_for_section_fast(
        ROUTINE_PAGE_ID,
        DAILY_SECTION_NAME,
        stop_names=[WEEKLY_SECTION_NAME]  # stop when we hit the Weekly section
    )
    t2 = time.perf_counter_ns()
    print(f"Getting Todos Time: {((t2 - t1) / 1_000_000):.2f} ms")

    task_status = status_map_from_todos(daily_todos)
    t3 = time.perf_counter_ns()
    print(f"Mapping Time: {((t3 - t2) / 1_000_000):.2f} ms")

    # ensure_daily_base_props()
    t4 = time.perf_counter_ns()
    print(f"Ensure Props Time: {((t4 - t3) / 1_000_000):.2f} ms")

    upsert_daily_summary_wide(date_iso, task_status)
    t5 = time.perf_counter_ns()
    print(f"Upsert Daily Time: {((t5 - t4) / 1_000_000):.2f} ms")

    reset_todos(daily_todos)
    t6 = time.perf_counter_ns()
    print(f"Reset Time: {((t6 - t5) / 1_000_000):.2f} ms")

    print(f"[daily] {date_iso} tasks={len(task_status)} reset={sum(1 for t in daily_todos if t['to_do'].get('checked'))}")
    print(f"Total Time: {((t6 - t0) / 1_000_000):.2f} ms")


def run_weekly():
    local_today = datetime.now(ZoneInfo(TZ)).date()
    ws, we = week_bounds(local_today - timedelta(days=1))
    week_start_iso, week_end_iso = ws.isoformat(), we.isoformat()

    weekly_todos = todos_for_section_fast(
        ROUTINE_PAGE_ID,
        WEEKLY_SECTION_NAME,
        stop_names=[DAILY_DB_NAME, WEEKLY_DB_NAME]  # TODO: Stop at the database
    )
    task_status = status_map_from_todos(weekly_todos)

    ensure_weekly_base_props()
    upsert_weekly_summary_wide(week_start_iso, week_end_iso, task_status)
    reset_todos(weekly_todos)
    print(f"[weekly] {week_start_iso}–{week_end_iso} tasks={len(task_status)} reset={sum(1 for t in weekly_todos if t['to_do'].get('checked'))}")


# ----------------- CLI -----------------
if __name__ == "__main__":
    import sys
    mode = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
    if mode == "daily":
        run_daily()
    elif mode == "weekly":
        run_weekly()
