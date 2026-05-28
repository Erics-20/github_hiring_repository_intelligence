"""
github_collector.py
-------------------
Stage 1 – GitHub REST API data collection with stratified sampling.

Buckets
  0  Low-value / Templates   : 0-2 stars, or "template" in name/description
  1  Intern / Junior         : homework / bootcamp / assignment keywords
  2  Senior / Lead           : 50-500 stars, recently active

Output
  data/raw/bucket_<n>_<label>.jsonl  – one JSON object per line
  data/raw/collected.csv             – merged flat file for quick inspection
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterator

import requests
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    sys.exit("[FATAL] GITHUB_TOKEN is not set. Add it to your .env file.")

BASE_URL = "https://api.github.com"
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

# GitHub Search API caps: 1 000 results per query, 30 items/page max (we use 100)
PER_PAGE = 100
TARGET_PER_BUCKET = 150       # repos we want per bucket
MAX_PAGES_PER_QUERY = 10      # safety ceiling per sub-query
SLEEP_BETWEEN_PAGES = 1.2     # seconds – well within 10 req/min search limit
SLEEP_ON_RATE_LIMIT = 62      # seconds to sleep when 403/429 received

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bucket definitions
# ---------------------------------------------------------------------------

# Each bucket is a list of query strings.  We cycle through them until we
# reach TARGET_PER_BUCKET unique repos for that bucket.

def _recent(days: int = 30) -> str:
    """Return a GitHub date-range qualifier for the last `days` days."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    return f">={cutoff}"

BUCKETS: dict[int, dict] = {
    0: {
        "label": "low_value_template",
        "queries": [
            f"stars:0..2 pushed:{_recent(30)} size:>0",
            f"stars:0..2 created:{_recent(60)} size:>0",
            'template in:name stars:0..5',
            'boilerplate in:name stars:0..5',
            'starter-kit in:name stars:0..3',
        ],
    },
    1: {
        "label": "intern_junior",
        "queries": [
            "homework in:name,description size:>0",
            "bootcamp in:name,description size:>0",
            "assignment in:name,description size:>0",
            "ejercicio in:name,description size:>0",          # Spanish equivalent
            "tarea in:name,description size:>0",
            "practica in:name,description size:>0",
            "first-project in:name size:>0 stars:0..5",
        ],
    },
    2: {
        "label": "senior_lead",
        "queries": [
            f"stars:50..500 pushed:{_recent(14)} size:>100",
            f"stars:100..500 pushed:{_recent(30)} size:>500",
            f"stars:50..200 pushed:{_recent(7)} language:Python",
            f"stars:50..200 pushed:{_recent(7)} language:TypeScript",
            f"stars:50..300 pushed:{_recent(14)} language:Go",
        ],
    },
}

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract(repo: dict) -> dict:
    """Flatten a GitHub Search API repository object into a clean dict."""
    license_name = None
    if repo.get("license") and repo["license"]:
        license_name = repo["license"].get("name")

    return {
        "full_name":        repo.get("full_name"),
        "description":      repo.get("description"),
        "html_url":         repo.get("html_url"),
        "created_at":       repo.get("created_at"),
        "updated_at":       repo.get("updated_at"),
        "pushed_at":        repo.get("pushed_at"),
        "size":             repo.get("size", 0),
        "stargazers_count": repo.get("stargazers_count", 0),
        "forks_count":      repo.get("forks_count", 0),
        "open_issues_count":repo.get("open_issues_count", 0),
        "language":         repo.get("language"),
        "has_wiki":         repo.get("has_wiki", False),
        "has_pages":        repo.get("has_pages", False),
        "license":          license_name,
        "topics":           repo.get("topics", []),
        "is_fork":          repo.get("fork", False),
        "archived":         repo.get("archived", False),
        "visibility":       repo.get("visibility", "public"),
    }

# ---------------------------------------------------------------------------
# GitHub Search pagination
# ---------------------------------------------------------------------------

def _search_pages(query: str) -> Iterator[list[dict]]:
    """
    Yield pages of raw repository dicts from the GitHub Search API.
    Handles 403 / 429 rate-limit responses with an exponential back-off.
    """
    page = 1
    while page <= MAX_PAGES_PER_QUERY:
        params = {"q": query, "per_page": PER_PAGE, "page": page, "sort": "updated"}
        retries = 0
        while retries < 4:
            try:
                resp = requests.get(
                    f"{BASE_URL}/search/repositories",
                    headers=HEADERS,
                    params=params,
                    timeout=20,
                )
            except requests.RequestException as exc:
                log.warning("Network error (%s) – retrying in 5 s", exc)
                time.sleep(5)
                retries += 1
                continue

            if resp.status_code == 200:
                break
            if resp.status_code in (403, 429):
                # Respect Retry-After if present, otherwise use default sleep
                retry_after = int(resp.headers.get("Retry-After", SLEEP_ON_RATE_LIMIT))
                log.warning(
                    "Rate-limited (HTTP %s). Sleeping %d s…",
                    resp.status_code, retry_after,
                )
                time.sleep(retry_after)
                retries += 1
                continue
            if resp.status_code == 422:
                # Unprocessable entity – query too complex or page > 10
                log.debug("HTTP 422 for query '%s' page %d – stopping.", query, page)
                return
            log.error("Unexpected HTTP %s for query '%s'", resp.status_code, query)
            return

        else:
            log.error("Exhausted retries for query '%s' page %d", query, page)
            return

        data = resp.json()
        items = data.get("items", [])
        total = data.get("total_count", 0)
        log.info("  page %2d | total_count=%6d | items=%d", page, total, len(items))

        if not items:
            break

        yield items

        # GitHub Search only exposes the first 1 000 results (10 pages × 100)
        if page * PER_PAGE >= min(total, 1000):
            break

        page += 1
        time.sleep(SLEEP_BETWEEN_PAGES)

        # Check secondary rate-limit headers
        remaining = int(resp.headers.get("X-RateLimit-Remaining", 10))
        if remaining < 3:
            reset_ts = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(0, reset_ts - int(time.time())) + 2
            log.warning("Search rate-limit almost exhausted. Sleeping %d s…", wait)
            time.sleep(wait)

# ---------------------------------------------------------------------------
# Bucket collection
# ---------------------------------------------------------------------------

def collect_bucket(bucket_id: int) -> list[dict]:
    bucket = BUCKETS[bucket_id]
    label  = bucket["label"]
    seen   = set()          # deduplicate by full_name across queries
    rows   = []

    log.info("=" * 60)
    log.info("BUCKET %d  |  %s  |  target=%d", bucket_id, label, TARGET_PER_BUCKET)
    log.info("=" * 60)

    for query in bucket["queries"]:
        if len(rows) >= TARGET_PER_BUCKET:
            break
        log.info("Query: %s", query)

        for page_items in _search_pages(query):
            for raw in page_items:
                name = raw.get("full_name")
                if name in seen:
                    continue
                seen.add(name)
                record = _extract(raw)
                record["bucket_id"] = bucket_id
                record["bucket_label"] = label
                rows.append(record)

            if len(rows) >= TARGET_PER_BUCKET:
                break

        log.info("  Collected so far: %d / %d", len(rows), TARGET_PER_BUCKET)

    log.info("Bucket %d done – %d repos collected.", bucket_id, len(rows))
    return rows[:TARGET_PER_BUCKET]

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_bucket(rows: list[dict], bucket_id: int, label: str) -> Path:
    path = RAW_DIR / f"bucket_{bucket_id}_{label}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
    log.info("Saved %d rows → %s", len(rows), path)
    return path


def merge_and_save_csv(all_rows: list[dict]) -> Path:
    df = pd.DataFrame(all_rows)
    # Explode list-typed topics into a pipe-separated string for CSV compatibility
    df["topics"] = df["topics"].apply(
        lambda t: "|".join(t) if isinstance(t, list) else ""
    )
    path = RAW_DIR / "collected.csv"
    df.to_csv(path, index=False)
    log.info("Merged CSV saved → %s  (%d rows, %d cols)", path, len(df), len(df.columns))
    return path

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    all_rows: list[dict] = []

    for bucket_id in sorted(BUCKETS.keys()):
        rows = collect_bucket(bucket_id)
        label = BUCKETS[bucket_id]["label"]
        save_bucket(rows, bucket_id, label)
        all_rows.extend(rows)

    if all_rows:
        merge_and_save_csv(all_rows)
        log.info("-" * 60)
        log.info("Collection complete.  Total repos: %d", len(all_rows))

        # Quick distribution summary
        df = pd.DataFrame(all_rows)
        summary = df.groupby("bucket_label").size().reset_index(name="count")
        log.info("\n%s", summary.to_string(index=False))
    else:
        log.error("No repos collected. Check your GITHUB_TOKEN and network.")


if __name__ == "__main__":
    main()
