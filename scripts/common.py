"""Shared helpers for the data pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CURATED = DATA / "curated"
SFT = DATA / "sft"
INDEX = DATA / "index"

for _d in (RAW / "youtube", RAW / "web", CURATED, SFT, INDEX):
    _d.mkdir(parents=True, exist_ok=True)


def load_sources() -> dict[str, Any]:
    with open(ROOT / "configs" / "sources.yaml") as f:
        return yaml.safe_load(f)


def window() -> tuple[date, date]:
    w = load_sources()["window"]
    return date.fromisoformat(w["start"]), date.fromisoformat(w["end"])


def in_window(d: date | None, start: date, end: date) -> bool:
    """Unknown dates are kept (filtered later by content heuristics)."""
    if d is None:
        return True
    return start <= d <= end


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


_DATE_PATTERNS = [
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})"), "%Y-%m-%d"),
    (re.compile(r"(\d{4})/(\d{2})/(\d{2})"), "%Y/%m/%d"),
    (re.compile(r"(\d{8})"), "%Y%m%d"),
]
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
_TEXT_DATE = re.compile(rf"\b({_MONTHS})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b")
_TEXT_DATE2 = re.compile(rf"\b(\d{{1,2}})\s+({_MONTHS})\.?\s+(\d{{4}})\b")


def parse_date(s: str | None) -> date | None:
    if not s:
        return None
    s = s.strip()
    for rx, fmt in _DATE_PATTERNS:
        m = rx.search(s)
        if m:
            try:
                return datetime.strptime(m.group(0), fmt).date()
            except ValueError:
                pass
    m = _TEXT_DATE.search(s)
    if m:
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                return datetime.strptime(f"{m.group(1).rstrip('.')[:3] if fmt.startswith('%b') else m.group(1)} {m.group(2)} {m.group(3)}", fmt).date()
            except ValueError:
                pass
    m = _TEXT_DATE2.search(s)
    if m:
        for fmt in ("%d %B %Y", "%d %b %Y"):
            try:
                mon = m.group(2).rstrip(".")
                mon = mon[:3] if fmt.endswith("%b %Y") else mon
                return datetime.strptime(f"{m.group(1)} {mon} {m.group(3)}", fmt).date()
            except ValueError:
                pass
    try:
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(s).date()
    except Exception:
        return None


def normalize_ws(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default
