#!/usr/bin/env python
"""Collect Jensen Huang speech transcripts from YouTube (captions via yt-dlp).

Output: data/raw/youtube/<video_id>.json with {id,title,uploader,upload_date,duration,url,text,source_type}
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from common import RAW, in_window, load_sources, normalize_ws, window

YT_OUT = RAW / "youtube"
YTDLP = [sys.executable, "-m", "yt_dlp"]


def run(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _flat(target: str, n: int) -> list[dict]:
    cmd = YTDLP + ["--flat-playlist", "--dump-single-json", "--no-warnings", "--playlist-end", str(n), target]
    p = run(cmd, timeout=240)
    if p.returncode != 0 or not p.stdout.strip():
        return []
    try:
        return json.loads(p.stdout).get("entries") or []
    except json.JSONDecodeError:
        return []


def search(query: str, n: int) -> list[dict]:
    """Return flat metadata for a query: YouTube 'sort by upload date' results + relevance results."""
    from urllib.parse import quote_plus

    out: list[dict] = []
    try:
        # sp=CAI%3D == "Sort by: Upload date"
        out += _flat(f"https://www.youtube.com/results?search_query={quote_plus(query)}&sp=CAI%253D", n)
        out += _flat(f"ytsearch{n}:{query}", n)
    except Exception as e:  # noqa: BLE001
        print(f"  [search] error for {query!r}: {e}")
    return out


def fetch_meta(video_id: str) -> dict | None:
    p = run(YTDLP + ["--dump-single-json", "--no-warnings", "--skip-download", f"https://www.youtube.com/watch?v={video_id}"], timeout=180)
    if p.returncode != 0:
        return None
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return None


def json3_to_text(path: Path) -> str:
    data = json.loads(path.read_text())
    parts: list[str] = []
    for ev in data.get("events", []):
        segs = ev.get("segs") or []
        line = "".join(s.get("utf8", "") for s in segs)
        if line.strip():
            parts.append(line.replace("\n", " "))
    text = " ".join(parts)
    text = re.sub(r"\[(music|applause|laughter|inaudible)[^\]]*\]", " ", text, flags=re.I)
    return normalize_ws(re.sub(r"\s+", " ", text))


def fetch_subs(video_id: str) -> str | None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        cmd = YTDLP + [
            "--skip-download", "--write-subs", "--write-auto-subs",
            "--sub-langs", "en,en-orig,en.*", "--sub-format", "json3",
            "--no-warnings", "-o", str(out / "%(id)s.%(ext)s"),
            f"https://www.youtube.com/watch?v={video_id}",
        ]
        p = run(cmd, timeout=300)
        files = sorted(out.glob(f"{video_id}*.json3"))
        if not files:
            print(f"  [subs] none for {video_id}: {p.stderr.strip()[-200:]}")
            return None
        # Prefer manual English subs over auto ("en-orig" is the auto track).
        files.sort(key=lambda f: (("orig" in f.name), len(f.name)))
        return json3_to_text(files[0])


def save(meta: dict, text: str) -> None:
    vid = meta["id"]
    up = meta.get("upload_date")
    rec = {
        "id": f"yt_{vid}",
        "source_type": "youtube",
        "url": f"https://www.youtube.com/watch?v={vid}",
        "title": meta.get("title"),
        "uploader": meta.get("uploader") or meta.get("channel"),
        "date": datetime.strptime(up, "%Y%m%d").date().isoformat() if up else None,
        "duration_sec": meta.get("duration"),
        "text": text,
    }
    (YT_OUT / f"{vid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))


def main() -> None:
    cfg = load_sources()["youtube"]
    start, end = window()
    must = [m.lower() for m in cfg["must_match_any"]]
    candidates: dict[str, dict] = {}

    for vid in cfg.get("video_ids", []):
        candidates[vid] = {"id": vid, "forced": True}

    for q in cfg["search_queries"]:
        print(f"[search] {q}")
        for e in search(q, cfg["results_per_query"]):
            vid = e.get("id")
            if not vid or vid in candidates:
                continue
            title = (e.get("title") or "").lower()
            uploader = (e.get("uploader") or e.get("channel") or "").lower()
            if not any(m in title or m in uploader for m in must):
                continue
            dur = e.get("duration") or 0
            if dur and dur < cfg["min_duration_sec"]:
                continue
            candidates[vid] = e

    print(f"{len(candidates)} candidate videos")
    kept = 0
    for vid, e in candidates.items():
        out = YT_OUT / f"{vid}.json"
        if out.exists():
            kept += 1
            continue
        meta = fetch_meta(vid)
        if not meta:
            print(f"  [meta] failed {vid}")
            continue
        up = meta.get("upload_date")
        d = datetime.strptime(up, "%Y%m%d").date() if up else None
        if not e.get("forced") and not in_window(d, start, end):
            continue
        if (meta.get("duration") or 0) < cfg["min_duration_sec"] and not e.get("forced"):
            continue
        text = fetch_subs(vid)
        if not text or len(text.split()) < 300:
            continue
        save(meta, text)
        kept += 1
        print(f"  + {meta.get('title')!r} [{d}] {len(text.split())} words")
    print(f"saved {kept} transcripts in {YT_OUT}")


if __name__ == "__main__":
    main()
