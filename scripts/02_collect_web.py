#!/usr/bin/env python
"""Collect web articles / transcripts / press releases about Jensen Huang and NVIDIA.

Sources (configs/sources.yaml): seed URLs, index pages (1-level crawl), RSS feeds.
Extraction: trafilatura (HTML) / pypdf (PDF). Output: data/raw/web/<sha1>.json
"""

from __future__ import annotations

import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from urllib.parse import parse_qs, urljoin, urlparse

import feedparser
import httpx
import trafilatura
from bs4 import BeautifulSoup
from pypdf import PdfReader

from common import RAW, in_window, load_sources, normalize_ws, parse_date, sha1, window

WEB_OUT = RAW / "web"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
HEADERS = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}
SKIP_HOSTS = {"youtube.com", "www.youtube.com", "twitter.com", "x.com", "facebook.com", "linkedin.com"}


def get(url: str, timeout: int) -> httpx.Response | None:
    try:
        with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=timeout) as c:
            r = c.get(url)
            if r.status_code == 200:
                return r
            print(f"  [{r.status_code}] {url}")
    except Exception as e:  # noqa: BLE001
        print(f"  [err] {url}: {type(e).__name__}")
    return None


def extract_pdf(content: bytes) -> tuple[str, str | None]:
    reader = PdfReader(io.BytesIO(content))
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    return normalize_ws(text), None


def extract_html(html: str, url: str) -> tuple[str | None, str | None, str | None]:
    """Returns (text, title, date_str)."""
    j = trafilatura.extract(html, url=url, output_format="json", with_metadata=True, include_comments=False, favor_recall=True)
    if not j:
        return None, None, None
    d = json.loads(j)
    return d.get("text"), d.get("title"), d.get("date")


def fetch_article(url: str, timeout: int) -> dict | None:
    r = get(url, timeout)
    if r is None:
        return None
    ctype = r.headers.get("content-type", "")
    if url.lower().endswith(".pdf") or "application/pdf" in ctype:
        text, title = extract_pdf(r.content)
        d = parse_date(url) or parse_date(text[:2000])
    else:
        text, title, dstr = extract_html(r.text, url)
        d = parse_date(dstr) or parse_date(text[:1500] if text else "")
        if not title:
            soup = BeautifulSoup(r.text, "lxml")
            title = soup.title.string.strip() if soup.title and soup.title.string else None
    if not text or len(text.split()) < 120:
        return None
    return {
        "id": f"web_{sha1(url)}",
        "source_type": "web",
        "url": str(r.url),
        "title": title,
        "domain": urlparse(str(r.url)).netloc,
        "date": d.isoformat() if isinstance(d, date) else None,
        "text": normalize_ws(text),
    }


def crawl_index(url: str, link_regex: str, timeout: int) -> list[str]:
    r = get(url, timeout)
    if r is None:
        return []
    soup = BeautifulSoup(r.text, "lxml")
    rx = re.compile(link_regex)
    links = set()
    for a in soup.find_all("a", href=True):
        full = urljoin(url, a["href"]).split("#")[0]
        if rx.search(full):
            links.add(full)
    return sorted(links)


def rss_links(feed_url: str, start: date, end: date) -> list[str]:
    try:
        fp = feedparser.parse(feed_url, request_headers=HEADERS)
    except Exception as e:  # noqa: BLE001
        print(f"  [rss err] {feed_url}: {e}")
        return []
    out = []
    for e in fp.entries:
        link = e.get("link")
        if not link:
            continue
        # Bing News wraps links in a tracking redirect that rejects bots; unwrap it.
        if "bing.com/news/apiclick" in link:
            qs = parse_qs(urlparse(link).query)
            link = (qs.get("url") or [None])[0]
            if not link:
                continue
        pub = e.get("published") or e.get("updated")
        d = parse_date(pub)
        if in_window(d, start, end):
            out.append(link)
    return out


def main() -> None:
    cfg = load_sources()["web"]
    start, end = window()
    timeout = cfg.get("request_timeout", 25)
    urls: dict[str, str] = {}

    for u in cfg["seed_urls"]:
        urls[u] = "seed"
    for idx in cfg.get("index_pages", []):
        found = crawl_index(idx["url"], idx["link_regex"], timeout)
        print(f"[index] {idx['url']} -> {len(found)} links")
        for u in found:
            urls.setdefault(u, "index")
    for feed in cfg.get("rss_feeds", []):
        found = rss_links(feed, start, end)
        print(f"[rss] {feed[:80]} -> {len(found)} links")
        for u in found:
            urls.setdefault(u, "rss")

    urls = {u: k for u, k in urls.items() if urlparse(u).netloc not in SKIP_HOSTS}
    todo = [u for u in urls if not (WEB_OUT / f"{sha1(u)}.json").exists()][: cfg["max_pages"]]
    print(f"{len(urls)} unique urls, {len(todo)} to fetch")

    must = cfg["must_mention_any"]
    saved = skipped = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_article, u, timeout): u for u in todo}
        for fut in as_completed(futs):
            u = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  [fail] {u}: {e}")
                continue
            if not rec:
                skipped += 1
                continue
            if not any(m in rec["text"] for m in must):
                skipped += 1
                continue
            d = date.fromisoformat(rec["date"]) if rec["date"] else None
            # Non-seed pages must be in the time window when a date is known.
            if urls[u] != "seed" and not in_window(d, start, end):
                skipped += 1
                continue
            rec["discovered_via"] = urls[u]
            (WEB_OUT / f"{sha1(u)}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
            saved += 1
            time.sleep(0.05)
    print(f"saved {saved}, skipped {skipped}; total files: {len(list(WEB_OUT.glob('*.json')))}")


if __name__ == "__main__":
    main()
