"""Question-grounding gate: does the retrieved evidence actually cover what the question names?

Fused retrieval scores are not a usable confidence signal (reciprocal-rank fusion gives ~0.04 for every top hit), so
coverage is checked lexically instead: the named events, places, periods and products in the question (``GTC
Washington``, ``Q3 FY2027``, ``Vera Rubin NVL72`` …) must appear in the retrieved passages. Anything missing is
(1) injected into the prompt as an explicit instruction not to substitute material about another event, (2) shown
to the reader as a coverage note, and (3) handed to the verifier so misattributed sentences are corrected.

Matching is word-bounded on lightly normalised text (punctuation → spaces), never on space-stripped blobs: otherwise
"CES" is found inside "services", "Paris" inside "comparison" and "DC" inside "broadcast".
"""

from __future__ import annotations

import re

_STOP = {
    "What", "Who", "When", "Where", "Why", "How", "Which", "Did", "Does", "Do", "Is", "Are", "Was", "Were", "Has", "Have",
    "Explain", "Describe", "Summarize", "Summarise", "Tell", "Give", "List", "Compare", "Please", "Can", "Could", "Would",
    "Should", "Will", "In", "On", "At", "The", "A", "An", "And", "Or", "Of", "For", "To", "About", "From", "With", "By",
    "Jensen", "Huang", "Jensen's", "NVIDIA", "Nvidia", "NVIDIA's", "Nvidia's", "AI", "I", "GPU", "GPUs", "CEO", "Mr",
    "Me", "Us", "We", "You", "It", "Its", "His", "Their", "This", "That", "These", "Those", "There", "Here", "Also",
    "New", "Latest", "Recent", "Today", "Yes", "No", "Project", "Keynote", "Event", "Announcement", "Announcements",
}
# Named events: "GTC Washington", "GTC Paris 2026", "Computex 2026", "CES 2027", "SIGGRAPH 2026", "GTC DC"
EVENT_RE = re.compile(r"\b(GTC|Computex|COMPUTEX|CES|SIGGRAPH|Supercomputing|SC\d{2})\b((?:\s+(?:[A-Z][A-Za-z.]+|D\.?C\.?)){0,2}(?:\s+\d{4})?)")
# Fiscal/calendar periods: "Q2 FY2027", "Q3 of FY27", "Q1 2026"
PERIOD_RE = re.compile(r"\b(Q[1-4])\s*(?:of\s*)?(FY\s?\d{2,4}|\d{4})\b", re.I)
CAP_RE = re.compile(r"\b([A-Z][A-Za-z0-9.'\-]*(?:\s+[A-Z][A-Za-z0-9.'\-]*){0,3})")
_EVENT_WORDS = {"gtc", "computex", "ces", "siggraph", "supercomputing"}
# "GTC DC" / "GTC Washington D.C.": accept "washington" anywhere in the window, or "dc" only directly after "gtc"
# (plain "dc" near "GTC" is usually "800 volts DC").
_PLACE_ALIASES = {"dc": ["washington", "gtc dc", "gtc d c"], "washington": ["washington", "gtc dc", "gtc d c"],
                  "dc.": ["washington", "gtc dc", "gtc d c"], "sanjose": ["san jose"], "taipei": ["taipei", "computex"]}
_ORDINAL = {"q1": "first", "q2": "second", "q3": "third", "q4": "fourth"}


def _norm(s: str) -> str:
    """Lowercase, punctuation → space, single spaces. Keeps word boundaries (unlike stripping all non-alphanumerics)."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s.lower())).strip()


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _pat(term: str) -> re.Pattern:
    """Word-bounded pattern for a term; tokens may be joined by nothing, spaces or hyphens ('NVL72' ~ 'NVL 72')."""
    toks = re.findall(r"[a-z]+|\d+", term.lower())
    return re.compile(r"(?<![a-z0-9])" + r"[\s\-]*".join(map(re.escape, toks)) + r"(?![a-z0-9])")


def key_terms(question: str) -> list[str]:
    """Named things in the question that the evidence must mention, most specific first."""
    terms: list[str] = []
    for m in EVENT_RE.finditer(question):
        t = re.sub(r"\s+", " ", m.group(0)).strip()
        if len(t.split()) > 1:  # bare "GTC" tells us nothing
            terms.append(t)
    for m in PERIOD_RE.finditer(question):
        terms.append(re.sub(r"\s+", " ", m.group(0)))
    for m in CAP_RE.finditer(question):
        words = [w.strip(".,;:?!'\"") for w in m.group(1).split()]
        words = [w for w in words if w and w not in _STOP and w.rstrip("'s") not in _STOP]
        if words and len("".join(words)) >= 4:
            terms.append(" ".join(words))
    out: list[str] = []
    for t in terms:
        if not any(_key(t) == _key(o) for o in out):
            out.append(t)
    # drop a term that is a strict sub-phrase of another kept term (e.g. "Washington" inside "GTC Washington")
    return [t for t in out if not any(t != o and _key(t) in _key(o) for o in out)][:6]


def _event_covered(term: str, hits: list[dict]) -> bool:
    parts = term.split()
    kw = parts[0].lower()
    year = next((p for p in parts[1:] if p.isdigit()), None)
    places = [p for p in parts[1:] if not p.isdigit()]
    alts: list[list[str]] = []
    for p in places:
        alts.append(_PLACE_ALIASES.get(_key(p), [p]))
    for h in hits:
        if _event_in(kw, alts, year, h):
            return True
    return False


def _event_in(kw: str, alts: list[list[str]], year: str | None, h: dict) -> bool:
    """Event keyword and place must be adjacent (at most two words between them): "GTC Washington", "GTC in
    Washington, D.C.", "Washington GTC". A list like "GTC in Berlin; an event in Washington D.C." does not count."""
    text = _norm(f"{h.get('title') or ''} {h.get('text') or ''}")
    if year and year not in text and not str(h.get("date") or "").startswith(year):
        return False  # the question asked about a specific year (CES 2027 ≠ CES 2026)
    if not alts:
        return bool(_pat(kw).search(text))
    gap = r"(?:\s+[a-z0-9]+){0,2}\s+"

    def near(a: str) -> bool:
        toks = re.findall(r"[a-z]+|\d+", a.lower())
        if toks and toks[0] == kw:  # alias already includes the keyword ("gtc dc")
            return bool(_pat(a).search(text))
        place = r"[\s\-]*".join(map(re.escape, toks))
        return bool(re.search(rf"(?<![a-z0-9]){kw}{gap}{place}(?![a-z0-9])|(?<![a-z0-9]){place}{gap}{kw}(?![a-z0-9])", text))

    return all(any(near(a) for a in alt) for alt in alts)


def _period_covered(term: str, hits: list[dict]) -> bool:
    """Q2 FY2027 is covered by 'Q2 FY2027', 'Q2 2027', 'second quarter of fiscal 2027', 'fiscal 2027 … second quarter'."""
    m = PERIOD_RE.fullmatch(term)
    if not m:
        return False
    quarter, year = m.group(1).lower(), m.group(2)[-2:]
    q = rf"(?:{quarter}|{_ORDINAL[quarter]}[\s-]quarter)"
    fy = rf"(?:fy\s?(?:20)?{year}|fiscal(?:\s*year)?\s?(?:20)?{year}|20{year})(?![0-9])"
    pat = re.compile(rf"(?<![a-z0-9]){q}\W{{0,12}}(?:of\s*)?{fy}|{fy}\W{{0,60}}{q}(?![a-z0-9])", re.I)
    return any(pat.search(_norm(f"{h.get('title') or ''} {h.get('text') or ''}")) for h in hits)


def _phrase_in(term: str, text: str) -> bool:
    """Full phrase, or — for multi-word names — the last word plus at least half the words ("Starmer" for
    "Keir Starmer", "Rubin NVL72" for "Vera Rubin NVL72"); never a single generic first word alone."""
    words = term.split()
    if _pat(term).search(text):
        return True
    if len(words) >= 2:
        found = [bool(_pat(w).search(text)) for w in words]
        return found[-1] and sum(found) * 2 >= len(words)
    return False


def mentions(term: str, hit: dict) -> bool:
    """Does one passage mention the term (same rules as coverage)? Used to merge targeted re-query results."""
    first = term.split()[0].lower()
    if first in _EVENT_WORDS or re.match(r"sc\d{2}$", first):
        return _event_covered(term, [hit])
    if PERIOD_RE.fullmatch(term):
        return _period_covered(term, [hit])
    return _phrase_in(term, _norm(f"{hit.get('title') or ''} {hit.get('text') or ''}"))


def coverage(question: str, hits: list[dict]) -> dict:
    """{"terms", "missing", "thin"}: ``missing`` are the named things no retrieved passage mentions; ``thin`` are
    those only one passage mentions, and only in its body (a passing mention, not a source about the thing)."""
    terms = key_terms(question or "")
    missing, thin = [], []
    for t in terms:
        where = [i for i, h in enumerate(hits) if mentions(t, h)]
        if not where:
            missing.append(t)
        elif len(where) == 1 and not mentions(t, {"title": hits[where[0]].get("title") or "", "text": "", "date": hits[where[0]].get("date")}):
            thin.append({"term": t, "source": where[0] + 1})
    return {"terms": terms, "missing": missing, "thin": thin}


def prompt_instruction(cov: dict, corpus_span: str) -> str:
    parts = []
    if cov.get("missing"):
        what = ", ".join(f"“{m}”" for m in cov["missing"])
        parts.append(
            f"COVERAGE WARNING: none of the retrieved sources mention {what}. The source corpus covers {corpus_span}. "
            f"Do NOT present material about a different event, place, period or product as if it were about {what}. "
            f"Begin the Summary with one plain sentence stating that the available sources do not cover {what}. "
            f"Then, only if genuinely related material exists, say explicitly which event/date it comes from "
            f"(e.g. “At GTC Berlin on 20 Oct 2026 …”) and keep every statement attributed to its real event. "
            f"If nothing related exists, say so and keep the answer to the Summary section.")
    for t in cov.get("thin") or []:
        parts.append(
            f"LIMITED COVERAGE: only source [{t['source']}] mentions “{t['term']}”, and only in passing. Say so in the Summary, "
            f"state exactly what that source says about “{t['term']}” (nothing more), and attribute everything else in the "
            f"answer to the event, date or product the other sources actually describe — never to “{t['term']}”.")
    return "\n\n".join(parts)


def reader_note(cov: dict, corpus_span: str) -> str:
    parts = []
    if cov.get("missing"):
        what = ", ".join(f"“{m}”" for m in cov["missing"])
        parts.append(f"Coverage note: the sources in this corpus ({corpus_span}) do not mention {what}. "
                     f"Anything below comes from the closest covered material and is attributed to its actual event.")
    for t in cov.get("thin") or []:
        parts.append(f"Limited coverage: only source [{t['source']}] mentions “{t['term']}”, in passing.")
    return " ".join(parts)


_DISCLAIM_RE = re.compile(r"(do(es)? not|don't|doesn't|no|only (one|a single)?\s*(source|excerpt|passage)?\s*)\s*"
                          r"(source|excerpt|transcript|coverage|cover|mention|material|in passing)", re.I)


def _fix_roots(text: str, missing: list[str]) -> str:
    """A use-case map titled with the uncovered event ("root((GTC Washington))") would re-assert the misattribution
    in the one place the reader looks first; retitle it."""
    for term in missing:
        text = re.sub(r"root\(\(([^)]*?)" + re.escape(term) + r"([^)]*?)\)\)",
                      lambda m: f"root((Related NVIDIA material, {term} not covered))", text, flags=re.I)
    return text


def ensure_disclaimer(text: str, cov: dict, corpus_span: str) -> str:
    """If the model ignored the warning, prepend the coverage note so the reader is never misled."""
    if not cov.get("missing") and not cov.get("thin"):
        return text
    text = _fix_roots(text, (cov.get("missing") or []) + [t["term"] for t in cov.get("thin") or []])
    head = text[:600]
    if _DISCLAIM_RE.search(head):
        return text
    note = reader_note(cov, corpus_span)
    m = re.match(r"\s*(### Summary\s*\n+)", text)
    if m:
        return text[: m.end()] + f"**{note}**\n\n" + text[m.end():]
    return f"**{note}**\n\n{text}"
