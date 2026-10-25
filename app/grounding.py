"""Question-grounding gate: does the retrieved evidence actually cover what the question names?

Fused retrieval scores are not a usable confidence signal (reciprocal-rank fusion gives ~0.04 for every top hit), so
coverage is checked lexically instead: the named events, places, periods and products in the question (``GTC
Washington``, ``Q3 FY2027``, ``Vera Rubin NVL72`` …) must appear in the retrieved passages. Anything missing is
(1) injected into the prompt as an explicit instruction not to substitute material about another event, (2) shown
to the reader as a coverage note, and (3) handed to the verifier so misattributed sentences are corrected.
"""

from __future__ import annotations

import re

_STOP = {
    "What", "Who", "When", "Where", "Why", "How", "Which", "Did", "Does", "Do", "Is", "Are", "Was", "Were", "Has", "Have",
    "Explain", "Describe", "Summarize", "Summarise", "Tell", "Give", "List", "Compare", "Please", "Can", "Could", "Would",
    "Should", "Will", "In", "On", "At", "The", "A", "An", "And", "Or", "Of", "For", "To", "About", "From", "With", "By",
    "Jensen", "Huang", "Jensen's", "NVIDIA", "Nvidia", "NVIDIA's", "Nvidia's", "AI", "I", "GPU", "GPUs", "CEO", "Mr",
    "Me", "Us", "We", "You", "It", "Its", "His", "Their", "This", "That", "These", "Those", "There", "Here", "Also",
}
# Named events: "GTC Washington", "GTC Paris 2026", "Computex 2026", "CES 2027", "SIGGRAPH 2026"
EVENT_RE = re.compile(r"\b(GTC|Computex|COMPUTEX|CES|SIGGRAPH|Supercomputing|SC\d{2})\b((?:\s+(?:[A-Z][A-Za-z.]+|D\.?C\.?)){0,2}(?:\s+\d{4})?)")
# Fiscal/calendar periods: "Q2 FY2027", "Q3 of FY27", "Q1 2026"
PERIOD_RE = re.compile(r"\b(Q[1-4])\s*(?:of\s*)?(FY\s?\d{2,4}|\d{4})\b", re.I)
CAP_RE = re.compile(r"\b([A-Z][A-Za-z0-9.'\-]*(?:\s+[A-Z][A-Za-z0-9.'\-]*){0,3})")
_EVENT_WORDS = {"gtc", "computex", "ces", "siggraph", "supercomputing"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


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
        if not any(_norm(t) == _norm(o) for o in out):
            out.append(t)
    # drop a term that is a strict sub-phrase of another kept term (e.g. "Washington" inside "GTC Washington")
    return [t for t in out if not any(t != o and _norm(t) in _norm(o) for o in out)][:6]


def _event_covered(term: str, chunks: list[str]) -> bool:
    n = _norm(term)
    parts = term.split()
    kw, rest = parts[0].lower(), [p for p in parts[1:] if not p.isdigit()]
    year = next((p for p in parts[1:] if p.isdigit()), None)
    for c in chunks:
        if n in c:
            return True
        # same chunk mentions the event keyword and the place (and the year, if one was asked for)
        if kw in c and all(_norm(r) in c for r in rest) and (year is None or year in c):
            if rest or year:
                return True
    return False


def coverage(question: str, hits: list[dict]) -> dict:
    """{"terms": [...], "missing": [...]} — ``missing`` are the named things no retrieved passage mentions."""
    terms = key_terms(question or "")
    chunks = [_norm(f"{h.get('title') or ''} {h.get('text') or ''}") for h in hits]
    blob = " ".join(chunks)
    missing = []
    for t in terms:
        first = t.split()[0].lower()
        if first in _EVENT_WORDS or PERIOD_RE.fullmatch(t):
            ok = _event_covered(t, chunks) if first in _EVENT_WORDS else _norm(t) in blob or _norm(t.replace("FY20", "FY")) in blob
        else:
            ok = all(_norm(w) in blob for w in t.split())
        if not ok:
            missing.append(t)
    return {"terms": terms, "missing": missing}


def prompt_instruction(cov: dict, corpus_span: str) -> str:
    if not cov.get("missing"):
        return ""
    what = ", ".join(f"“{m}”" for m in cov["missing"])
    return (f"COVERAGE WARNING: none of the retrieved sources mention {what}. The source corpus covers {corpus_span}. "
            f"Do NOT present material about a different event, place, period or product as if it were about {what}. "
            f"Begin the Summary with one plain sentence stating that the available sources do not cover {what}. "
            f"Then, only if genuinely related material exists, say explicitly which event/date it comes from "
            f"(e.g. “At GTC Berlin on 20 Oct 2026 …”) and keep every statement attributed to its real event. "
            f"Never use {what} as the title or root of a diagram or use-case map; name the real event or topic instead. "
            f"If nothing related exists, say so and keep the answer to the Summary section.")


def reader_note(cov: dict, corpus_span: str) -> str:
    if not cov.get("missing"):
        return ""
    what = ", ".join(f"“{m}”" for m in cov["missing"])
    return (f"Coverage note: the sources in this corpus ({corpus_span}) do not mention {what}. "
            f"Anything below comes from the closest covered material and is attributed to its actual event.")


_DISCLAIM_RE = re.compile(r"(do(es)? not|don't|doesn't|no)\s+(source|excerpt|transcript|coverage|cover|mention|material)", re.I)


def ensure_disclaimer(text: str, cov: dict, corpus_span: str) -> str:
    """If the model ignored the warning, prepend the coverage note so the reader is never misled."""
    if not cov.get("missing"):
        return text
    for miss in cov["missing"]:  # an uncovered event must not be the root/title of a diagram
        text = re.sub(r"root\(\((?:[^()]*\b" + re.escape(miss) + r"\b[^()]*)\)\)", "root((Closest covered material))", text, flags=re.I)
        text = re.sub(r"(Figure:[^\n]*?)\b" + re.escape(miss) + r"\b", r"\1the closest covered material", text, flags=re.I)
    head = text[:600]
    if _DISCLAIM_RE.search(head):
        return text
    note = reader_note(cov, corpus_span)
    m = re.match(r"\s*(### Summary\s*\n+)", text)
    if m:
        return text[: m.end()] + f"**{note}**\n\n" + text[m.end():]
    return f"**{note}**\n\n{text}"
