"""Agentic claim verification: check every cited paragraph of a briefing against the sources it cites.

The verifier is the second step of the "precise" workflow (retrieve → draft → verify → correct). For each prose
paragraph that carries citations, the judge model sees the paragraph and the full text of the cited passages and
returns, per sentence, one of: supported | corrected (with a rewrite that only states what the sources say) |
remove (no support at all). Corrections are applied in place, and a verification report is returned.

Exposed as ``POST /api/verify`` so that the NeMo Agent Toolkit workflow (``nat/``) can call it as a tool, and used
inline by the app when ``verify=true`` (default for expert mode).
"""

from __future__ import annotations

import json
import re

CITE_RE = re.compile(r"\[(\d{1,2})\]")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'“(*])")

JUDGE_PROMPT = """You are a strict fact-checker for an analyst briefing about NVIDIA. For each numbered claim below you are given the claim text and the full text of the sources it cites. Decide per sentence whether the sources support it.

Rules:
- "supported": every fact, number, name and date in the sentence appears in (or follows directly from) the cited sources. Paraphrase is fine.
- "corrected": the sentence is partly right; give a rewritten sentence that keeps only what the sources say (keep its citation marks, keep it in the same style). Do not add new facts.
- "remove": the sources do not support the sentence at all, or it contradicts them.
- Sentences ending with "(inspiration)" are explicitly labelled extrapolations — mark them "supported".
- Be precise but not pedantic: rounding (96.2B vs 96B) and obvious paraphrase are supported.

Return ONLY a JSON object: {"claims": [{"id": <claim id>, "sentences": [{"i": <sentence index>, "verdict": "supported|corrected|remove", "text": "<rewritten sentence, only when corrected>"}]}]}

"""


def _paragraphs(text: str) -> list[tuple[int, int, str]]:
    """(start, end, paragraph) for prose paragraphs outside fences/headings."""
    out, pos, fence = [], 0, False
    for block in re.split(r"(\n{2,})", text):
        if block.startswith("```") or "```" in block:
            fence = not fence if block.count("```") % 2 else fence
        if not fence and block.strip() and not block.lstrip().startswith(("#", "```", "Figure:", "**Figure")) and not block.startswith("\n"):
            out.append((pos, pos + len(block), block))
        pos += len(block)
    return out


def build_judge_input(answer: str, hits: list[dict], max_src_chars: int = 2200) -> tuple[str, list[dict]]:
    claims = []
    stop = re.search(r"^### Use-case inspiration\s*$", answer, re.M)  # labelled extrapolations are not fact-checked
    limit = stop.start() if stop else len(answer)
    for pi, (s, e, para) in enumerate(_paragraphs(answer)):
        if s >= limit:
            break
        ids = sorted({int(n) for n in CITE_RE.findall(para) if 1 <= int(n) <= len(hits)})
        if not ids:
            continue
        sents = _SENT.split(para.strip())
        claims.append({"id": len(claims) + 1, "start": s, "end": e, "para": para, "sentences": sents, "src_ids": ids})
    parts = []
    for c in claims:
        parts.append(f"### Claim {c['id']}")
        for i, snt in enumerate(c["sentences"]):
            parts.append(f"  sentence {i}: {snt}")
        for n in c["src_ids"]:
            h = hits[n - 1]
            parts.append(f"  source [{n}] ({h.get('title', '')[:80]}, {h.get('date', '')}): {h.get('text', '')[:max_src_chars]}")
    return "\n".join(parts), claims


def apply_verdicts(answer: str, claims: list[dict], verdict_json: str) -> tuple[str, dict]:
    try:
        m = re.search(r"\{.*\}", verdict_json, re.S)
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    by_id = {c["id"]: c for c in claims}
    report = {"checked": sum(len(c["sentences"]) for c in claims), "supported": 0, "corrected": 0, "removed": 0, "changes": []}
    edits: list[tuple[int, int, str]] = []
    seen_ids = set()
    for item in data.get("claims", []):
        c = by_id.get(item.get("id"))
        if not c:
            continue
        seen_ids.add(c["id"])
        sents = list(c["sentences"])
        for v in item.get("sentences", []):
            i = v.get("i")
            if not isinstance(i, int) or not 0 <= i < len(sents):
                continue
            verdict = (v.get("verdict") or "supported").lower()
            norm = lambda x: re.sub(r"\s+|\[\d+\]", "", x).lower()  # noqa: E731
            if verdict == "corrected" and (v.get("text") or "").strip() and norm(v["text"]) == norm(sents[i]):
                verdict = "supported"
            if verdict == "corrected" and (v.get("text") or "").strip():
                new = v["text"].strip()
                if not CITE_RE.search(new):
                    cites = "".join(f"[{n}]" for n in CITE_RE.findall(sents[i])[:2])
                    new = f"{new} {cites}".strip()
                report["changes"].append({"claim": c["id"], "from": sents[i], "to": new})
                sents[i] = new
                report["corrected"] += 1
            elif verdict == "remove":
                report["changes"].append({"claim": c["id"], "from": sents[i], "to": None})
                sents[i] = ""
                report["removed"] += 1
            else:
                report["supported"] += 1
        new_para = " ".join(s for s in sents if s).strip()
        if new_para != c["para"].strip():
            lead = re.match(r"^(\*\*[^*\n]{2,60}\*\*\s*)", c["para"].strip())
            if lead and not new_para.startswith("**"):
                new_para = lead.group(1) + new_para
            edits.append((c["start"], c["end"], new_para))
    for c in claims:  # claims the judge skipped count as supported (no evidence against them)
        if c["id"] not in seen_ids:
            report["supported"] += len(c["sentences"])
    for s, e, new in sorted(edits, reverse=True):
        answer = answer[:s] + new + answer[e:]
    answer = re.sub(r"\*\*[^*\n]{2,60}\*\*\s*(?=\n|$)", "", answer)  # lead-ins whose paragraph was fully removed
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    total = max(1, report["checked"])
    report["score"] = round(report["supported"] / total, 3)
    return answer, report
