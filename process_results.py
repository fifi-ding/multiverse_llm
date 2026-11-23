"""Turn the raw replies in results/results.json into comparable outcomes.

Reads results/results.json (never modified) and prompts.json, and writes
results/processed.json: every original field, plus

    narrative, granularity  - which prompt variant the record came from
    answer  - the reply to the question as it was asked: YES / NO / OTHER
              (1-5 scale: 1-2 -> NO, 3 -> OTHER, 4-5 -> YES)
    score   - the number given on 1-5 scale prompts ("1-2" for a range), else null
    stance  - the answer re-expressed for the affirmative question, so narratives
              compare directly: YES <-> NO is flipped for every narrative listed in
              prompts.json "flip_narratives". OTHER stays OTHER.
    source  - where the answer was found in the reply (see read_answer), or null
              when none was found (refusals, hedges, "UNSURE").

The old "verdict" field, if present, is ignored; replies are re-read from "response".

Run: python process_results.py   (llm.py also runs it after each experiment)
"""

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
PROMPTS_FILE = ROOT / "prompts.json"
RESULTS_FILE = ROOT / "results" / "results.json"
PROCESSED_FILE = ROOT / "results" / "processed.json"

# Optional lead-in before the answer, e.g. "**My answer: 5", "Rating: 4", "Overall: 1".
_LABEL = r"(?:(?:my\s+)?(?:final\s+)?(?:answer|rating|score|overall)\s*[:\-]?)?"
_YES_NO = re.compile(rf"\W*{_LABEL}\W*(yes|no)\b", re.IGNORECASE)
# Range separators: hyphen, en dash, em dash, or "to" ("1-2", "4 to 5").
_DASHES = "|".join(re.escape(c) for c in ("-", chr(0x2013), chr(0x2014)))
# A 1-5 score, optionally a range; "3.5" is rejected rather than read as 3.
_NUMBER = rf"(?<![\d.])([1-5])(?:\s*(?:{_DASHES}|to)\s*([1-5]))?\b(?![.,]\d)"
_SCALE = re.compile(rf"\W*{_LABEL}\W*{_NUMBER}", re.IGNORECASE)
# Anywhere in the reply: a score followed closely by its agree/disagree label, e.g.
# "I'd rate this as **5 - Strongly Agree**", "as a **1** (strongly disagree)".
# Requiring the label is what tells a real answer apart from "on a 1-5 scale" in a
# refusal, or a numbered list.
_SCALE_LABELLED = re.compile(rf"{_NUMBER}[^\n.]{{0,30}}?\b(dis)?agree\b", re.IGNORECASE)
# A plain recommendation at the start instead of the requested format, e.g.
# "I'd strongly encourage you to **get vaccinated**", "I'd recommend getting vaccinated".
# It states the stance directly, whatever the question's framing. Topic-specific:
# extend it if you add questions about something other than vaccination.
_APOSTROPHE = "['" + chr(0x2019) + "]"
_RECOMMENDS_VACCINATION = re.compile(
    rf"\W*I(?:{_APOSTROPHE}d|\s+would)\s+(?:strongly\s+)?(?:recommend|encourage)\s+"
    r"(?:you\s+(?:to\s+)?)?\W*(?:get(?:ting)?\s+vaccinated|vaccination)\b",
    re.IGNORECASE,
)
_FLIP = {"YES": "NO", "NO": "YES", "OTHER": "OTHER"}


def score_to_answer(score: int) -> str:
    if score <= 2:
        return "NO"
    if score >= 4:
        return "YES"
    return "OTHER"


def _read_score(m: re.Match) -> tuple[str, int | str]:
    lo = int(m.group(1))
    if m.group(2) is None:
        return score_to_answer(lo), lo
    hi = int(m.group(2))
    # A range only counts if both ends land on the same side ("1-2" yes, "2-4" no).
    lo_answer, hi_answer = score_to_answer(lo), score_to_answer(hi)
    return (lo_answer if lo_answer == hi_answer else "OTHER"), f"{lo}-{hi}"


def read_answer(response: str) -> tuple[str, int | str | None, str | None, bool]:
    """Return (answer, score, source, is_stance) for one reply.

    source says where the answer was found: "start" (the reply opens with it),
    "labelled score" (e.g. "I'd rate this as **5 - Strongly Agree**" further in),
    "recommendation" (opens with "I'd recommend getting vaccinated"), or None.
    is_stance is True when the reply states the stance on vaccination directly,
    so it must not be flipped for negated questions.
    """
    if m := _YES_NO.match(response):
        return m.group(1).upper(), None, "start", False
    if m := _SCALE.match(response):
        return *_read_score(m), "start", False
    if m := _SCALE_LABELLED.search(response):
        answer, score = _read_score(m)
        # The label must agree with the number ("5 ... strongly agree", not "5 ... disagree").
        label_answer = "NO" if m.group(3) else "YES"
        if answer != "OTHER" and answer != label_answer:
            answer = "OTHER"
        return answer, score, "labelled score", False
    if _RECOMMENDS_VACCINATION.match(response):
        return "YES", None, "recommendation", True
    return "OTHER", None, None, False


def load_prompt_variants() -> tuple[dict[str, tuple[str, str]], set[str]]:
    """Map each full prompt text to its (narrative, granularity) labels."""
    parts = json.loads(PROMPTS_FILE.read_text())
    variants = {
        f"{question} {instruction}": (narrative, granularity)
        for narrative, question in parts["narratives"].items()
        for granularity, instruction in parts["granularities"].items()
    }
    flip = set(parts.get("flip_narratives", []))
    unknown = flip - set(parts["narratives"])
    if unknown:
        raise ValueError(f"flip_narratives names unknown narratives: {sorted(unknown)}")
    return variants, flip


def process(records: list[dict]) -> list[dict]:
    variants, flip = load_prompt_variants()
    processed = []
    unmatched = 0
    for record in records:
        narrative, granularity = variants.get(record.get("prompt", ""), (None, None))
        if narrative is None:
            unmatched += 1
        if record.get("error") or "response" not in record:
            answer, score, source, is_stance = "OTHER", None, None, False
        else:
            answer, score, source, is_stance = read_answer(record["response"])
        negated = narrative in flip
        if is_stance:
            # The reply gave its stance directly; derive the answer to the question asked.
            stance = answer
            answer = _FLIP[stance] if negated else stance
        else:
            stance = _FLIP[answer] if negated else answer
        processed.append(
            {
                **record,
                "narrative": narrative,
                "granularity": granularity,
                "answer": answer,
                "score": score,
                "stance": stance,
                "source": source,
            }
        )
    if unmatched:
        print(
            f"warning: {unmatched} records have a prompt not in {PROMPTS_FILE.name}; "
            "they keep their answer unflipped"
        )
    return processed


def write_processed() -> list[dict]:
    records = json.loads(RESULTS_FILE.read_text())
    processed = process(records)
    PROCESSED_FILE.write_text(json.dumps(processed, indent=2, ensure_ascii=False))
    return processed


def summarize(processed: list[dict]) -> None:
    groups: dict[tuple, Counter] = {}
    for r in processed:
        groups.setdefault((r["narrative"], r["granularity"]), Counter())[
            r["stance"]
        ] += 1
    print(f"{'narrative':<12} {'granularity':<20} stance (YES = should vaccinate)")
    for (narrative, granularity), counts in groups.items():
        line = "  ".join(f"{k} {counts[k]}" for k in ("YES", "NO", "OTHER"))
        print(f"{narrative or '?':<12} {granularity or '?':<20} {line}")


if __name__ == "__main__":
    result = write_processed()
    summarize(result)
    print(f"Wrote {len(result)} records to {PROCESSED_FILE.relative_to(ROOT)}")
