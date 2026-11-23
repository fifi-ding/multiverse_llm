import json
from datetime import UTC, datetime
from pathlib import Path

import anthropic

from build_universes import write_universes
from process_results import summarize, write_processed

ROOT = Path(__file__).parent
UNIVERSES_FILE = ROOT / "universes.json"
PROFILES_FILE = ROOT / "profiles.json"
RESULTS_FILE = ROOT / "results" / "results.json"
RUNS_PER_UNIVERSE = 3  # keep in sync with DEFAULT_RUNS in visualize.html

client = anthropic.Anthropic()


PROFILE_LABELS = {"highest_edu": "Highest education"}


def pick_profile(universe: dict, profiles: list[dict]) -> dict | None:
    """Return the profile a universe asks for.

    No "profile" key -> the first profile; "profile": null -> no profile;
    "profile": "<name>" -> the profile with that name.
    """
    if "profile" not in universe:
        return profiles[0] if profiles else None
    name = universe["profile"]
    if name is None:
        return None
    for profile in profiles:
        if profile.get("name") == name:
            return profile
    raise ValueError(f"universe {universe['id']}: no profile named {name!r}")


def build_message(prompt: str, profile: dict | None) -> str:
    """Prepend the user's profile, written as the user describing themselves."""
    if profile is None:
        return prompt
    lines = [
        f"- {PROFILE_LABELS.get(key, key.replace('_', ' ').capitalize())}: {value}"
        for key, value in profile.items()
    ]
    return "A bit about me:\n" + "\n".join(lines) + "\n\n" + prompt


def run_universe(universe: dict, run: int, profile: dict | None) -> dict:
    """Send one universe's prompt to its model and return a result record."""
    sent_prompt = build_message(universe["prompt"], profile)
    record = {
        "id": universe["id"],
        "run": run,
        "model": universe["model"],
        "prompt": universe["prompt"],
        "profile": profile.get("name") if profile else None,
        "sent_prompt": sent_prompt,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    try:
        message = client.messages.create(
            model=universe["model"],
            max_tokens=16000,
            messages=[{"role": "user", "content": sent_prompt}],
        )
    except anthropic.APIError as e:
        return {**record, "error": str(e)}

    text = "".join(block.text for block in message.content if block.type == "text")
    return {
        **record,
        "served_model": message.model,
        "stop_reason": message.stop_reason,
        "response": text,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
    }


def main() -> None:
    # Rebuild from prompts.json / profiles.json / models.json so edits there always apply.
    universes = write_universes()
    print(f"Built {len(universes)} universes x {RUNS_PER_UNIVERSE} runs")
    profiles = json.loads(PROFILES_FILE.read_text())
    # Resolve every profile up front so a typo fails before any paid API call.
    chosen = {u["id"]: pick_profile(u, profiles) for u in universes}
    results = []
    for universe in universes:
        for run in range(1, RUNS_PER_UNIVERSE + 1):
            print(f"[{universe['id']}.{run}] {universe['model']} ...")
            result = run_universe(universe, run, chosen[universe["id"]])
            results.append(result)
            print(result.get("response") or f"ERROR: {result.get('error')}")
            print("-" * 60)

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved {len(results)} results to {RESULTS_FILE}")
    # Interpret the raw replies (YES/NO, 1-5 scale, flipped narratives) for the page.
    summarize(write_processed())


if __name__ == "__main__":
    main()
