"""Generate universes.json from the files you edit by hand.

Inputs (edit these):
    prompts.json   - "narratives": {label: question} and
                     "granularities": {label: answer instruction}.
                     Every narrative is combined with every granularity:
                     prompt = "<question> <instruction>".
    profiles.json  - list of profile objects, each with a unique "name"
    models.json    - list of model ids

Output (don't edit by hand, it's overwritten):
    universes.json - every narrative x granularity x profile x model combination,
                     plus a "no profile" version of each as a baseline.

Run directly with `python build_universes.py`; llm.py also calls it before each run.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent
PROMPTS_FILE = ROOT / "prompts.json"
PROFILES_FILE = ROOT / "profiles.json"
MODELS_FILE = ROOT / "models.json"
UNIVERSES_FILE = ROOT / "universes.json"

# Also send each prompt with no profile, so you can see what the profile changes.
INCLUDE_NO_PROFILE = True


def _load_list(path: Path) -> list:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path.name} must be a non-empty JSON list")
    return data


def _load_labelled(data: dict, key: str) -> dict[str, str]:
    section = data.get(key)
    if not isinstance(section, dict) or not section:
        raise ValueError(
            f'{PROMPTS_FILE.name} needs a non-empty "{key}" object of label: text'
        )
    return section


def _check_unique(values: list, what: str) -> None:
    seen = set()
    for v in values:
        if v in seen:
            raise ValueError(f"duplicate {what}: {v!r}")
        seen.add(v)


def build_universes() -> list[dict]:
    """Return one universe per combination, ordered as the tree shows them."""
    prompt_parts = json.loads(PROMPTS_FILE.read_text())
    narratives = _load_labelled(prompt_parts, "narratives")
    granularities = _load_labelled(prompt_parts, "granularities")
    profiles = _load_list(PROFILES_FILE)
    models = _load_list(MODELS_FILE)

    profile_names = []
    for p in profiles:
        if not p.get("name"):
            raise ValueError(f"every profile in {PROFILES_FILE.name} needs a name: {p}")
        profile_names.append(p["name"])
    _check_unique(profile_names, "profile name")
    _check_unique(models, "model")

    profile_options: list[str | None] = list(profile_names)
    if INCLUDE_NO_PROFILE:
        profile_options.append(None)

    universes = []
    for narrative, question in narratives.items():
        for granularity, instruction in granularities.items():
            for profile in profile_options:
                for model in models:
                    universes.append(
                        {
                            "id": len(universes) + 1,
                            "narrative": narrative,
                            "granularity": granularity,
                            "profile": profile,
                            "model": model,
                            "prompt": f"{question} {instruction}",
                        }
                    )
    return universes


def write_universes() -> list[dict]:
    """Build the universes and save them to universes.json."""
    universes = build_universes()
    UNIVERSES_FILE.write_text(
        json.dumps(universes, indent=4, ensure_ascii=False) + "\n"
    )
    return universes


if __name__ == "__main__":
    written = write_universes()
    print(f"Wrote {len(written)} universes to {UNIVERSES_FILE.name}")
