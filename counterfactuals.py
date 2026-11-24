"""Counterfactual explanations with DiCE: which choices make every run say YES?

Reads results/processed.json (run `python process_results.py` first) and:

1. Trains a random forest on every run: inputs are the four choices that define a
   universe (narrative, granularity, profile, model), output is stance == YES.
   Runs of one universe share inputs, so the forest learns P(YES) per universe.
2. For each universe with at least one non-YES run (observed, not predicted), asks
   DiCE for changes to those choices that push the forest's P(YES) to at least
   TARGET_P_YES, i.e. changes it expects to make every run YES.
3. Checks each counterfactual against the real runs. The universes cover every
   combination of choices, so the changed universe was actually run. A
   counterfactual "holds" only if all of that universe's runs were YES.

Writes results/counterfactuals.json and prints a summary.
Needs the analysis dependencies: pip install --group analysis
"""

import json
import os
import random
from collections import Counter
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")  # hide DiCE's progress bar

import dice_ml
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).parent
PROCESSED_FILE = ROOT / "results" / "processed.json"
OUTPUT_FILE = ROOT / "results" / "counterfactuals.json"

FEATURES = ["narrative", "granularity", "profile", "model"]
OUTCOME = "yes"
COUNTERFACTUALS_PER_UNIVERSE = 3
# The forest's P(YES) a counterfactual must reach. High on purpose: the goal is
# universes where every run is YES, not just most. (In the current data, all-YES
# universes score >= 0.975 and the rest <= 0.67.)
TARGET_P_YES = 0.9
SEED = 0


def load_runs() -> pd.DataFrame:
    """One row per run: the four choices plus yes = 1 if the stance was YES."""
    records = json.loads(PROCESSED_FILE.read_text())
    rows = [
        {
            "id": r["id"],
            "narrative": r["narrative"],
            "granularity": r["granularity"],
            "profile": r["profile"] or "none",
            "model": r["model"],
            OUTCOME: int(r["stance"] == "YES"),
        }
        for r in records
        # Skip API errors and records whose prompt isn't in prompts.json any more.
        if not r.get("error") and r.get("narrative") is not None
    ]
    if not rows:
        raise SystemExit(f"No usable runs in {PROCESSED_FILE.relative_to(ROOT)}")
    return pd.DataFrame(rows)


def train(runs: pd.DataFrame) -> Pipeline:
    pipeline = Pipeline(
        [
            (
                "onehot",
                ColumnTransformer(
                    [("choices", OneHotEncoder(handle_unknown="ignore"), FEATURES)]
                ),
            ),
            ("forest", RandomForestClassifier(n_estimators=300, random_state=SEED)),
        ]
    )
    pipeline.fit(runs[FEATURES], runs[OUTCOME])
    return pipeline


def summarize_universes(runs: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """One row per universe: observed YES runs and the forest's predicted P(YES)."""
    universes = (
        runs.groupby(FEATURES, as_index=False)
        .agg(id=("id", "first"), runs=(OUTCOME, "size"), yes_runs=(OUTCOME, "sum"))
        .sort_values("id")
        .reset_index(drop=True)
    )
    universes["observed_yes_rate"] = universes["yes_runs"] / universes["runs"]
    universes["predicted_p_yes"] = model.predict_proba(universes[FEATURES])[:, 1]
    universes["all_yes"] = universes["yes_runs"] == universes["runs"]
    universes["predicted_all_yes"] = universes["predicted_p_yes"] >= TARGET_P_YES
    return universes


def explain(runs: pd.DataFrame, model: Pipeline, universes: pd.DataFrame) -> list[dict]:
    # Every universe where at least one run wasn't YES, whatever the majority said.
    queries = universes.query("not all_yes").reset_index(drop=True)
    if queries.empty:
        return []

    explainer = dice_ml.Dice(
        dice_ml.Data(
            dataframe=runs[[*FEATURES, OUTCOME]],
            continuous_features=[],
            outcome_name=OUTCOME,
        ),
        dice_ml.Model(model=model, backend="sklearn"),
        method="random",
    )
    # DiCE's random_seed alone isn't enough: it also draws from Python's and NumPy's
    # global generators, so seed those too or every run gives different answers.
    random.seed(SEED)
    np.random.seed(SEED)
    found = explainer.generate_counterfactuals(
        queries[FEATURES],
        total_CFs=COUNTERFACTUALS_PER_UNIVERSE,
        desired_class=1,  # pyright: ignore[reportArgumentType] (DiCE types it as str)
        stopping_threshold=TARGET_P_YES,
        random_seed=SEED,
    )

    # Look up any combination of choices in the real runs.
    observed = {
        tuple(u[f] for f in FEATURES): u for u in universes.to_dict(orient="records")
    }
    explanations = []
    for query, example in zip(
        queries.to_dict(orient="records"), found.cf_examples_list, strict=True
    ):
        cfs = example.final_cfs_df
        counterfactuals = []
        for cf in [] if cfs is None else pd.DataFrame(cfs).to_dict(orient="records"):
            changes = {
                f: {"from": query[f], "to": cf[f]}
                for f in FEATURES
                if cf[f] != query[f]
            }
            row = observed.get(tuple(cf[f] for f in FEATURES))
            counterfactuals.append(
                {
                    "changes": changes,
                    "universe_id": int(row["id"]) if row else None,
                    "observed_yes_runs": int(row["yes_runs"]) if row else None,
                    "observed_runs": int(row["runs"]) if row else None,
                    # Holds only if every observed run of the changed universe was YES.
                    "holds": bool(row["all_yes"]) if row else None,
                }
            )
        counterfactuals.sort(key=lambda c: len(c["changes"]))
        explanations.append(
            {
                "universe_id": int(query["id"]),
                "choices": {f: query[f] for f in FEATURES},
                "observed_yes_runs": int(query["yes_runs"]),
                "observed_runs": int(query["runs"]),
                "predicted_p_yes": round(float(query["predicted_p_yes"]), 3),
                "counterfactuals": counterfactuals,
            }
        )
    return explanations


def print_report(universes: pd.DataFrame, explanations: list[dict]) -> None:
    agree = (universes["predicted_all_yes"] == universes["all_yes"]).mean()
    print(
        f"{len(universes)} universes; the forest's P(YES) >= {TARGET_P_YES} matches "
        f"'every run was YES' in {agree:.0%} of them."
    )
    print(
        f"{len(explanations)} universes have at least one non-YES run; "
        "counterfactuals for each:\n"
    )

    changed_features: Counter = Counter()
    changed_to: Counter = Counter()
    held = total = 0
    for e in explanations:
        c = e["choices"]
        print(
            f"Universe #{e['universe_id']}  {c['narrative']} | {c['granularity']} | "
            f"{c['profile']} | {c['model']}  "
            f"(observed YES {e['observed_yes_runs']}/{e['observed_runs']})"
        )
        if not e["counterfactuals"]:
            print("    DiCE found no change that makes every run YES")
        for cf in e["counterfactuals"]:
            change_text = ", ".join(
                f"{f}: {v['from']} -> {v['to']}" for f, v in cf["changes"].items()
            )
            if cf["observed_runs"]:
                total += 1
                verdict = cf["holds"]
                held += verdict
                check = (
                    f"observed YES {cf['observed_yes_runs']}/{cf['observed_runs']} "
                    f"in #{cf['universe_id']} "
                    + ("(holds)" if verdict else "(doesn't hold)")
                )
            else:
                check = "never run, can't check"
            print(f"    change {change_text}  ->  {check}")
            changed_features.update(cf["changes"].keys())
            changed_to.update(f"{f} = {v['to']}" for f, v in cf["changes"].items())
        print()

    if total:
        print(
            f"{held}/{total} counterfactuals hold (every observed run of the changed "
            "universe was YES)."
        )
    if changed_features:
        print("\nChoices DiCE changed most often to reach all-YES:")
        for feature, n in changed_features.most_common():
            print(f"    {feature:<12} {n}")
        print("\nMost common new values:")
        for value, n in changed_to.most_common(6):
            print(f"    {value:<40} {n}")


def main() -> None:
    runs = load_runs()
    model = train(runs)
    universes = summarize_universes(runs, model)
    explanations = explain(runs, model, universes)
    print_report(universes, explanations)
    OUTPUT_FILE.write_text(json.dumps(explanations, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
