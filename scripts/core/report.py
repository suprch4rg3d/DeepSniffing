"""The results of a model in one place: a table of what ``evaluate`` measured, and its figures.

Reads the ``evaluation_<split>.json`` files that ``evaluate`` wrote and writes ``results.md`` and the figures to
``<output>/report``. It measures nothing itself.

    python -m scripts.core.report [--tag NAME]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.core.plots import gain_histogram, generation_bars
from scripts.core.training import OUTPUT_DIR

SPLITS = ("validation", "test", "dream")


def read_evaluations(folder: Path, tag: str | None) -> dict[str, dict]:
    """The evaluations that exist, by split."""
    suffix = f"_{tag}" if tag else ""
    found = {}
    for split in SPLITS:
        path = folder / f"evaluation_{split}{suffix}.json"
        if path.is_file():
            found[split] = json.loads(path.read_text(encoding="utf-8"))
    return found


def classifier_folder(tag: str | None) -> Path:
    """Where the classifier keeps its evaluations: ``stage3``, or ``stage3_<tag>`` for a tagged run."""
    return OUTPUT_DIR / (f"stage3_{tag}" if tag else "stage3")


def classifier_table(evaluations: dict[str, dict]) -> str:
    """The classifier's scores, the half of the project the generator tables say nothing about."""
    rows = [
        "| split | molecules | ROC-AUC | average precision | precision at 5 | precision at its own count |",
        "|---|---|---|---|---|---|",
    ]
    for split, result in evaluations.items():
        rows.append(
            f"| {split} | {result['molecules']} | {result['roc_auc_macro']:.3f} "
            f"| {result['average_precision_macro']:.3f} | {result['precision_at_5']:.3f} "
            f"| {result['precision_at_own_count']:.3f} |"
        )
    return "\n".join(rows)


def read_baseline() -> dict | None:
    """The ridge baseline, if it has been run. It is written by ``scripts.analysis.baseline``."""
    path = OUTPUT_DIR.parent / "analysis" / "baseline.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def read_eda() -> dict | None:
    """The data summary, if it has been built. It is written by ``scripts.analysis.eda``."""
    path = OUTPUT_DIR.parent / "analysis" / "eda.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def eda_table(eda: dict) -> str:
    """The size and shape of the data, before any result is read."""
    splits = eda["splits"]
    per = eda["tokens_per_molecule"]
    rows = [
        "| measure | value |",
        "|---|---|",
        f"| molecules | {eda['molecules']:,}: {splits.get('train', 0):,} train, "
        f"{splits.get('validation', 0)} validation, {splits.get('test', 0)} test, "
        f"{splits.get('dream', 0)} DREAM |",
        f"| odor tokens | {eda['odor_tokens']}, {eda['odor_tokens_on_one_molecule']} on a single molecule |",
        f"| odor tokens per molecule | median {per['median']:.0f}, mean {per['mean']:.1f}, most {per['most']} |",
        f"| sources | {len(eda['sources'])} |",
        f"| weight | median {eda['weight']['median']:.0f} Da, "
        f"{eda['weight']['share_over_350']:.1%} over 350 |",
    ]
    return "\n".join(rows)


def baseline_table(baseline: dict, classifier: dict) -> str:
    """The ridge baseline beside the transformer, split by split."""
    rows = ["| split | ridge on Morgan bits | the transformer | difference |", "|---|---|---|---|"]
    for split, found in baseline["splits"].items():
        neural = classifier.get(split, {}).get("roc_auc_macro")
        ridge = found.get("roc_auc_macro")
        if neural is None or ridge is None:
            continue
        rows.append(f"| {split} | {ridge:.3f} | {neural:.3f} | {neural - ridge:+.3f} |")
    return "\n".join(rows)


def results_table(evaluations: dict[str, dict]) -> str:
    rows = [
        "| split | molecules | loss | perplexity | gain over no odor tokens | odor tokens help |",
        "|---|---|---|---|---|---|",
    ]
    for split, result in evaluations.items():
        c = result["conditioning"]
        gain = c["gain_over_no_odor_tokens"]
        rows.append(
            f"| {split} | {c['molecules']} | {c['loss_with_own_odor_tokens']:.4f} "
            f"| {c['perplexity_with_own_odor_tokens']:.2f} "
            f"| {gain['mean']:+.3f} ± {gain['interval_95']:.3f} | {gain['helped']:.0%} |"
        )
    return "\n".join(rows)


def generation_table(generation: dict) -> str:
    rows = [
        "| request | valid | unique | novel | median weight | over 350 | nearest training | over 0.7 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for request, g in generation.items():
        similarity = g.get("similarity") or {}
        nearest = f"{similarity['median']:.2f}" if similarity else "n/a"
        close = f"{similarity['above_0.7']:.0%}" if similarity else "n/a"
        rows.append(
            f"| {request} | {g['valid']:.2f} | {g['unique']:.2f} | {g['novel']:.2f} "
            f"| {g['weight_median']:.0f} | {g['weight_over_350']:.1%} | {nearest} | {close} |"
        )
    return "\n".join(rows)


def per_token_table(rows: list[dict]) -> str:
    """The odor tokens the classifier ranks worst, with how many molecules carry each."""
    lines = ["| odor token | ROC-AUC | molecules |", "|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['token']} | {row['auc']:.3f} | {row['molecules']} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tables and figures of the evaluations.")
    parser.add_argument("--tag", help="a stage 2 run that was evaluated with --tag")
    tag = parser.parse_args().tag
    evaluations = read_evaluations(OUTPUT_DIR, tag)
    if not evaluations:
        raise SystemExit(f"no evaluation_*.json in {OUTPUT_DIR}. Run evaluate first.")
    classifier = read_evaluations(classifier_folder(tag), None)
    folder = OUTPUT_DIR / "report"
    folder.mkdir(exist_ok=True)
    text = ["# Results", "", results_table(evaluations)]
    eda = read_eda()
    if eda:
        text += ["", "## The data", "", eda_table(eda)]
    generation = None
    for split, result in evaluations.items():
        own_and_none = result["conditioning"]["per_molecule"]
        gain_histogram(
            own_and_none["own"], own_and_none["none"], folder / f"gain_{split}.png", split
        )
        # What the model writes depends on its requests and the seed, not on which molecules were held out,
        # so every split answers with the same sample. It is one result, and it is shown once.
        if generation is None:
            generation = result.get("generation")
    if generation:
        generation_bars(generation, folder / "generation.png", "the model")
        text += ["", "## What it wrote", "", generation_table(generation)]
    if classifier:
        text += ["", "## The classifier", "", classifier_table(classifier)]
        shown = classifier.get("test", next(iter(classifier.values()), {}))
        baseline = read_baseline()
        if baseline:
            text += ["", "## The baseline", "", baseline_table(baseline, classifier)]
        if shown.get("per_token"):
            text += [
                "",
                "### The ten odor tokens it ranks worst on the test molecules",
                "",
                per_token_table(shown["per_token"][:10]),
            ]
    (folder / "results.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"wrote {folder / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
