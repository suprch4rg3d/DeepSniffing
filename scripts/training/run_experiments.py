"""Run a matrix of model and input-branch experiments and compare their scores.

Every run uses the same data bundle, the same split, and the same training loop,
so differences in the final score come from the model configuration alone.

Examples:
    poetry run python -m scripts.training.run_experiments --list
    poetry run python -m scripts.training.run_experiments --only hybrid_smiles_only
    poetry run python -m scripts.training.run_experiments --output-dir results/ablation
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from scripts.core.data import DatasetBundle, prepare_data
from scripts.core.train import TrainConfig, resolve_device, seed_everything, train_and_evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/experiments"

ALL_BRANCHES = frozenset({"smiles", "descriptors", "fingerprints"})


@dataclass(frozen=True)
class Experiment:
    """One row of the ablation matrix."""

    name: str
    module: str
    branches: frozenset[str] = field(default=ALL_BRANCHES)
    notes: str = ""


DEFAULT_EXPERIMENTS = [
    Experiment("hybrid", "scripts.training.custom_model", ALL_BRANCHES, "all three inputs"),
    Experiment("hybrid_smiles_only", "scripts.training.custom_model", frozenset({"smiles"})),
    Experiment(
        "hybrid_descriptors_only", "scripts.training.custom_model", frozenset({"descriptors"})
    ),
    Experiment(
        "hybrid_fingerprints_only", "scripts.training.custom_model", frozenset({"fingerprints"})
    ),
    Experiment(
        "hybrid_smiles_descriptors",
        "scripts.training.custom_model",
        frozenset({"smiles", "descriptors"}),
    ),
    Experiment(
        "hybrid_smiles_fingerprints",
        "scripts.training.custom_model",
        frozenset({"smiles", "fingerprints"}),
    ),
    Experiment("cnn", "scripts.training.cnn", frozenset({"smiles"}), "SMILES only baseline"),
    Experiment("transformer", "scripts.training.transformer", frozenset({"smiles"}), "attention"),
]


def load_experiments(config_path: Path | None) -> list[Experiment]:
    """Return the default matrix, or one described by a JSON file."""
    if config_path is None:
        return list(DEFAULT_EXPERIMENTS)
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config does not exist: {config_path}")

    entries = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("Experiment config must be a JSON list.")
    experiments = []
    for entry in entries:
        experiments.append(
            Experiment(
                name=str(entry["name"]),
                module=str(entry["module"]),
                branches=frozenset(entry.get("branches", ALL_BRANCHES)),
                notes=str(entry.get("notes", "")),
            )
        )
    return experiments


def load_builder(module_path: str):
    """Return the build_model callable of a model module, or None if absent."""
    module = importlib.import_module(module_path)
    builder = getattr(module, "build_model", None)
    if builder is None:
        return None
    return builder


def run_experiment(
    experiment: Experiment,
    bundle: DatasetBundle,
    config: TrainConfig,
    device,
    builder,
) -> dict:
    """Train one configuration and return a flat summary row."""
    seed_everything(config.seed)
    model = builder(bundle, branches=experiment.branches)
    result = train_and_evaluate(model, bundle, config, device=device, verbose=False)
    metrics = result.test_metrics
    row = {
        "name": experiment.name,
        "module": experiment.module,
        "branches": "+".join(sorted(experiment.branches)),
        "notes": experiment.notes,
        "best_epoch": result.best_epoch,
        "dream_challenge_score": metrics["dream_challenge_score"],
    }
    for key, value in metrics["pearson_correlations"].items():
        row[f"pearson_{key}"] = value
    for key, value in metrics["z_scores"].items():
        row[f"z_{key}"] = value
    return row


def write_summary(rows: list[dict], output_dir: Path) -> Path:
    """Write the comparison table as Markdown and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values("dream_challenge_score", ascending=False)

    csv_path = output_dir / "summary.csv"
    frame.to_csv(csv_path, index=False)

    lines = [
        "# Ablation summary",
        "",
        "Every row uses the same split, the same seed, and the same training loop.",
        "The score is the official DREAM Q2 mean Z-score on the 69 test molecules.",
        "",
        "| Model | Inputs | DREAM score | Intensity r | Valence r | Descriptors r | Epoch |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.name} | {row.branches} | {row.dream_challenge_score:.3f} "
            f"| {row.pearson_intensity_mean:.3f} | {row.pearson_valence_mean:.3f} "
            f"| {row.pearson_descriptors_mean:.3f} | {row.best_epoch} |"
        )
    lines.append("")
    markdown_path = output_dir / "summary.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ablation matrix.")
    parser.add_argument("--config", type=Path, help="Optional JSON description of the matrix.")
    parser.add_argument(
        "--only", action="append", dest="only", help="Run only the named experiment. Repeatable."
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override the epoch budget.")
    parser.add_argument("--device", default=None, help="Torch device to train on.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where the summary table is written.",
    )
    parser.add_argument("--list", action="store_true", help="List the experiments and exit.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiments = load_experiments(args.config)

    if args.only:
        known = {experiment.name for experiment in experiments}
        unknown = sorted(set(args.only) - known)
        if unknown:
            raise SystemExit(f"Unknown experiment(s): {', '.join(unknown)}")
        experiments = [experiment for experiment in experiments if experiment.name in args.only]

    if args.list:
        for experiment in experiments:
            print(f"{experiment.name:<28} {experiment.module:<34} {sorted(experiment.branches)}")
        return

    device = resolve_device(args.device)
    config = TrainConfig(epochs=args.epochs) if args.epochs else TrainConfig()
    print(f"Running {len(experiments)} experiment(s) on {device}")

    rows = []
    skipped = []
    bundle = prepare_data()
    for experiment in experiments:
        builder = load_builder(experiment.module)
        if builder is None:
            skipped.append(experiment)
            print(f"{experiment.name}: skipped, {experiment.module} has no build_model yet")
            continue
        print(f"{experiment.name}: training")
        row = run_experiment(experiment, bundle, config, device, builder)
        rows.append(row)
        print(f"{experiment.name}: DREAM score {row['dream_challenge_score']:.3f}")

    if not rows:
        raise SystemExit(
            "No experiment could run. Implement build_model in the model modules first."
        )

    summary_path = write_summary(rows, args.output_dir)
    print(f"Summary: {summary_path}")
    if skipped:
        print("Skipped: " + ", ".join(experiment.name for experiment in skipped))


if __name__ == "__main__":
    main()
