"""Compare stage 2 runs on the held-out molecules, molecule by molecule.

    python -m scripts.core.compare REFERENCE OTHER [OTHER ...]       (runs named by their --tag)
    python -m scripts.core.compare --split test REFERENCE OTHER

The numbers come from ``evaluate`` (run it for each tag first). Two runs are scored on the same
molecules with the same requests, so the difference between them can be taken for each molecule and
averaged, and that is much more exact than setting two averages side by side: the molecules that are hard
are hard for both. A difference is only worth believing when its 95% interval leaves out zero.

For every run it shows the loss of writing a held-out molecule after a request for its own odor tokens (lower is
better), then how that moves against the reference, and the gain the odor tokens give over no odor tokens.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.core.training import OUTPUT_DIR


def read_run(tag: str | None, split: str = "validation", folder: Path = OUTPUT_DIR) -> dict:
    """The conditioning results of a run; ``stage2`` (or no tag) is the run trained with no ``--tag``."""
    suffix = f"_{tag}" if tag and tag != "stage2" else ""
    path = folder / f"evaluation_{split}{suffix}.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run evaluate for this run first.")
    report = json.loads(path.read_text(encoding="utf-8"))
    if "per_molecule" not in report.get("conditioning", {}):
        raise ValueError(
            f"{path} was made before the per-molecule losses were kept. Run evaluate again."
        )
    return report["conditioning"]


def paired(reference: np.ndarray, other: np.ndarray) -> dict:
    """The mean of other minus reference, its 95% interval, and the share of molecules where other is lower."""
    difference = np.asarray(other) - np.asarray(reference)
    return {
        "mean": float(difference.mean()),
        "interval_95": float(1.96 * difference.std(ddof=1) / np.sqrt(len(difference))),
        "lower_for": float((difference < 0).mean()),
    }


def compare(reference: dict, others: dict[str, dict]) -> list[dict]:
    """One row for the reference and one for each other run, in the order given."""
    rows = []
    base = np.array(reference["per_molecule"]["own"])
    for name, run in {"reference": reference, **others}.items():
        own = np.array(run["per_molecule"]["own"])
        none = np.array(run["per_molecule"]["none"])
        if len(own) != len(base):
            raise ValueError(
                f"{name} was scored on {len(own)} molecules, the reference on {len(base)}"
            )
        row = {
            "run": name,
            "own_odor_tokens": float(own.mean()),
            "gain": float((none - own).mean()),
            "gain_interval_95": float(1.96 * (none - own).std(ddof=1) / np.sqrt(len(own))),
        }
        if name != "reference":
            row["against_reference"] = paired(base, own)
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare stage 2 runs molecule by molecule.")
    parser.add_argument(
        "reference", help="the tag of the run the others are compared with ('' for stage2)"
    )
    parser.add_argument("others", nargs="+", help="the tags of the runs to compare with it")
    parser.add_argument("--split", choices=("validation", "test", "dream"), default="validation")
    arguments = parser.parse_args(argv)
    reference = read_run(arguments.reference or None, arguments.split)
    others = {tag: read_run(tag, arguments.split) for tag in arguments.others}
    print(
        f"{'run':16s} {'own odor tokens':>10s} {'gain over no odor tokens':>20s} {'change against the reference':>34s}"
    )
    for row in compare(reference, others):
        text = ""
        if "against_reference" in row:
            a = row["against_reference"]
            sure = "  <- real" if abs(a["mean"]) > a["interval_95"] else "  (within the noise)"
            text = f"{a['mean']:+8.4f} +/- {a['interval_95']:.4f}   lower for {100 * a['lower_for']:3.0f}%{sure}"
        name = row["run"] if row["run"] != "reference" else arguments.reference or "stage2"
        print(
            f"{name:16s} {row['own_odor_tokens']:10.4f} {row['gain']:+9.4f} +/- {row['gain_interval_95']:.4f}   {text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
