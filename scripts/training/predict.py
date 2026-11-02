"""Score a molecule that the model has never seen.

This is the deliverable that the project proposal promised: a trained model that
accepts a novel SMILES string and returns the predicted odor profile.

Example:
    poetry run python -m scripts.training.predict --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from scripts.core.data import (
    FINGERPRINT_RADIUS,
    FINGERPRINT_SIZE,
    MAX_SEQUENCE_LENGTH,
)
from scripts.core.metrics import TARGETS
from scripts.core.train import resolve_device
from scripts.preprocessing.enrich_molecules import describe_molecule
from scripts.preprocessing.smiles_tokenizer import SmilesTokenizer
from scripts.training.custom_model import CustomOdorModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "output/custom_model"

PREDICTION_EXAMPLE = "CC(=O)OC1=CC=CC=C1C(=O)O"


def load_model(model_dir: Path = DEFAULT_MODEL_DIR, device: torch.device | None = None):
    """Load a checkpoint together with the artifacts that shaped its inputs."""
    checkpoint_path = model_dir / "checkpoints/custom_model.pt"
    vocabulary_path = model_dir / "checkpoints/smiles_vocabulary.json"
    feature_scaler_path = model_dir / "checkpoints/feature_scaler.joblib"
    target_scaler_path = model_dir / "checkpoints/target_scaler.joblib"

    for path in (checkpoint_path, vocabulary_path, feature_scaler_path, target_scaler_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing model artifact: {path}. Train the model first with "
                "python -m scripts.training.custom_model."
            )

    device = device or resolve_device()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CustomOdorModel(
        vocab_size=checkpoint["vocabulary_size"],
        feature_count=len(checkpoint["descriptor_columns"]),
        fingerprint_count=checkpoint["morgan_fingerprint_size"],
        target_count=checkpoint["prediction_outputs"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    tokenizer = SmilesTokenizer()
    tokenizer.load(vocabulary_path)

    return {
        "model": model,
        "device": device,
        "checkpoint": checkpoint,
        "tokenizer": tokenizer,
        "feature_scaler": joblib.load(feature_scaler_path),
        "target_scaler": joblib.load(target_scaler_path),
    }


def featurize_smiles(smiles: str, descriptor_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Turn one SMILES string into its descriptor vector and Morgan fingerprint."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse this SMILES string: {smiles!r}")

    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    descriptors = describe_molecule(canonical, cid=0)
    missing = set(descriptor_columns) - set(descriptors)
    if missing:
        raise ValueError(f"Descriptor columns missing from RDKit output: {sorted(missing)}")
    descriptor_values = np.array(
        [[float(descriptors[column]) for column in descriptor_columns]], dtype=np.float64
    )

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FINGERPRINT_RADIUS,
        countSimulation=False,
        includeChirality=True,
        useBondTypes=True,
        fpSize=FINGERPRINT_SIZE,
    )
    bits = np.zeros(FINGERPRINT_SIZE, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), bits)
    return descriptor_values, bits.astype(np.float32)


def predict(smiles: str, artifacts: dict) -> pd.DataFrame:
    """Predict the mean and the spread of the perceived odor of one molecule."""
    checkpoint = artifacts["checkpoint"]
    descriptor_columns = checkpoint["descriptor_columns"]
    max_length = int(checkpoint.get("max_sequence_length", MAX_SEQUENCE_LENGTH))
    target_names = checkpoint.get("target_names", TARGETS)

    descriptors, fingerprint = featurize_smiles(smiles, descriptor_columns)
    tokens = artifacts["tokenizer"].encode(smiles, max_length)

    features_tensor = torch.tensor(
        artifacts["feature_scaler"].transform(descriptors), dtype=torch.float32
    ).to(artifacts["device"])
    tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(artifacts["device"])
    fingerprint_tensor = torch.from_numpy(fingerprint).unsqueeze(0).to(artifacts["device"])

    with torch.no_grad():
        scaled = (
            artifacts["model"](tokens_tensor, features_tensor, fingerprint_tensor).cpu().numpy()
        )
    restored = np.clip(artifacts["target_scaler"].inverse_transform(scaled), 0.0, 100.0)[0]

    count = len(target_names)
    means = np.round(restored[:count].astype(np.float64), 2)
    sigmas = np.round(restored[count : 2 * count].astype(np.float64), 2)
    return pd.DataFrame({"attribute": target_names, "mean": means, "sigma": sigmas})


def describe_molecule_label(smiles: str) -> str:
    """Human readable form of the input, canonicalised where possible."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return smiles
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict the odor profile of a molecule from its SMILES string."
    )
    parser.add_argument(
        "--smiles",
        action="append",
        dest="smiles",
        help="SMILES string to score. Repeat the flag to score several molecules.",
    )
    parser.add_argument(
        "--smiles-file",
        type=Path,
        help="Optional file with one SMILES string per line.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory holding the trained checkpoint and its artifacts.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, for example cpu, cuda, or mps. Defaults to CUDA if present.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the predictions as JSON instead of a table.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only show the N strongest attributes, ranked by predicted mean.",
    )
    return parser


def collect_smiles(args: argparse.Namespace) -> list[str]:
    values = list(args.smiles or [])
    if args.smiles_file:
        if not args.smiles_file.is_file():
            raise FileNotFoundError(f"SMILES file does not exist: {args.smiles_file}")
        values.extend(
            line.strip()
            for line in args.smiles_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not values:
        raise SystemExit(
            "No input given. Pass --smiles with a SMILES string, or --smiles-file. "
            f'Example: --smiles "{PREDICTION_EXAMPLE}"'
        )
    return values


def main() -> None:
    args = build_parser().parse_args()
    smiles_values = collect_smiles(args)

    device = torch.device(args.device) if args.device else None
    artifacts = load_model(args.model_dir, device)

    results = {}
    for smiles in smiles_values:
        try:
            table = predict(smiles, artifacts)
        except ValueError as error:
            raise SystemExit(f"Cannot score {smiles!r}: {error}") from error
        if args.top is not None:
            table = table.sort_values("mean", ascending=False).head(args.top)
        results[describe_molecule_label(smiles)] = table

    if args.json:
        print(
            json.dumps(
                {
                    name: table.set_index("attribute")[["mean", "sigma"]].to_dict(orient="index")
                    for name, table in results.items()
                },
                indent=2,
            )
        )
        return

    for name, table in results.items():
        print(f"\nMolecule: {name}")
        print(f"{'attribute':<24}{'mean':>8}{'sigma':>8}")
        print("-" * 40)
        for row in table.itertuples(index=False):
            print(f"{row.attribute:<24}{row.mean:>8.2f}{row.sigma:>8.2f}")


if __name__ == "__main__":
    main()
