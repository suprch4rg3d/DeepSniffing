"""Score a DREAM molecule with the trained custom model.

The custom model reads the Dragon descriptors, the odor name, the community features
(EPI Suite, NSPDK, and similarity) and a 3D conformer of each molecule. Dragon is
proprietary software, so those descriptors only exist for the 476 molecules that the
DREAM challenge published, and the community features are published for the same ones. A molecule is therefore
looked up in the prepared molecule tables, by SMILES or by compound identifier,
and scored at high concentration. A SMILES string that is not in the tables
cannot be scored.

Example:
    poetry run python -m scripts.training.predict --smiles "CCO"
    poetry run python -m scripts.training.predict --smiles 702
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from scripts.core.data import (
    STANDARD_LOG10_DILUTION,
    apply_std_curve,
    collate_molecules,
    dragon_group_indices,
    dream_similarity_matrices,
    load_conformers,
    molecule_graph,
    name_windows,
    neighbour_features,
    read_rating_table,
    transform_features,
)
from scripts.core.metrics import TARGETS
from scripts.core.splits import LEADERBOARD_RATINGS_PATH, TEST_RATINGS_PATH, TRAIN_RATINGS_PATH
from scripts.core.train import inverse_target_transform, resolve_device
from scripts.preprocessing.community_features import OUTPUT as COMMUNITY_PATH
from scripts.preprocessing.enrich_molecules import OUTPUT_PATH as GRAPH_MOLECULES_PATH
from scripts.preprocessing.morgan_fingerprints import OUTPUT as FINGERPRINTS_PATH
from scripts.training.custom_model import GraphDescriptorModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "output/q2"

PREDICTION_EXAMPLE = "CCO"


def load_model(model_dir: Path = DEFAULT_MODEL_DIR, device: torch.device | str | None = None):
    """Load a trained checkpoint, which carries everything that shaped its inputs."""
    checkpoint_path = model_dir / "checkpoints/custom_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing model artifact: {checkpoint_path}. Train the model first with "
            "python -m scripts.core.train."
        )

    device = torch.device(device) if device is not None else resolve_device()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != "GraphDescriptorModel":
        raise ValueError(
            f"{checkpoint_path} was not written by the graph descriptor model. "
            "Train the model again with python -m scripts.core.train."
        )
    if checkpoint.get("challenge", "q2") != "q2":
        raise ValueError(
            f"{checkpoint_path} predicts every participant's own rating (Subchallenge 1). "
            "This script reports the population means and spreads of Subchallenge 2."
        )
    scaler_paths = {
        name: model_dir / f"checkpoints/{name}.joblib"
        for name in ("feature_scaler", "target_scaler")
    }
    for path in scaler_paths.values():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing model artifact: {path}. Train the model first with "
                "python -m scripts.core.train."
            )

    def restore(state_dict: dict) -> GraphDescriptorModel:
        model = GraphDescriptorModel(
            dragon_group_indices(checkpoint["dragon_columns"], checkpoint["dragon_feature_groups"]),
            checkpoint["numeric_count"],
            checkpoint.get("name_count", len(checkpoint["name_ngram_vocabulary"])),
            len(checkpoint["target_columns"]),
            structure_count=checkpoint["structure_count"],
            neighbour_count=checkpoint["neighbours"].get(
                "width", checkpoint["neighbours"]["labels"].shape[1]
            )
            if checkpoint.get("neighbours")
            else 0,
            experts=bool(checkpoint.get("experts", False)),
        )
        model.load_state_dict(state_dict)
        return model.to(device).eval()

    # A model trained with one epoch per score group keeps a copy for each group.
    group_models = {
        name: restore(state) for name, state in checkpoint.get("group_state_dicts", {}).items()
    }
    return {
        "model": restore(checkpoint["state_dict"]),
        "group_models": group_models,
        "device": device,
        "checkpoint": checkpoint,
        "feature_scaler": joblib.load(scaler_paths["feature_scaler"]),
        "target_scaler": joblib.load(scaler_paths["target_scaler"]),
    }


@functools.lru_cache(maxsize=2)
def dream_similarities(names: tuple[str, ...]) -> list[pd.DataFrame]:
    """The similarity matrices that the model reads, read once."""
    return dream_similarity_matrices(names)


def neighbour_input(cid: int, checkpoint: dict) -> np.ndarray:
    """What the most similar training molecules of the model were rated, for one molecule."""
    saved = checkpoint["neighbours"]
    reference = pd.DataFrame(
        saved["labels"], index=pd.Index(saved["cids"], dtype="int64"), columns=list(TARGETS)
    )
    names = tuple(saved.get("similarities", ["morgan"]))
    similarities = dream_similarities(names)
    if cid not in similarities[0].index:
        raise ValueError(
            f"CID {cid} has no similarity to the DREAM molecules, which the model needs."
        )
    statistics = {"mean": saved["mean"], "scale": saved["scale"]}
    return np.concatenate(
        [
            neighbour_features(
                np.array([cid], dtype=np.int64),
                reference,
                similarity,
                statistics,
                int(saved["power"]),
                int(saved["count"]),
            )[0]
            for similarity in similarities
        ]
    )


def load_tables(
    molecules_path: Path = GRAPH_MOLECULES_PATH, fingerprints_path: Path = FINGERPRINTS_PATH
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the prepared molecule and fingerprint tables the model was trained on."""
    for path in (molecules_path, fingerprints_path):
        if not path.is_file():
            raise FileNotFoundError(f"Prepared molecule table is missing: {path}")
    return pd.read_csv(molecules_path), pd.read_csv(fingerprints_path)


def load_community_table(path: Path = COMMUNITY_PATH) -> pd.DataFrame | None:
    """Read the community feature table when it exists."""
    return pd.read_csv(path) if path.is_file() else None


@functools.lru_cache(maxsize=1)
def listed_concentrations() -> dict[int, tuple[float, float]]:
    """The dilution and the high flag that the descriptors of each DREAM molecule were rated at.

    This is what the training and the scoring of the model give the molecule: for a training molecule
    its high rating, and for a leaderboard or test molecule the dilution that the file lists for it,
    flagged high only when it is stronger than 1/1000.
    """
    found: dict[int, tuple[float, float]] = {}
    for path, held_out in (
        (TRAIN_RATINGS_PATH, False),
        (LEADERBOARD_RATINGS_PATH, True),
        (TEST_RATINGS_PATH, True),
    ):
        ratings = read_rating_table(path).rename(columns={"Compound Identifier": "CID"})
        high = ratings.loc[ratings["Intensity"].astype(str).str.strip().str.lower().eq("high")]
        dilution = (
            high["Dilution"]
            .astype(str)
            .str.strip()
            .str.strip(" \"'")
            .str.replace(",", "", regex=False)
        )
        log10 = -np.log10(pd.to_numeric(dilution.str.split("/", n=1).str[-1], errors="raise"))
        for cid, value in zip(pd.to_numeric(high["CID"]).astype(int), log10, strict=True):
            flag = float(value > STANDARD_LOG10_DILUTION) if held_out else 1.0
            found.setdefault(int(cid), (float(value), flag))
    return found


def canonical_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse this SMILES string: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def find_molecule(query: str, molecules: pd.DataFrame) -> pd.Series:
    """Return the table row of a molecule given as a SMILES string or a CID."""
    query = query.strip()
    if query.isdigit():
        matches = molecules.loc[molecules["CID"] == int(query)]
    else:
        matches = molecules.loc[molecules["rdkit_canonical_smiles"] == canonical_smiles(query)]
    if matches.empty:
        raise ValueError(
            f"{query!r} is not one of the DREAM molecules. The model needs Dragon descriptors, "
            "which only exist for the molecules published with the challenge."
        )
    return matches.iloc[0]


def build_inputs(
    row: pd.Series,
    fingerprints: pd.DataFrame,
    artifacts: dict,
    community: pd.DataFrame | None = None,
) -> dict[str, torch.Tensor | int]:
    """Rebuild, for one molecule, the inputs that the model saw during training."""
    checkpoint = artifacts["checkpoint"]
    cid = int(row["CID"])
    fingerprint = fingerprints.loc[fingerprints["CID"] == cid]
    if fingerprint.empty:
        raise ValueError(f"No Morgan fingerprint is stored for CID {cid}.")

    # One table row in the layout the training rows had, scored at high concentration.
    merged = {**row.to_dict(), **fingerprint.iloc[0].to_dict()}
    needed = checkpoint.get("community_columns", {})
    if community is not None:
        match = community.loc[community["CID"] == cid]
        if match.empty:
            raise ValueError(f"No community features are stored for CID {cid}.")
        merged.update(match.iloc[0].to_dict())
    elif needed.get("name") or needed.get("structure"):
        raise ValueError("This model needs the community features. Run ds-build to create them.")
    frame = pd.DataFrame([merged])
    typical = (checkpoint["high_log10_concentration"], 1.0)
    frame["log10_concentration"], frame["is_high_concentration"] = listed_concentrations().get(
        cid, typical
    )
    dragon, numeric = transform_features(
        frame,
        artifacts["feature_scaler"],
        checkpoint["dragon_columns"],
        checkpoint["morgan_columns"],
    )

    windows = name_windows(str(row["OdorName"]))
    own = [float(w in windows) for w in checkpoint["name_ngram_vocabulary"]]
    organizer = [float(frame[column].iloc[0]) for column in needed.get("name", [])]
    name_features = np.nan_to_num(np.array(own + organizer))
    structure_features = np.nan_to_num(frame[needed["structure"]].to_numpy(dtype=np.float32)[0])
    atoms, edges, edge_features = molecule_graph(
        str(row["rdkit_canonical_smiles"]),
        cid,
        [float(value) for value in json.loads(row["rdkit_gasteiger_atom_charges"])],
        [float(value) for value in json.loads(row["rdkit_gasteiger_implicit_hydrogen_charges"])],
        load_conformers().get(cid),
    )
    item = {
        "atoms": torch.from_numpy(atoms),
        "edges": torch.from_numpy(edges),
        "edge_features": torch.from_numpy(edge_features),
        "dragon": torch.from_numpy(dragon[0]),
        "numeric": torch.from_numpy(numeric[0]),
        "name_features": torch.tensor(name_features, dtype=torch.float32),
        "structure_features": torch.tensor(structure_features, dtype=torch.float32),
        "targets": torch.zeros(len(checkpoint["target_columns"])),
        "cid": cid,
    }
    if checkpoint.get("neighbours"):
        item["neighbour_features"] = torch.tensor(neighbour_input(cid, checkpoint))
    return item


def predict(
    query: str,
    artifacts: dict,
    tables: tuple[pd.DataFrame, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Predict the mean and the spread of the perceived odor of one DREAM molecule."""
    checkpoint = artifacts["checkpoint"]
    molecules, fingerprints, *extra = tables if tables is not None else load_tables()
    community = extra[0] if extra else (load_community_table() if tables is None else None)
    row = find_molecule(query, molecules)

    batch = collate_molecules([build_inputs(row, fingerprints, artifacts, community)])
    device = artifacts["device"]

    def forward(model: GraphDescriptorModel, standard: bool = False) -> np.ndarray:
        numeric = batch["numeric"].to(device)
        if standard:
            numeric = numeric.clone()
            numeric[:, -2] = checkpoint["intensity_standard_log10_dilution"]
            numeric[:, -1] = 1.0 - numeric[:, -1]
        with torch.no_grad():
            return (
                model(
                    batch["atoms"].to(device),
                    batch["edges"].to(device),
                    batch["edge_features"].to(device),
                    batch["graph_ids"].to(device),
                    batch["dragon"].to(device),
                    numeric,
                    batch["name_features"].to(device),
                    batch["structure_features"].to(device),
                    batch["neighbour_features"].to(device)
                    if "neighbour_features" in batch
                    else None,
                )
                .cpu()
                .numpy()
            )

    target_names = checkpoint.get("target_names", TARGETS)
    count = len(target_names)
    group_columns = checkpoint.get("group_columns", {})
    group_models = artifacts.get("group_models", {})
    curve = checkpoint.get("std_curve")
    curve_keys = ("k0", "k1", "center", "shrink")

    def answer(model: GraphDescriptorModel) -> np.ndarray:
        """What one saved copy of the model answers, the way the model is scored.

        Intensity is learned at one dilution, so it is predicted at that dilution. The curve of the
        spread is read off the means of this same copy, and a score group then takes its columns
        from the copy that was kept for it.
        """
        scaled = forward(model)
        if checkpoint.get("intensity_standard_log10_dilution") is not None:
            columns = [0, count]
            scaled[:, columns] = forward(model, standard=True)[:, columns]
        restored = inverse_target_transform(scaled, artifacts["target_scaler"])
        if curve:
            restored = apply_std_curve(
                restored, {key: curve[key] for key in curve_keys if key in curve}, curve["weight"]
            )
        return restored[0]

    restored = answer(artifacts["model"])
    for name, group_model in group_models.items():
        columns = group_columns[name]
        restored[columns] = answer(group_model)[columns]

    means = np.round(restored[:count].astype(np.float64), 2)
    sigmas = np.round(restored[count : 2 * count].astype(np.float64), 2)
    return pd.DataFrame({"attribute": target_names, "mean": means, "sigma": sigmas})


def describe_molecule_label(smiles: str) -> str:
    """Human readable form of the input, canonicalised where possible."""
    if smiles.strip().isdigit():
        return f"CID {smiles.strip()}"
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return smiles
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict the odor profile of a DREAM molecule from its SMILES string or CID."
    )
    parser.add_argument(
        "--smiles",
        action="append",
        dest="smiles",
        help="SMILES string or CID to score. Repeat the flag to score several molecules.",
    )
    parser.add_argument(
        "--smiles-file",
        type=Path,
        help="Optional file with one SMILES string or CID per line.",
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
            "No input given. Pass --smiles with a SMILES string or CID, or --smiles-file. "
            f'Example: --smiles "{PREDICTION_EXAMPLE}"'
        )
    return values


def main() -> None:
    args = build_parser().parse_args()
    smiles_values = collect_smiles(args)

    device = torch.device(args.device) if args.device else None
    artifacts = load_model(args.model_dir, device)
    # The community features are part of what the model was trained on, so they are loaded too.
    tables = (*load_tables(), load_community_table())

    results = {}
    for smiles in smiles_values:
        try:
            table = predict(smiles, artifacts, tables)
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
