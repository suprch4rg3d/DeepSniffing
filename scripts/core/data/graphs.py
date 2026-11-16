"""Molecule graphs and batches: a molecule becomes atoms, bonds and, where PubChem has a 3D structure,
distances between atoms."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset

from scripts.core.data.scaling import name_feature_columns, structure_feature_columns
from scripts.preprocessing.conformers import OUTPUT as CONFORMERS_PATH

ATOMIC_NUMBERS = (1, 5, 6, 7, 8, 9, 11, 15, 16, 17, 35, 53)


ATOM_FEATURE_COUNT = len(ATOMIC_NUMBERS) + 1 + 7 + 6 + 6 + 1 + 6 + 4 + 2


BOND_FEATURE_COUNT = 4 + 2 + 6


DISTANCE_CUTOFF = 6.0


DISTANCE_RBF_COUNT = 16


EDGE_FEATURE_COUNT = BOND_FEATURE_COUNT + DISTANCE_RBF_COUNT


GRAPH_CACHE: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def one_hot(value: int, categories: tuple[int, ...]) -> list[float]:
    return [float(value == category) for category in categories] + [float(value not in categories)]


@functools.lru_cache(maxsize=2)
def _read_conformers(path: str, modified: int) -> dict[int, np.ndarray]:
    table = pd.read_csv(path)
    poses = {}
    for cid, atoms in table.groupby("CID"):
        atoms = atoms.sort_values("atom_index")
        poses[int(cid)] = atoms[["x", "y", "z"]].to_numpy(dtype=np.float64)
    return poses


def load_conformers(path: Path | None = None) -> dict[int, np.ndarray]:
    """PubChem's 3D structure of each molecule that has one, in the atom order of its SMILES.

    A molecule that is not in the table has no 3D structure, and its graph then has the bonds
    only. The table is required, because without it no molecule would have a 3D structure.
    """
    path = path or CONFORMERS_PATH
    if not path.is_file():
        raise FileNotFoundError(f"The 3D structures are missing: {path}. Run ds-build.")
    return _read_conformers(str(path), path.stat().st_mtime_ns)


def molecule_graph(
    smiles: str,
    cid: int,
    atom_charges: list[float],
    implicit_hydrogen_charges: list[float],
    coordinates: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build atom features, directed edges, and edge features from a 3D structure.

    ``coordinates`` are the heavy atom positions in the atom order of ``smiles``, which
    ``ds-build`` takes from PubChem. Without them the graph has the bonds only, and
    every distance feature is zero."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"CID {cid} has invalid RDKit SMILES: {smiles!r}")
    atom_total = molecule.GetNumAtoms()
    if len(atom_charges) != atom_total or len(implicit_hydrogen_charges) != atom_total:
        raise ValueError(f"CID {cid} has charge vectors that do not match its atom count.")

    if coordinates is not None:
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if coordinates.shape != (atom_total, 3) or not np.isfinite(coordinates).all():
            raise ValueError(
                f"CID {cid} has 3D coordinates that do not fit its {atom_total} atoms."
            )

    hybridizations = (
        int(Chem.HybridizationType.SP),
        int(Chem.HybridizationType.SP2),
        int(Chem.HybridizationType.SP3),
        int(Chem.HybridizationType.SP3D),
        int(Chem.HybridizationType.SP3D2),
    )
    atom_rows = []
    for atom_index, atom in enumerate(molecule.GetAtoms()):
        atom_rows.append(
            one_hot(atom.GetAtomicNum(), ATOMIC_NUMBERS)
            + one_hot(atom.GetTotalDegree(), (0, 1, 2, 3, 4, 5))
            + one_hot(max(-2, min(2, atom.GetFormalCharge())) + 2, (0, 1, 2, 3, 4))
            + one_hot(int(atom.GetHybridization()), hybridizations)
            + [float(atom.GetIsAromatic())]
            + one_hot(atom.GetTotalNumHs(includeNeighbors=True), (0, 1, 2, 3, 4))
            + one_hot(int(atom.GetChiralTag()), (0, 1, 2))
            + [atom_charges[atom_index], implicit_hydrogen_charges[atom_index]]
        )

    edge_pairs: list[tuple[int, int]] = []
    edge_rows: list[list[float]] = []
    bond_features: dict[tuple[int, int], list[float]] = {}
    bond_types = (
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    )
    for bond in molecule.GetBonds():
        bond_feature = [float(bond.GetBondType() == kind) for kind in bond_types]
        bond_feature += [float(bond.GetIsConjugated()), float(bond.IsInRing())]
        bond_feature += one_hot(int(bond.GetStereo()), (0, 1, 2, 3, 4))
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_features[(left, right)] = bond_feature
        bond_features[(right, left)] = bond_feature

    centers = np.linspace(0.0, DISTANCE_CUTOFF, DISTANCE_RBF_COUNT)
    rbf_width = DISTANCE_CUTOFF / (DISTANCE_RBF_COUNT - 1)
    if coordinates is None:
        # No 3D structure: the bonds are all that is known. Their distance features are zero,
        # which says "no distance", where a made-up position would give every pair 0 angstrom.
        for (left, right), bond_feature in bond_features.items():
            edge_pairs.append((left, right))
            edge_rows.append(bond_feature + [0.0] * DISTANCE_RBF_COUNT)
    else:
        for left in range(atom_total):
            for right in range(atom_total):
                if left == right:
                    continue
                distance = float(np.linalg.norm(coordinates[left] - coordinates[right]))
                if distance > DISTANCE_CUTOFF:
                    continue
                radial_features = np.exp(-(((distance - centers) / rbf_width) ** 2))
                edge_pairs.append((left, right))
                edge_rows.append(
                    bond_features.get((left, right), [0.0] * BOND_FEATURE_COUNT)
                    + radial_features.tolist()
                )

    atoms = np.asarray(atom_rows, dtype=np.float32)
    edges = (
        np.asarray(edge_pairs, dtype=np.int64).T if edge_pairs else np.empty((2, 0), dtype=np.int64)
    )
    edge_features = np.asarray(edge_rows, dtype=np.float32).reshape(-1, EDGE_FEATURE_COUNT)
    return atoms, edges, edge_features


class MoleculeRows(Dataset):
    """One item per rating row: its molecule graph, features, and scaled targets."""

    def __init__(
        self,
        rows: pd.DataFrame,
        dragon: np.ndarray,
        numeric: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray | None = None,
        names: np.ndarray | None = None,
        structure: np.ndarray | None = None,
        neighbours: np.ndarray | None = None,
    ):
        conformers = load_conformers()
        self.graphs = []
        for cid, smiles, atom_charges, hydrogen_charges in rows[
            [
                "CID",
                "rdkit_canonical_smiles",
                "rdkit_gasteiger_atom_charges",
                "rdkit_gasteiger_implicit_hydrogen_charges",
            ]
        ].itertuples(index=False, name=None):
            cid = int(cid)
            if cid not in GRAPH_CACHE:
                GRAPH_CACHE[cid] = molecule_graph(
                    str(smiles),
                    cid,
                    [float(value) for value in json.loads(atom_charges)],
                    [float(value) for value in json.loads(hydrogen_charges)],
                    conformers.get(cid),
                )
            self.graphs.append(GRAPH_CACHE[cid])
        self.dragon = torch.tensor(dragon, dtype=torch.float32)
        self.numeric = torch.tensor(numeric, dtype=torch.float32)
        self.name_columns = name_feature_columns(rows)
        if names is None:
            names = rows[self.name_columns].fillna(0.0).to_numpy(dtype=np.float32)
        self.name_features = torch.tensor(names, dtype=torch.float32)
        self.structure_columns = structure_feature_columns(rows)
        if structure is None:
            structure = rows[self.structure_columns].fillna(0.0).to_numpy(dtype=np.float32)
        self.structure_features = torch.tensor(structure, dtype=torch.float32)
        self.neighbour_features = (
            torch.tensor(neighbours, dtype=torch.float32) if neighbours is not None else None
        )
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.masks = (
            torch.tensor(masks, dtype=torch.bool)
            if masks is not None
            else torch.ones_like(self.targets, dtype=torch.bool)
        )
        self.cids = rows["CID"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        atoms, edges, edge_features = self.graphs[index]
        item = {
            "atoms": torch.from_numpy(atoms),
            "edges": torch.from_numpy(edges),
            "edge_features": torch.from_numpy(edge_features),
            "dragon": self.dragon[index],
            "numeric": self.numeric[index],
            "name_features": self.name_features[index],
            "structure_features": self.structure_features[index],
            "targets": self.targets[index],
            "mask": self.masks[index],
            "cid": int(self.cids[index]),
        }
        if self.neighbour_features is not None:
            item["neighbour_features"] = self.neighbour_features[index]
        return item


def collate_molecules(items: list[dict[str, torch.Tensor | int]]) -> dict[str, torch.Tensor]:
    """Merge molecule graphs into one disconnected batch graph."""
    atom_parts, edge_parts, edge_feature_parts, graph_ids = [], [], [], []
    offset = 0
    for graph_id, item in enumerate(items):
        atoms = item["atoms"]
        atom_parts.append(atoms)
        edge_parts.append(item["edges"] + offset)
        edge_feature_parts.append(item["edge_features"])
        graph_ids.append(torch.full((len(atoms),), graph_id, dtype=torch.long))
        offset += len(atoms)
    targets = torch.stack([item["targets"] for item in items])
    return {
        "atoms": torch.cat(atom_parts),
        "edges": torch.cat(edge_parts, dim=1),
        "edge_features": torch.cat(edge_feature_parts),
        "graph_ids": torch.cat(graph_ids),
        "dragon": torch.stack([item["dragon"] for item in items]),
        "numeric": torch.stack([item["numeric"] for item in items]),
        "name_features": torch.stack([item["name_features"] for item in items]),
        "structure_features": (
            torch.stack([item["structure_features"] for item in items])
            if "structure_features" in items[0]
            else torch.zeros((len(items), 0))
        ),
        **(
            {"neighbour_features": torch.stack([item["neighbour_features"] for item in items])}
            if "neighbour_features" in items[0]
            else {}
        ),
        "targets": targets,
        "mask": (
            torch.stack([item["mask"] for item in items])
            if "mask" in items[0]
            else torch.ones_like(targets, dtype=torch.bool)
        ),
        "cids": torch.tensor([item["cid"] for item in items], dtype=torch.long),
    }
