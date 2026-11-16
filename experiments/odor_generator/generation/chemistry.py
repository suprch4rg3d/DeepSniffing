"""RDKit and SELFIES helpers shared by the whole package.

SMILES is the usual text form of a molecule (``CCOC(C)=O``). SELFIES is a second text form (``[C][C][O]``
and so on) in which every sequence of symbols decodes to a valid molecule, so a model that writes it can
never write an impossible one. A molecule is always rewritten as a stereo-free canonical SMILES first, so
that one molecule has one SELFIES string, and the conversion in both directions is exact.
"""

from __future__ import annotations

import random

import numpy as np
import selfies as sf
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")

SEED = 42
ALLOWED_ELEMENTS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}  # H C N O F P S Cl Br I
MIN_ATOMS, MAX_ATOMS = 3, 40
MORGAN_BITS = 2048


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_smiles(smiles: str) -> str | None:
    """A stereo-free canonical SMILES of one molecule, or None when the molecule is not wanted.

    Not wanted: mixtures and salts, isotope-labelled copies, elements outside ``ALLOWED_ELEMENTS``, and
    molecules with fewer than ``MIN_ATOMS`` or more than ``MAX_ATOMS`` heavy atoms.
    """
    if not isinstance(smiles, str) or "." in smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    if any(atom.GetIsotope() for atom in molecule.GetAtoms()):
        return None  # a deuterated copy of a molecule is not a different odorant
    molecule = Chem.RemoveHs(
        molecule
    )  # a few sources write hydrogens as atoms; here they are implicit
    if not MIN_ATOMS <= molecule.GetNumAtoms() <= MAX_ATOMS:
        return None
    if any(atom.GetAtomicNum() not in ALLOWED_ELEMENTS for atom in molecule.GetAtoms()):
        return None
    return Chem.MolToSmiles(molecule, isomericSmiles=False)


def selfies_tokens(smiles: str) -> list[str] | None:
    """The SELFIES symbols of a SMILES string, or None when it cannot be converted."""
    try:
        return list(sf.split_selfies(sf.encoder(smiles)))
    except Exception:
        return None


def selfies_to_smiles(tokens: list[str]) -> str | None:
    """The canonical SMILES of SELFIES symbols, or None when RDKit cannot read the result."""
    try:
        smiles = sf.decoder("".join(tokens))
    except Exception:
        return None
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(molecule, isomericSmiles=False)


def molecule_features(smiles_list: list[str]) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    """Morgan counts (radius 2) and RDKit descriptors of every molecule, and which could be read.

    These are what the two judges read. RDKit computes them for any molecule, which is why the judges can
    score molecules that the generator has just written.
    """
    counts, descriptors, readable = [], [], []
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            counts.append(np.zeros(MORGAN_BITS, dtype=np.float32))
            descriptors.append(np.zeros(len(Descriptors.descList)))
            readable.append(False)
            continue
        vector = np.zeros(MORGAN_BITS, dtype=np.float32)
        fingerprint = AllChem.GetHashedMorganFingerprint(molecule, 2, nBits=MORGAN_BITS)
        for position, value in fingerprint.GetNonzeroElements().items():
            vector[position] = value
        counts.append(np.log1p(vector))
        descriptors.append(np.array([function(molecule) for _, function in Descriptors.descList]))
        readable.append(True)
    return np.stack(counts), np.stack(descriptors), readable


def finite(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def morgan_bitvector(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    return (
        AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=MORGAN_BITS) if molecule else None
    )


def tanimoto(a, b) -> float:
    return float(DataStructs.TanimotoSimilarity(a, b))
