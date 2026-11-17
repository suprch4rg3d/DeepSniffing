"""Ask PubChem for what an odor source leaves out: the CID of a molecule, or the structure of a CID.

``scripts/preprocessing/odor_words.py`` combines the odor sources. Whatever a source lacks is written to a
``missing`` list while the sources are read, and ``PubChem.resolve`` then asks for all of it at once, with
the API that fits what is known:

- a CID without a structure: one request for 100 CIDs (PubChem answers lists of CIDs);
- a SMILES without a CID: one request for each, with an exact match, stereochemistry included;
- a CAS number or a name without a CID: one request for each (PubChem does not answer lists of names), and
  the first answer is taken. When PubChem does not know the name, two other services turn it into a
  structure: OPSIN (Cambridge, reads systematic chemical names) and the NCI Chemical Identifier Resolver
  (names and CAS numbers). That structure is then matched exactly in PubChem to find the CID.

Every answer is kept in one small table, so PubChem is asked once and a rerun needs no network. A molecule
that PubChem does not know is remembered with an empty answer.

    odor_pubchem.csv     kind (cid, smiles, cas or name), query, cid, smiles
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from rdkit import Chem, RDLogger

from scripts.data.prepare_data import PROJECT_ROOT, PUG_REST, RAW_DIR, RETRIES

CACHE_PATH = RAW_DIR / "pubchem/odor_pubchem.csv"
BATCH_SIZE = 100
WORKERS = 8  # requests that run at the same time
TIMEOUT_SECONDS = 20  # a server that says nothing for this long is given up on
EXACT = "identity_type=same_stereo_isotope"
CONNECTIVITY = "identity_type=same_connectivity"
MAX_RELATIVES = 50
RDLogger.DisableLog("rdApp.*")
PUG = PUG_REST.removesuffix("/cid")  # https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound
OPSIN = "https://opsin.ch.cam.ac.uk/opsin/{}.smi"  # names only
CIR = "https://cactus.nci.nih.gov/chemical/structure/{}/smiles"  # names and CAS numbers


def ask(url: str) -> dict | None:
    """The JSON that PubChem answers, or None when it has no answer for the query.

    A query that PubChem rejects or does not know (errors 400 and 404), or that keeps failing on its
    server (error 500), is left without an answer, so one odd name never stops the others.
    """
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code in (400, 404) or (error.code == 500 and attempt == RETRIES - 1):
                return None
            if attempt == RETRIES - 1:
                raise
        except urllib.error.URLError:
            if attempt == RETRIES - 1:
                raise
        time.sleep(2 * (attempt + 1))
    return None


def ask_text(url: str) -> str | None:
    """The first line that a service answers, or None when it does not know the query."""
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            lines = response.read().decode("utf-8", errors="replace").split()
    except (urllib.error.URLError, TimeoutError):
        return None
    return lines[0] if lines else None


def structure_from_name(kind: str, query: str) -> str | None:
    """A SMILES for a name or a CAS number from OPSIN and then the NCI resolver, or None."""
    text = urllib.parse.quote(query, safe="")
    services = [CIR] if kind == "cas" else [OPSIN, CIR]
    for service in services:
        smiles = ask_text(service.format(text))
        if smiles:
            return smiles
    return None


def find_cid(kind: str, query: str) -> tuple[int | None, list[int]]:
    """The CID of a molecule given by its ``smiles``, ``cas`` number or ``name``, and the CIDs of its relatives.

    The relatives come only when a structure found from a name matches no CID exactly. A name rarely
    says which mirror image is meant, so PubChem is asked for every CID with the same connectivity, and
    ``odor_words`` decides which of them, if any, the molecule is.
    """
    text = urllib.parse.quote(query, safe="")
    if kind == "smiles":
        return first_cid(ask(f"{PUG}/fastidentity/smiles/{text}/cids/JSON?{EXACT}")), []
    cid = first_cid(ask(f"{PUG}/name/{text}/cids/JSON"))
    if cid is not None:
        return cid, []
    smiles = structure_from_name(kind, query)
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None:
        return None, []
    # A name is about the compound, not one mirror image of it, so its structure is matched without
    # stereochemistry: PubChem then answers with the record that has none (the one of a plain name).
    text = urllib.parse.quote(Chem.MolToSmiles(molecule, isomericSmiles=False), safe="")
    cid = first_cid(ask(f"{PUG}/fastidentity/smiles/{text}/cids/JSON?{EXACT}"))
    if cid is not None:
        return cid, []
    return None, all_cids(ask(f"{PUG}/fastidentity/smiles/{text}/cids/JSON?{CONNECTIVITY}"))


def all_cids(payload: dict | None) -> list[int]:
    found = payload["IdentifierList"]["CID"] if payload else []
    return [int(cid) for cid in found if cid > 0][:MAX_RELATIVES]


def first_cid(payload: dict | None) -> int | None:
    found = payload["IdentifierList"]["CID"] if payload else []
    return int(found[0]) if found and found[0] > 0 else None


class PubChem:
    """The answers of PubChem for the odor sources, kept on disk in one table."""

    def __init__(self) -> None:
        self.answers: dict[tuple[str, str], tuple[int | None, str | None]] = {}
        self.relatives: dict[tuple[str, str], list[int]] = {}  # CIDs of the same connectivity
        self.missing: list[tuple[str, str]] = []  # (kind, query) that no answer is kept for
        self.asked = 0
        if CACHE_PATH.is_file():
            table = pd.read_csv(CACHE_PATH, dtype={"query": str})
            if "relatives" not in table.columns:
                table["relatives"] = None
            for kind, query, cid, smiles, relatives in table[
                ["kind", "query", "cid", "smiles", "relatives"]
            ].itertuples(index=False):
                self.answers[(kind, query)] = (
                    None if pd.isna(cid) else int(cid),
                    smiles if isinstance(smiles, str) else None,
                )
                if isinstance(relatives, str):
                    self.relatives[(kind, query)] = [int(c) for c in relatives.split()]

    def need(self, kind: str, query: object) -> None:
        """Write down that an answer is wanted. Nothing is asked until ``resolve``."""
        key = (kind, str(query))
        if key not in self.answers and key not in self.missing:
            self.missing.append(key)

    def get(self, kind: str, query: object) -> tuple[int | None, str | None]:
        """The (CID, SMILES) that PubChem gave for a query, or (None, None)."""
        return self.answers.get((kind, str(query)), (None, None))

    def resolve(self) -> None:
        """Ask PubChem for everything on the ``missing`` list, then keep the answers."""
        if self.missing:
            print(f"Asking PubChem for {len(self.missing)} molecules the sources leave incomplete.")
        by_cid = [q for k, q in self.missing if k == "cid"]
        for start in range(0, len(by_cid), BATCH_SIZE):
            batch = by_cid[start : start + BATCH_SIZE]
            payload = ask(f"{PUG}/cid/{','.join(batch)}/property/SMILES/JSON")
            found = {
                str(entry["CID"]): entry.get("SMILES")
                for entry in (payload["PropertyTable"]["Properties"] if payload else [])
            }
            for query in batch:
                self.answers[("cid", query)] = (int(query), found.get(query))
            self._pause()
        singles = [(k, q) for k, q in self.missing if k != "cid"]
        # The requests wait for slow servers, so several run at the same time.
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(find_cid, kind, query): (kind, query) for kind, query in singles}
            for index, future in enumerate(as_completed(futures), start=1):
                cid, relatives = future.result()
                self.answers[futures[future]] = (cid, None)
                self.relatives[futures[future]] = relatives
                self.asked += 1
                if index % 50 == 0 or index == len(singles):
                    self.save()  # an interrupted run keeps what it has asked
                    print(f"Looked up {index}/{len(singles)}")
        # the structures of the CIDs just found
        wanted = sorted(
            ({cid for cid, _ in self.answers.values() if cid} | self._relative_cids())
            - self._cids_with_structure()
        )
        for start in range(0, len(wanted), BATCH_SIZE):
            batch = wanted[start : start + BATCH_SIZE]
            payload = ask(f"{PUG}/cid/{','.join(map(str, batch))}/property/SMILES/JSON")
            for entry in payload["PropertyTable"]["Properties"] if payload else []:
                self.answers[("cid", str(entry["CID"]))] = (int(entry["CID"]), entry.get("SMILES"))
            self._pause()
        if self.missing or self.asked:
            self.missing.clear()
            self.save()

    def _cids_with_structure(self) -> set[int]:
        return {
            cid for (kind, _), (cid, smiles) in self.answers.items() if kind == "cid" and smiles
        }

    def _relative_cids(self) -> set[int]:
        return {cid for found in self.relatives.values() for cid in found}

    def relatives_of(self, kind: str, query: object) -> list[int]:
        """The CIDs with the connectivity of a structure found from a name, when no CID matched exactly."""
        return self.relatives.get((kind, str(query)), [])

    def structure(self, cid: int) -> str | None:
        """The SMILES that PubChem has for a CID, if it was asked for."""
        return self.answers.get(("cid", str(cid)), (None, None))[1]

    def _pause(self) -> None:
        self.asked += 1

    def save(self) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(
            [
                {
                    "kind": kind,
                    "query": query,
                    "cid": cid,
                    "smiles": smiles,
                    "relatives": " ".join(map(str, self.relatives.get((kind, query), []))),
                }
                for (kind, query), (cid, smiles) in sorted(self.answers.items())
            ],
            columns=["kind", "query", "cid", "smiles", "relatives"],
        )
        table["cid"] = table["cid"].astype("Int64")
        table.to_csv(CACHE_PATH, index=False)
        found = int(table["cid"].notna().sum())
        print(
            f"PubChem was asked {self.asked} times. {found} of {len(table)} answers have a CID; "
            f"kept in {CACHE_PATH.relative_to(PROJECT_ROOT)}"
        )
