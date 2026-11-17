"""A step of ``ds-build``: the odor words of every molecule, from all the sources, in one vocabulary.

Each source lists the words that a perfumery or flavor reference gives to a molecule, in its own
spelling: ``fruity``, ``Fruity aroma??``, ``['fruity']``, ``Fragrant/Fruity``. This step

1. reads every source in ``SOURCES``, each described by a few lines (to add a source, add a line, and its
   files to ``ODOR_SOURCES`` in ``scripts/data/prepare_data.py``),
2. gives every molecule its PubChem CID and its structure, which is the key. A mirror image has its own CID,
   and it can smell different (carvone), so stereochemistry is kept. What a source leaves out (a CID, or a
   structure) is written to a list while the sources are read, and PubChem is asked for all of it at once
   (``scripts/data/odor_pubchem.py``),
3. normalizes every word: lower case, split at ``; , / |``, filler words removed, each word reduced to its
   lemma (``flowers`` -> ``flower``), and the synonyms of ``SYNONYMS`` merged,
4. writes one yes-or-no table: a molecule has a word or it does not.

    molecules.csv    cid, smiles (isomeric, canonical), sources (which references list it), words (; between)
    words.csv        word, molecules, sources, and the spellings that were merged into it
    dropped.csv      every molecule that was left out, with its source, its name or CAS number, and why

The graded Dravnieks ratings are not in this table. They are a test of the word probabilities, not
training labels.
"""

from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from simplemma import lemmatize

from scripts.data.odor_pubchem import PubChem
from scripts.data.prepare_data import ODOR_SOURCES, PYRFUME_DIR

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ODOR_DIR = PROJECT_ROOT / "data/processed/odor"
MOLECULES_PATH = ODOR_DIR / "molecules.csv"
WORDS_PATH = ODOR_DIR / "words.csv"
DROPPED_PATH = ODOR_DIR / "dropped.csv"
SEPARATORS = r"[;,/|()]"


@dataclass(frozen=True)
class Source:
    """How one source writes its words and its molecules. Every file is in ``data/raw/pyrfume-data/<name>/``."""

    behavior: str = "behavior.csv"  # the file with the words
    form: str = (
        "text"  # text: words in a column, split at ; , / |. list: a Python list written as text.
    )
    # binary: one 0 or 1 column for each word.
    words: tuple[str, ...] = ()  # the columns that hold words (not for binary)
    cid: str | None = (
        "CID"  # the column of stimuli.csv with the CID. None: the stimulus is the CID.
    )
    smiles: str | None = None  # the column of stimuli.csv with a structure, when it has one
    cas: str | None = None  # the column of stimuli.csv with the CAS number
    name: str | None = None  # the column of stimuli.csv with the chemical name
    exclude: frozenset[str] = frozenset()  # words of this source that are not used


SOURCES: dict[str, Source] = {
    "leffingwell": Source(form="binary", smiles="IsomericSMILES"),
    "goodscents": Source(words=("Descriptors",)),
    "arctander_1960": Source(
        behavior="behavior_1.csv", form="binary", cid="new_CID", cas="CAS", name="ChemicalName"
    ),
    # FlavorDB is downloaded but not used for now. Only 525 of its molecules have odor words, and the rest
    # have flavor words, most of them basic tastes ("sweet" would be given to four molecules in five).
    # To use its odor and flavor words as odor words, remove the comment marks. The five basic tastes
    # are left out, because they are not smells.
    # "flavordb": Source(
    #     words=("Odor Percepts", "Flavor Percepts"),
    #     exclude=frozenset({"sweet", "bitter", "sour", "salty", "umami"}),
    # ),
    "sharma_2021a": Source(words=("Decriptors",)),
    # Only the sub odor. The primary odor is a coarse category that is given to nearly every molecule
    # (a third of them are "lemon"), so it says nothing.
    "sharma_2021b": Source(behavior="behavior_1.csv", words=("Sub Odor",)),
    "ifra_2019": Source(words=("Descriptor 1", "Descriptor 2", "Descriptor 3")),
    "aromadb": Source(words=("Filtered Descriptors",)),
    "sigma_2014": Source(form="list", words=("descriptors",), cid=None, cas="CAS"),
    "flavornet": Source(words=("Descriptors",)),
}

# Words that say nothing about what a molecule smells like.
FILLER_WORDS = frozenset(
    {
        "",
        "nan",
        "none",
        "odorless",
        "odourless",
        "odor",
        "odour",
        "aroma",
        "aromatic note",
        "characteristic",
        "note",
        "like",
        "type",
        "slight",
        "faint",
        "weak",
        "strong",
        "mild",
        "very",
        "pleasant",
        "unpleasant",
        "bland",
        "other",
        "no",
        "wee",
        "natural",
        "cook",
        "black",
        "brown",
        "water",
        "limonene",
        # words that FlavorDB's scraped descriptions leave behind
        "can",
        "low",
        "high",
        "used",
        "food",
    }
)
# Words that are dropped from the end or the middle of a phrase: ``nutty aroma`` is ``nutty``.
FILLER_TOKENS = frozenset(
    {"aroma", "odor", "odour", "note", "like", "type", "slight", "faint", "very"}
)
# Spellings that the lemma does not join. Reviewed by hand against the most frequent words.
SYNONYMS: dict[str, str] = {
    "sulphurous": "sulfurous",
    "mouldy": "musty",
    "camphoreous": "camphoraceous",
    "camphor": "camphoraceous",
    "caramelic": "caramel",
    "caramellic": "caramel",
    "jasmin": "jasmine",
    "smokey": "smoky",
    "musky": "musk",
    "burn": "burnt",
    "wax": "waxy",
    "herb": "herbal",
    "ferment": "fermented",
    "roast": "roasted",
    "ambre": "amber",
    "mint": "minty",
    "fruit": "fruity",
    "flower": "floral",
    "spice": "spicy",
    "acid": "acidic",
    "alcohol": "alcoholic",
    "wood": "woody",
    "terpene": "terpenic",
    "wine": "winey",
}


@dataclass
class Entry:
    """One molecule of one source, as the source gives it."""

    source: str
    words: list[str]
    cid: int | None = None
    smiles: str | None = None
    cas: str | None = None
    name: str | None = None
    asked: tuple[str, str] | None = None  # the (kind, query) that PubChem is asked to find the CID
    stimulus: object = None  # how the source names it
    reason: str = ""  # why the molecule is left out, when it is


def read_words(row: pd.Series, spec: Source) -> list[str]:
    if spec.form == "binary":
        return [column for column, value in row.items() if column != "Stimulus" and value > 0]
    found: list[str] = []
    for column in spec.words:
        text = row[column]
        if pd.isna(text):
            continue
        if spec.form == "list":
            found += ast.literal_eval(text)
        else:
            found += re.split(SEPARATORS, str(text))
    return found


def read_source(name: str, spec: Source) -> list[Entry]:
    """The molecules of a source with their raw words, one entry for each stimulus."""
    behavior = pd.read_csv(PYRFUME_DIR / name / spec.behavior)
    stimuli = pd.read_csv(PYRFUME_DIR / name / "stimuli.csv").drop_duplicates("Stimulus")
    stimuli = stimuli.set_index("Stimulus")
    words: dict[object, list[str]] = defaultdict(list)
    for _, row in behavior.iterrows():
        words[row["Stimulus"]] += read_words(row, spec)

    def text(row: pd.Series, column: str | None) -> str | None:
        value = row[column] if column and column in row.index else None
        return value.strip() if isinstance(value, str) and value.strip() else None

    entries = []
    for stimulus, found in words.items():
        row = stimuli.loc[stimulus] if stimulus in stimuli.index else pd.Series(dtype=object)
        cid = stimulus if spec.cid is None else row.get(spec.cid)
        cid = int(cid) if pd.notna(cid) and cid > 0 else None
        found = [word for word in found if normalize_word(word) not in spec.exclude]
        entries.append(
            Entry(
                source=name,
                words=found,
                stimulus=stimulus,
                cid=cid,
                smiles=text(row, spec.smiles),
                cas=text(row, spec.cas),
                name=text(row, spec.name),
            )
        )
    return entries


def known_structures() -> dict[int, str]:
    """The structure of every CID that a downloaded source's own molecule file gives, FlavorDB's included."""
    structures: dict[int, str] = {}
    for name in [*ODOR_SOURCES, "molecules"]:
        path = PYRFUME_DIR / name / "molecules.csv"
        table = pd.read_csv(path) if path.is_file() else pd.DataFrame()
        if "IsomericSMILES" in table.columns:
            table = table.dropna(subset=["IsomericSMILES"])
            structures.update(zip(table["CID"].astype(int), table["IsomericSMILES"], strict=True))
    return structures


def complete(entries: list[Entry], pubchem: PubChem) -> list[Entry]:
    """Give every entry a CID and a structure, asking PubChem for what the sources leave out.

    The API follows from what is known: a CID without a structure is asked for its structure, a
    structure without a CID for its CID, a CAS number or a name likewise. An entry that PubChem cannot
    place is dropped, because the CID is the key.
    """
    structures = known_structures()
    for entry in entries:
        if entry.cid is not None:
            if entry.smiles is None and entry.cid in structures:
                entry.smiles = structures[entry.cid]
            if entry.smiles is None:
                pubchem.need("cid", entry.cid)
            continue
        for kind, query in (("smiles", entry.smiles), ("cas", entry.cas), ("name", entry.name)):
            if query:
                entry.asked = (kind, query)
                pubchem.need(kind, query)
                break
    pubchem.resolve()

    taken = {entry.cid for entry in entries if entry.cid is not None}
    complete_entries = []
    for entry in entries:
        if entry.cid is None and entry.asked:
            entry.cid = pubchem.get(*entry.asked)[0]
        if entry.cid is None and entry.asked:
            entry.cid = unspecified_relative(pubchem.relatives_of(*entry.asked), taken, pubchem)
        if entry.cid is not None:
            taken.add(entry.cid)
        if entry.cid is not None and entry.smiles is None:
            entry.smiles = pubchem.structure(entry.cid)
        if entry.cid is None:
            entry.reason = "PubChem has no CID for it"
        elif not entry.smiles:
            entry.reason = "PubChem has no structure for its CID"
        else:
            complete_entries.append(entry)
    return complete_entries


def unspecified_relative(relatives: list[int], taken: set[int], pubchem: PubChem) -> int | None:
    """The CID for a name that says nothing about stereochemistry, from the CIDs of the same connectivity.

    If any of them is already a molecule of the table, the name is taken to be that one (it has its own
    words) and no CID is added, so that a molecule never appears twice. Otherwise the CID that PubChem
    records without stereochemistry is taken, and none if there is no such CID.
    """
    if not relatives or any(cid in taken for cid in relatives):
        return None
    for cid in relatives:
        smiles = pubchem.structure(cid)
        if smiles and "@" not in smiles and "/" not in smiles and "\\" not in smiles:
            return cid
    return None


def canonical_smiles(smiles: object) -> str | None:
    """The canonical SMILES of a molecule, with its stereochemistry. None if RDKit cannot read it."""
    if not isinstance(smiles, str):
        return None
    molecule = Chem.MolFromSmiles(smiles)
    return None if molecule is None else Chem.MolToSmiles(molecule)


def normalize_word(raw: str) -> str:
    """One word in its lemma form, or an empty string when it says nothing about the smell."""
    text = re.sub(r"[^a-z\- ]", " ", str(raw).lower())
    tokens = [lemmatize(token, lang="en").lower() for token in re.split(r"[\s\-]+", text) if token]
    tokens = [token for token in tokens if token not in FILLER_TOKENS]
    word = " ".join(tokens)
    word = SYNONYMS.get(word, word)
    return "" if word in FILLER_WORDS else word


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pubchem = PubChem()
    entries = [e for name, spec in SOURCES.items() for e in read_source(name, spec)]
    # a molecule that a source lists without an odor word (FlavorDB has many) is of no use here
    entries = [e for e in entries if any(normalize_word(word) for word in e.words)]
    print(f"{len(entries)} molecules with odor words in {len(SOURCES)} sources.")
    kept = complete(entries, pubchem)

    smiles_of: dict[int, str] = {}
    molecule_words: dict[int, set[str]] = defaultdict(set)
    molecule_sources: dict[int, set[str]] = defaultdict(set)
    word_sources: dict[str, set[str]] = defaultdict(set)
    spellings: dict[str, Counter] = defaultdict(Counter)
    for entry in kept:
        smiles = canonical_smiles(entry.smiles)
        if smiles is None or entry.cid is None:
            entry.reason = "RDKit cannot read the structure"
            continue
        for raw in entry.words:
            word = normalize_word(raw)
            if word:
                smiles_of[entry.cid] = smiles
                molecule_words[entry.cid].add(word)
                molecule_sources[entry.cid].add(entry.source)
                word_sources[word].add(entry.source)
                spellings[word][str(raw).strip().lower()] += 1

    cids = sorted(molecule_words)
    molecules = pd.DataFrame(
        {
            "cid": cids,
            "smiles": [smiles_of[c] for c in cids],
            "sources": [";".join(sorted(molecule_sources[c])) for c in cids],
            "words": [";".join(sorted(molecule_words[c])) for c in cids],
        }
    )
    counts = Counter(word for found in molecule_words.values() for word in found)
    words = pd.DataFrame(
        {
            "word": list(counts),
            "molecules": list(counts.values()),
            "sources": [len(word_sources[w]) for w in counts],
            "spellings": [";".join(s for s, _ in spellings[w].most_common(6)) for w in counts],
        }
    ).sort_values(["molecules", "word"], ascending=[False, True], ignore_index=True)
    dropped = pd.DataFrame(
        [
            {
                "source": e.source,
                "stimulus": e.stimulus,
                "name": e.name,
                "cas": e.cas,
                "smiles": e.smiles,
                "cid": e.cid,
                "words": ";".join(sorted({normalize_word(w) for w in e.words} - {""})),
                "reason": e.reason,
            }
            for e in entries
            if e.reason
        ],
        columns=["source", "stimulus", "name", "cas", "smiles", "cid", "words", "reason"],
    ).sort_values(["reason", "source", "stimulus"], ignore_index=True)
    return molecules, words, dropped


def main() -> None:
    molecules, words, dropped = build_tables()
    ODOR_DIR.mkdir(parents=True, exist_ok=True)
    molecules.to_csv(MOLECULES_PATH, index=False)
    words.to_csv(WORDS_PATH, index=False)
    dropped.to_csv(DROPPED_PATH, index=False)
    print(f"Saved {len(molecules)} molecules with odor words to {MOLECULES_PATH}")
    print(f"Saved {len(words)} words to {WORDS_PATH}")
    print(f"Left out {len(dropped)} molecules, listed in {DROPPED_PATH}:")
    print(dropped["reason"].value_counts().to_string())
    print(f"{int((words['molecules'] >= 30).sum())} words have at least 30 molecules")


if __name__ == "__main__":
    main()
