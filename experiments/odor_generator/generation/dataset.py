"""The molecules of the generator and their odor words.

Sources, all from the Pyrfume repository (pinned commit 8054ea98, files in ``data/raw``):
- ``molecules/molecules.csv``: about 10,000 odor and flavor molecules, with no odor words. They teach chemistry.
- ``leffingwell`` (3,522 molecules, 113 odor words) and ``goodscents`` (4,626 molecules with descriptor text),
  as cleaned by Sanchez-Lengeling et al. They give the odor words. A molecule that both list has the words
  of both, and the GoodScents words are matched to the 113 Leffingwell words.

Molecules are written without stereochemistry, so the decoder never has to invent a mirror image, and
a molecule is kept only if ``chemistry.clean_smiles`` accepts it and none of its SELFIES symbols is rare
(``MIN_SYMBOL_MOLECULES``). The DREAM molecules are held out completely: the generator and the oracle
never learn from them, and they are the hardest test of whether the generator follows an odor request.

Run:  poetry run python -m generation.dataset
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from .chemistry import clean_smiles, selfies_tokens
from .paths import DATASET, DREAM_MOLECULES, RAW, WORDS

MIN_WORD_MOLECULES = 30  # an odor word that fewer training molecules have is dropped
MIN_SYMBOL_MOLECULES = (
    3  # a SELFIES symbol that fewer molecules have is a data error, and its molecules are dropped
)
VALIDATION_FRACTION = TEST_FRACTION = 0.05
SEED = 42


def read_leffingwell() -> tuple[pd.DataFrame, list[str]]:
    behavior = pd.read_csv(RAW / "leffingwell/behavior.csv").set_index("Stimulus")
    stimuli = pd.read_csv(RAW / "leffingwell/stimuli.csv").set_index("Stimulus")
    words = list(behavior.columns)
    frame = pd.DataFrame({"smiles": stimuli.loc[behavior.index, "IsomericSMILES"].to_numpy()})
    return pd.concat([frame, behavior.reset_index(drop=True)], axis=1), words


def read_goodscents(words: list[str]) -> pd.DataFrame:
    behavior = pd.read_csv(RAW / "goodscents/behavior.csv").set_index("Stimulus")
    stimuli = (
        pd.read_csv(RAW / "goodscents/stimuli.csv")
        .drop_duplicates("Stimulus")
        .set_index("Stimulus")
    )
    molecules = (
        pd.read_csv(RAW / "goodscents/molecules.csv").drop_duplicates("CID").set_index("CID")
    )
    rows = []
    for stimulus, text in behavior["Descriptors"].items():
        if stimulus not in stimuli.index or pd.isna(stimuli.loc[stimulus, "CID"]):
            continue
        cid = int(stimuli.loc[stimulus, "CID"])
        if cid not in molecules.index:
            continue
        listed = set(str(text).split(";"))
        rows.append(
            {
                "smiles": molecules.loc[cid, "IsomericSMILES"],
                **{w: float(w in listed) for w in words},
            }
        )
    return pd.DataFrame(rows)


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    """The table of molecules (smiles, split, words, ``word_list``) and the list of odor words."""
    frame = pd.read_csv(DATASET).fillna({"words": ""})
    words = json.loads(WORDS.read_text(encoding="utf-8"))
    frame["word_list"] = frame["words"].map(lambda text: text.split(";") if text else [])
    return frame, words


def main() -> None:
    leffingwell, words = read_leffingwell()
    labeled = pd.concat([leffingwell, read_goodscents(words)], ignore_index=True)
    labeled["smiles"] = labeled["smiles"].map(clean_smiles)
    labeled = labeled.dropna(subset=["smiles"]).groupby("smiles", sort=False)[words].max()
    labeled = labeled[
        labeled.sum(axis=1) > 0
    ]  # a molecule whose sources list no word is not labeled

    registry = (
        pd.read_csv(RAW / "molecules/molecules.csv")["IsomericSMILES"].map(clean_smiles).dropna()
    )
    dream = pd.read_csv(DREAM_MOLECULES, usecols=["rdkit_canonical_smiles"])[
        "rdkit_canonical_smiles"
    ]
    dream_set = set(dream.map(clean_smiles).dropna())

    others = (set(registry) | set(labeled.index)) - dream_set
    symbol_molecules = Counter(s for smiles in others for s in set(selfies_tokens(smiles) or []))
    rare = {s for s, n in symbol_molecules.items() if n < MIN_SYMBOL_MOLECULES}
    dropped = {m for m in others if rare & set(selfies_tokens(m) or [])}
    print(f"{len(rare)} rare SELFIES symbols; {len(dropped)} molecules that use them are dropped")

    frame = pd.DataFrame({"smiles": sorted((others - dropped) | dream_set)})
    frame["dream"] = frame["smiles"].isin(dream_set)
    frame["labeled"] = frame["smiles"].isin(labeled.index)
    print(
        f"{len(frame)} molecules; {int(frame['labeled'].sum())} with odor words; "
        f"{int(frame['dream'].sum())} DREAM molecules held out"
    )

    # one random split of the labeled molecules that are not DREAM molecules
    rng = np.random.RandomState(SEED)
    split = np.array(["train"] * len(frame), dtype=object)
    free = rng.permutation(np.flatnonzero(~frame["dream"].to_numpy() & frame["labeled"].to_numpy()))
    validation, test = int(VALIDATION_FRACTION * len(free)), int(TEST_FRACTION * len(free))
    split[free[:validation]] = "validation"
    split[free[validation : validation + test]] = "test"
    split[frame["dream"].to_numpy()] = "dream"
    frame["split"] = split

    train_labels = labeled.loc[frame.loc[(frame["split"] == "train") & frame["labeled"], "smiles"]]
    counts = train_labels.sum()
    kept = [w for w in words if counts[w] >= MIN_WORD_MOLECULES]
    print(
        f"{len(kept)} of {len(words)} odor words have at least {MIN_WORD_MOLECULES} training molecules"
    )
    listed = labeled[kept].apply(lambda row: ";".join(w for w in kept if row[w] > 0), axis=1)
    frame["words"] = frame["smiles"].map(listed).fillna("")

    DATASET.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATASET, index=False)
    WORDS.write_text(json.dumps(kept), encoding="utf-8")
    print(frame["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
