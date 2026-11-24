"""``ds-build``: build every file that the odor models read, in order.

1. ``molecules.csv`` and ``odor_tokens.csv``: the odor tokens of every molecule, in one vocabulary
   (``odor_tokens``), and ``dropped.csv``, the molecules that were left out and why.
2. ``structures.csv``, ``odor_dataset.csv`` and ``vocabulary.json``: the tables of the two training stages
   and the tokens of the model (``odor_tables``).
3. ``pubchem_structures.csv``: many more molecules for the pretraining, from PubChem
   (``pubchem_structures``). It is built before the tables, which take it in.
4. ``stage1_tokens.npy`` and its companions: the pretraining molecules as token numbers, packed once so
   that millions of them cost no memory to train on (``molecule_tokens``). It is built after the tables.

Every step is its own module with its own ``main()``, so one can be run alone with ``python -m``. This
command runs all of them, so that the files are always complete and in step with each other."""

from __future__ import annotations

from scripts.preprocessing import molecule_tokens, odor_tables, odor_tokens, pubchem_structures


def main() -> None:
    odor_tokens.main()
    pubchem_structures.main([])  # no options: the defaults, not whatever this command was given
    odor_tables.main()
    molecule_tokens.main()


if __name__ == "__main__":
    main()
