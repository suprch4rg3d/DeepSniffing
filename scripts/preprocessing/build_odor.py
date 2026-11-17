"""``ds-build``: build every file that the odor models read, in order.

1. ``molecules.csv`` and ``words.csv``: the odor words of every molecule, in one vocabulary
   (``odor_words``), and ``dropped.csv``, the molecules that were left out and why.
2. ``structures.csv``, ``odor_dataset.csv`` and ``vocabulary.json``: the tables of the two training stages
   and the tokens of the model (``odor_tables``).
3. ``selfies_variants.csv``: verified other writings of the same molecules (``odor_variants``).

Every step is its own module with its own ``main()``, so one can be run alone with ``python -m``. This
command runs all of them, so that the files are always complete and in step with each other. The files of the
DREAM models (Subchallenge 1 and 2) are built by ``scripts.preprocessing.enrich_molecules``, which is not
part of this command for now."""

from __future__ import annotations

from scripts.preprocessing import odor_tables, odor_variants, odor_words


def main() -> None:
    odor_words.main()
    odor_tables.main()
    odor_variants.main()


if __name__ == "__main__":
    main()
