"""``ds-experiments``: reserved for the ablations of Subchallenge 1 and 2. It is not active yet.

The ablations that were run while the two models were built (input blocks left out, token dropout, the blend
weight, and so on) are recorded, with their scores, in ``Q1_SETUP.md`` and ``Q2_SETUP.md``. The command stays in
the list so that the pipeline keeps its place for them, and so that nobody runs an outdated matrix by accident.
"""

from __future__ import annotations


def main() -> None:
    print(
        "ds-experiments is not active yet. The ablations of Subchallenge 1 and 2 that were run are recorded "
        "in Q1_SETUP.md and Q2_SETUP.md."
    )


if __name__ == "__main__":
    main()
