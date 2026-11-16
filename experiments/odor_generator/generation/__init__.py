"""Odor-conditioned molecule generation: a GPT-style decoder, and the models that judge what it writes.

    dataset     the molecules and their odor words, and the splits
    vocabulary  the tokens of the decoder: special tokens, odor words and SELFIES symbols
    generator   the decoder: it reads odor words and writes a molecule symbol by symbol
    oracle      judge 1: the probability of every odor word for any molecule
    student     judge 2: a lightweight stand-in for Q2, the 21 DREAM ratings of any molecule
    judge       asks the generator for molecules and ranks them with both judges
    evaluate    how well the generator follows a request, on molecules it never saw
    checks      correctness checks of the sequences and of the decoder
    chemistry   RDKit and SELFIES helpers shared by all of them
    paths       every file location, in one place

Every module has a ``main()`` and is run as ``python -m generation.<module>`` from the folder that holds
``generation/``, ``data/`` and ``results/``. Nothing here imports the DREAM models. Only ``student`` reads
the DREAM ratings, through the project's own ``scripts.core`` helpers.
"""
