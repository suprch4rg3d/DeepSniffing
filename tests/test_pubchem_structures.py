"""Tests for the pretraining molecules taken from PubChem."""

from __future__ import annotations

import numpy as np
import selfies as sf

from scripts.preprocessing import pubchem_structures as source


def archive(path, rows):
    """A stand-in for one of PubChem's unpacked tab-separated files."""
    path.write_text(
        "".join("\t".join(str(part) for part in row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def test_a_molecule_is_converted_to_canonical_smiles_and_selfies():
    cid, smiles, selfies = source.convert((7, "C(C)O"))
    assert cid == 7
    assert smiles == "CCO"  # canonical, whatever the source wrote
    assert sf.decoder(selfies) is not None
    assert "[C]" in selfies


def test_stereochemistry_is_kept():
    _, smiles, _ = source.convert((16724, "C=C(C)[C@H]1CC=C(C)C(=O)C1"))
    assert "@" in smiles  # a mirror image must stay its own molecule


def test_what_cannot_be_read_or_is_too_long_is_reported_by_reason():
    assert source.convert((1, "not a molecule")) == 1
    assert source.convert((2, "C" * (source.MAX_SMILES + 1))) == 0
    assert source.REASONS[0] == "too long"


def test_a_molecule_too_heavy_to_smell_is_left_out():
    assert source.convert((3, "C" * 40)) == 3  # a 40-carbon chain weighs some 560
    assert "heavy" in source.REASONS[3]


def test_nothing_is_left_out_for_the_elements_it_is_made_of():
    """The odor molecules themselves hold sodium and zinc, so the pretraining must read them."""
    for smiles in ("CSC", "[Na+].CC(=O)[O-]", "O=[Zn]", "B(O)(O)O"):
        assert not isinstance(source.convert((7, smiles)), int), smiles


def test_the_default_is_the_number_the_models_were_trained_on_not_the_whole_pool():
    """ds-build must not set off the conversion of 58.85 million molecules the later steps cannot read."""
    assert source.TARGET == 3_400_000
    parser_default = source.TARGET
    assert 0 < parser_default < 58_854_892


def test_ds_build_runs_every_step_in_order_and_packs_the_tokens(monkeypatch):
    from scripts.preprocessing import build_odor

    calls = []
    for name in ("odor_tokens", "pubchem_structures", "odor_tables", "molecule_tokens"):
        module = getattr(build_odor, name)
        monkeypatch.setattr(module, "main", lambda *a, _n=name: calls.append((_n, a)))
    build_odor.main()
    assert [name for name, _ in calls] == [
        "odor_tokens",
        "pubchem_structures",
        "odor_tables",
        "molecule_tokens",
    ]
    assert calls[1][1] == ([],)  # the PubChem step gets no options of this command's


def test_the_pieces_cover_every_line_once_and_start_where_a_line_starts(tmp_path):
    path = archive(tmp_path / "lines", [(cid, "C" * (cid % 7 + 1)) for cid in range(1, 400)])
    cuts = source.pieces(path, each=200)
    assert len(cuts) > 1  # it really was cut up
    assert cuts[0][0] == 0 and cuts[-1][1] == path.stat().st_size
    assert all(a[1] == b[0] for a, b in zip(cuts, cuts[1:], strict=False))  # no gap, no overlap
    seen = [line for start, end in cuts for line in source.lines_of(path, start, end)]
    assert seen == path.read_text(encoding="utf-8").splitlines()  # every line, once, in order


def test_a_piece_of_one_byte_boundary_still_reads_whole_lines(tmp_path):
    path = archive(tmp_path / "lines", [(1, "CCO"), (2, "CCN"), (3, "CCC")])
    for each in range(1, 40):
        seen = [
            line for a, b in source.pieces(path, each=each) for line in source.lines_of(path, a, b)
        ]
        assert seen == ["1\tCCO", "2\tCCN", "3\tCCC"], each


def test_the_mask_marks_the_light_compounds_by_their_cid(tmp_path):
    path = archive(
        tmp_path / "CID-Mass",
        [(1, "CH4", 16.03), (2, "C40H82", 562.6), (3, "C9H17NO4", 203.1), (9, "x", 350.0)],
    )
    mask = source.light_mask(path, limit=350.0, workers=1)
    assert [i for i in range(10) if mask[i]] == [1, 3, 9]  # the limit itself is kept
    assert mask.shape == (source.MAX_CID + 1,)


def test_a_weight_that_cannot_be_read_is_skipped(tmp_path):
    path = archive(tmp_path / "CID-Mass", [(1, "CH4", "not a number"), (2, "CH4", 16.0)])
    mask = source.light_mask(path, workers=1)
    assert not mask[1] and mask[2]


def saved_mask(tmp_path, cids):
    mask = np.zeros(source.MAX_CID + 1, dtype=bool)
    mask[list(cids)] = True
    path = tmp_path / "light.npy"
    np.save(path, mask)
    return path


def test_the_select_step_writes_the_light_molecules_and_does_no_chemistry(tmp_path):
    molecules = archive(
        tmp_path / "CID-SMILES",
        [(1, "CCO"), (2, "C" * 40), (3, "not a molecule"), (4, "CCN"), (5, "")],
    )
    mask_path = saved_mask(tmp_path, [1, 3, 4, 5])  # 2 is heavy, 5 has no SMILES
    source.SHARDS_DIR = tmp_path / "parts"
    light = tmp_path / "pubchem_light.csv"
    assert source.select(molecules, mask_path, light, workers=1) == 3
    # whatever the molecule is, this step only copies it: nothing is read by RDKit here
    assert light.read_text(encoding="utf-8").splitlines() == [
        "cid,smiles",
        "1,CCO",
        "3,not a molecule",
        "4,CCN",
    ]
    assert not source.SHARDS_DIR.exists()


def test_the_convert_step_reads_that_table_and_counts_every_reason(tmp_path):
    light = tmp_path / "pubchem_light.csv"
    light.write_text(
        "cid,smiles\n"
        "1,CCO\n"  # kept
        "2,CC(=O)O\n"  # kept
        "3,not a molecule\n"  # RDKit cannot read it
        f"4,{'C' * 40}\n"  # too heavy
        f"5,{'C' * (source.MAX_SMILES + 1)}\n",  # too long
        encoding="utf-8",
    )
    source.SHARDS_DIR = tmp_path / "parts"
    destination = tmp_path / "pubchem_structures.csv"
    kept, rejected, read = source.build(light, destination, workers=1)

    assert (kept, read) == (2, 5)
    assert len(rejected) == len(source.REASONS)
    assert sum(rejected) == 3
    written = destination.read_text(encoding="utf-8").splitlines()
    assert written[0] == "cid,smiles,selfies"
    assert len(written) == 3
    assert not destination.with_suffix(".tmp").exists()
    assert not source.SHARDS_DIR.exists()


def test_a_cutoff_takes_about_that_share_and_the_same_one_every_time(tmp_path):
    light = tmp_path / "pubchem_light.csv"
    light.write_text(
        "cid,smiles\n" + "".join(f"{cid},CCO\n" for cid in range(1, 2001)), encoding="utf-8"
    )
    source.SHARDS_DIR = tmp_path / "parts"
    half, _, _ = source.build(light, tmp_path / "half.csv", cutoff=source.ALL // 2, workers=1)
    assert 850 < half < 1150  # about a half of 2000, drawn at random, not exactly
    again, _, _ = source.build(light, tmp_path / "again.csv", cutoff=source.ALL // 2, workers=1)
    assert again == half  # the same sample every run
    everything, _, _ = source.build(light, tmp_path / "all.csv", workers=1)
    assert everything == 2000


def test_the_validation_molecules_do_not_depend_on_what_else_was_sampled(tmp_path):
    """The pretraining's yardstick: the same molecules in every build, so losses can be compared."""
    marked = [cid for cid in range(1, 200_001) if source.for_validation(cid)]
    assert 5 < len(marked) < 40  # about 5,000 in the pool of 58.85M, so a handful in 200k
    assert marked == [cid for cid in range(1, 200_001) if source.for_validation(cid)]
    # and a validation molecule is converted even when the sample would have passed it over
    light = tmp_path / "pubchem_light.csv"
    light.write_text(
        "cid,smiles\n" + "".join(f"{cid},CCO\n" for cid in marked[:3]), encoding="utf-8"
    )
    source.SHARDS_DIR = tmp_path / "parts"
    kept, _, _ = source.build(light, tmp_path / "out.csv", cutoff=1, workers=1)
    assert kept == 3  # a cutoff of 1 takes nothing, yet all three are here


def test_a_molecule_without_a_pubchem_id_still_gets_a_split():
    """``odor_tokens`` gives a molecule PubChem cannot identify a negative id, and those come here too."""
    for cid in (-955348933095, -1, 0):
        assert isinstance(source.for_validation(cid), bool)
        assert isinstance(source.chosen(cid, source.ALL // 2), bool)


def test_the_sample_is_not_tied_to_runs_of_consecutive_cids(tmp_path):
    """A batch of analogues sits on consecutive CIDs, so taking every n-th is not an even draw."""
    picked = [cid for cid in range(1, 10001) if source.chosen(cid, source.ALL // 10)]
    assert 800 < len(picked) < 1200
    gaps = {b - a for a, b in zip(picked, picked[1:], strict=False)}
    assert len(gaps) > 5  # not one fixed step, as a stride would give
