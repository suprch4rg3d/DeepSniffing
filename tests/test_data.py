"""Tests for the data contract."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.core.metrics import TARGET_COUNT, TARGETS
from scripts.core.splits import leaderboard_cids, official_test_cids
from scripts.core.train import Q1_BLEND_ALPHA


def write_label_table(path, rdkit_version):
    import pandas as pd

    pd.DataFrame(
        [{"CID": 1, "rdkit_canonical_smiles": "CCO", "rdkit_version": rdkit_version}]
    ).to_csv(path, index=False)


def test_enrichment_accepts_smiles_canonicalized_by_an_older_rdkit(tmp_path):
    import pandas as pd

    from scripts.preprocessing import enrich_molecules

    labels = tmp_path / "labels.csv"
    write_label_table(labels, next(iter(enrich_molecules.OLDER_RDKIT_VERSIONS)))

    result = enrich_molecules.add_rdkit_descriptors(pd.read_csv(labels))

    assert result.loc[0, "rdkit_atom_count_c"] == 2


def test_enrichment_rejects_an_unknown_rdkit_version(tmp_path):
    import pandas as pd

    from scripts.preprocessing import enrich_molecules

    labels = tmp_path / "labels.csv"
    write_label_table(labels, "1999.01.1")

    with pytest.raises(ValueError, match="RDKit version differs"):
        enrich_molecules.add_rdkit_descriptors(pd.read_csv(labels))


# Subchallenge 1 targets: individual ratings, blended training labels, masked scaling.
def rating_file(tmp_path):
    import pandas as pd

    rows = []
    for cid in (10, 20):
        for subject in (1, 2, 3):
            for repeat in (0, 1):
                row = {"Compound Identifier": cid, "Intensity": "high", "subject #": subject}
                row.update(dict.fromkeys(TARGETS, np.nan))
                row.update({"intensity_strength": 10.0 * subject + repeat})
                row.update({"valence_pleasantness": 50.0})
                rows.append(row)
    rows.append({"Compound Identifier": 10, "Intensity": "low", "subject #": 1})
    path = tmp_path / "ratings.txt"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def test_individual_ratings_are_averaged_per_participant(tmp_path):
    from scripts.core.data import aggregate_individual_ratings

    cids, ratings = aggregate_individual_ratings(rating_file(tmp_path))

    assert cids == [10, 20]
    assert ratings.shape == (2, TARGET_COUNT, 49)
    assert ratings[0, 0, 0] == 10.5 and ratings[0, 0, 2] == 30.5
    assert np.isnan(ratings[0, 0, 3])
    assert np.isnan(ratings[0, 5]).all()


def test_blend_moves_training_ratings_towards_the_population_mean():
    import pandas as pd

    from scripts.core.data import blend_q1_training_targets
    from scripts.core.metrics import TARGETS

    column = f"q1_{TARGETS[0]}_subject_1"
    columns = {f"mean_{target}": [50.0] for target in TARGETS}
    columns.update({f"q1_{t}_subject_{n}": [np.nan] for t in TARGETS for n in range(1, 50)})
    columns[column] = [90.0]
    rows = pd.DataFrame(columns)

    assert blend_q1_training_targets(rows, 1.0)[column][0] == 90.0
    assert blend_q1_training_targets(rows, 0.0)[column][0] == 50.0
    assert np.isclose(blend_q1_training_targets(rows, 0.25)[column][0], 60.0)
    assert np.isnan(blend_q1_training_targets(rows, 0.5)[f"q1_{TARGETS[1]}_subject_1"][0])
    with pytest.raises(ValueError, match="between 0 and 1"):
        blend_q1_training_targets(rows, 1.5)


def test_masked_target_scaler_uses_only_the_ratings_that_exist():
    import pandas as pd

    from scripts.core.data import fit_masked_target_scaler, scale_masked_targets

    training = pd.DataFrame({"a": [1.0, 3.0, np.nan], "b": [np.nan] * 3, "c": [5.0, 5.0, 5.0]})

    scaler = fit_masked_target_scaler(training, ["a", "b", "c"])
    scaled, mask = scale_masked_targets(training, ["a", "b", "c"], scaler)

    assert scaler.mean_[0] == 2.0 and scaler.scale_[0] == 1.0
    assert scaler.mean_[1] == 0.0 and scaler.scale_[1] == 1.0
    assert scaler.scale_[2] == 1.0
    assert mask.tolist() == [[True, False, True], [True, False, True], [False, False, True]]
    assert (scaled[~mask] == 0).all() and scaled[0, 0] == -1.0


def test_graph_preparation_validates_the_challenge_before_loading_data():
    from scripts.core.data import prepare_graph_data

    with pytest.raises(ValueError, match="Unknown challenge"):
        prepare_graph_data("q3")
    with pytest.raises(ValueError, match="blend alpha"):
        prepare_graph_data("q1")


@pytest.fixture(scope="module")
def graph_inputs():
    from scripts.core.data import load_graph_inputs

    return load_graph_inputs()


@pytest.mark.parametrize("challenge", ["q1", "q2"])
def test_graph_data_validates_on_the_leaderboard_and_refits_on_every_labelled_molecule(
    graph_inputs, challenge
):
    from scripts.core.data import prepare_graph_data

    data = prepare_graph_data(challenge, blend_alpha=Q1_BLEND_ALPHA, inputs=graph_inputs)

    assert len(data.train_cids) == 338
    assert set(data.validation_cids) == set(leaderboard_cids())
    assert set(data.full_cids) == set(data.train_cids) | set(data.validation_cids)
    assert len(data.full_cids) == 407
    assert set(data.test_cids) == set(official_test_cids())
    assert set(data.full_cids).isdisjoint(official_test_cids())


def test_q1_graph_data_trains_on_both_concentrations_and_gets_the_dilution(graph_inputs):
    from scripts.core.data import prepare_graph_data

    data = prepare_graph_data("q1", blend_alpha=Q1_BLEND_ALPHA, inputs=graph_inputs)
    training = data.train_loader.dataset

    assert 0.3 < float(training.numeric[:, -1].mean()) < 0.7  # high and low rows are both used
    assert float(training.numeric[:, -2].min()) < float(training.numeric[:, -2].max())
    for loader in (data.validation_loader, data.test_loader):
        # Every row is scored on its listed dilution, with the flag of its pair.
        numeric = loader.dataset.numeric
        assert (numeric[:, -1] == (numeric[:, -2] > -3.0).float()).all()


def test_graph_data_carries_the_community_features(graph_inputs):
    from scripts.core.data import prepare_graph_data
    from scripts.preprocessing.community_features import STRUCTURE_PREFIXES

    data = prepare_graph_data("q2", inputs=graph_inputs)
    dataset = data.train_loader.dataset

    # Every kept structure column is an input as it is.
    assert data.structure_count == len(dataset.structure_columns) > 1000
    assert {column.split("_")[0] + "_" for column in dataset.structure_columns} == set(
        STRUCTURE_PREFIXES
    )
    # Every kept name window is an input as it is.
    assert data.name_count == len(dataset.name_columns)
    # Our own windows come first, and the organizers' windows follow them.
    assert len(dataset.name_columns) > len(data.name_vocabulary)
    assert dataset.name_columns[: len(data.name_vocabulary)] == [
        column for column in dataset.name_columns if column.startswith("dream_odor_name_")
    ]
    assert any(column.startswith("orgname_") for column in dataset.name_columns)
    assert dataset.name_features.shape == (len(dataset), data.name_count)
    assert dataset.structure_features.shape == (len(dataset), data.structure_count)
    assert not dataset.structure_features.isnan().any()


def test_every_official_test_molecule_has_a_pubchem_3d_structure():
    from scripts.core.data import load_conformers

    poses = load_conformers()

    assert set(official_test_cids()) <= set(poses)


def test_only_a_few_molecules_have_no_pubchem_3d_structure():
    from scripts.core.data import load_conformers
    from scripts.core.splits import leaderboard_cids, training_pool_cids

    poses = load_conformers()
    labelled = set(training_pool_cids()) | set(leaderboard_cids())

    assert len(labelled - set(poses)) <= 5  # bonds only for these, by design
    assert all(len(pose) > 0 and pose.shape[1] == 3 for pose in poses.values())


def test_the_epoch_of_each_score_group_is_chosen_and_used_for_its_own_outputs(graph_inputs):
    from scripts.core.data import prepare_graph_data
    from scripts.core.train import TrainConfig, output_groups, train_q2

    data = prepare_graph_data("q2", inputs=graph_inputs)
    config = TrainConfig(epochs=3, patience=3, batch_size=32, learning_rate=0.0005)

    result = train_q2(config, graph_inputs, output_dir=None, device="cpu", verbose=False)
    score = result.validation_score

    groups = output_groups("q2")
    assert set(result.group_epochs) == set(groups) == set(result.group_states)
    assert all(1 <= epoch <= 3 for epoch in result.group_epochs.values())
    assert result.best_epoch == max(result.group_epochs.values())
    assert result.test_predictions.shape == (len(data.test_cids), TARGET_COUNT * 2)
    assert np.isfinite(result.test_predictions).all()
    assert np.isfinite(score)


def test_subchallenge_2_learns_intensity_at_the_standard_dilution_and_fits_the_curve(graph_inputs):
    from scripts.core.data import STANDARD_LOG10_DILUTION, prepare_graph_data

    data = prepare_graph_data("q2", inputs=graph_inputs)
    training = data.train_loader.dataset

    at_standard = (training.numeric[:, -2] - STANDARD_LOG10_DILUTION).abs() < 0.5
    intensity = training.masks[:, [0, TARGET_COUNT]]
    assert (intensity.any(dim=1) == at_standard).all()
    assert training.masks[:, 1:TARGET_COUNT].all()  # the other attributes keep every row
    assert data.std_curve["k0"].shape == data.std_curve["k1"].shape == (TARGET_COUNT,)


def test_subchallenge_1_has_no_curve_of_the_spread(graph_inputs):
    from scripts.core.data import prepare_graph_data

    assert (
        prepare_graph_data("q1", blend_alpha=Q1_BLEND_ALPHA, inputs=graph_inputs).std_curve is None
    )


def test_molecules_listed_at_1_100000_are_the_low_rating_of_their_pair(graph_inputs):
    from scripts.core.data import prepare_graph_data

    data = prepare_graph_data("q1", blend_alpha=Q1_BLEND_ALPHA, inputs=graph_inputs)

    for loader in (data.validation_loader, data.test_loader):
        numeric = loader.dataset.numeric
        listed = {round(float(value)) for value in numeric[:, -2]}
        assert listed == {-1, -5}  # the two listed dilutions of the held-out molecules
        assert (numeric[numeric[:, -2] == -5.0][:, -1] == 0.0).all()
        assert (numeric[numeric[:, -2] == -1.0][:, -1] == 1.0).all()


def test_the_nspdk_similarity_is_a_unit_similarity_between_the_dream_molecules():
    from scripts.core.data import load_dream_nspdk_similarity, load_dream_similarity

    nspdk = load_dream_nspdk_similarity()
    morgan = load_dream_similarity()

    assert nspdk.shape == morgan.shape == (476, 476)
    assert list(nspdk.index) == list(morgan.index) == list(nspdk.columns)
    # The published vectors have unit length up to the rounding of the file (the shortest is 0.994).
    assert np.allclose(np.diag(nspdk.to_numpy()), 1.0, atol=1e-2)
    assert np.allclose(nspdk.to_numpy(), nspdk.to_numpy().T, atol=1e-6)
    assert nspdk.to_numpy().min() >= -1e-6


def test_subchallenge_2_reads_the_neighbours_by_two_similarities_and_subchallenge_1_by_none(
    graph_inputs,
):
    from scripts.core.data import NEIGHBOUR_SIMILARITIES, prepare_graph_data

    q2 = prepare_graph_data("q2", inputs=graph_inputs)
    q1 = prepare_graph_data("q1", blend_alpha=Q1_BLEND_ALPHA, inputs=graph_inputs)

    assert q2.neighbours["width"] == 21 * len(NEIGHBOUR_SIMILARITIES) == 42
    assert q2.train_loader.dataset.neighbour_features.shape[1] == 42
    assert q2.validation_loader.dataset.neighbour_features.shape[1] == 42
    assert q1.neighbours is None
    assert q1.train_loader.dataset.neighbour_features is None


# ds-build: every scaler is fitted on training molecules only, and a stale file is refused.
def test_the_scalers_are_fitted_on_the_training_molecules_of_each_stage():
    from scripts.preprocessing.scalers import fit_scalers

    features, targets = fit_scalers()

    assert len(features["cids"]["validation"]) == 338
    assert len(features["cids"]["final"]) == 407
    assert set(features["cids"]["validation"]) < set(features["cids"]["final"])
    assert set(features["cids"]["final"]).isdisjoint(official_test_cids())
    assert (
        set(features["validation"])
        == set(features["final"])
        == {"dragon", "rdkit", "interaction", "epi"}
    )
    assert targets["q1"]["blend_alpha"] == Q1_BLEND_ALPHA and targets["q2"]["blend_alpha"] is None
    assert targets["q2"]["validation"].mean_.shape == (TARGET_COUNT * 2,)
    assert targets["q1"]["final"].mean_.shape == (TARGET_COUNT * 49,)


def test_scalers_that_do_not_match_the_run_are_refused(graph_inputs, tmp_path, monkeypatch):
    import joblib

    from scripts.core.data import load_scalers, training_frames
    from scripts.core.data import scaling as scaling_module
    from scripts.core.splits import load_split
    from scripts.preprocessing.scalers import fit_scalers

    split = load_split()
    frames = training_frames("q1", graph_inputs, split, Q1_BLEND_ALPHA)
    features, targets = fit_scalers()
    monkeypatch.setattr(scaling_module, "FEATURE_SCALERS_PATH", tmp_path / "features.joblib")
    monkeypatch.setattr(scaling_module, "TARGET_SCALERS_PATH", tmp_path / "targets.joblib")

    with pytest.raises(FileNotFoundError, match="ds-build"):
        load_scalers("q1", frames, graph_inputs.dragon_columns, Q1_BLEND_ALPHA)

    joblib.dump(features, tmp_path / "features.joblib")
    joblib.dump(targets, tmp_path / "targets.joblib")
    load_scalers("q1", frames, graph_inputs.dragon_columns, Q1_BLEND_ALPHA)  # these match

    with pytest.raises(ValueError, match="blend weight"):
        load_scalers("q1", frames, graph_inputs.dragon_columns, 0.5)

    targets["q1"]["cids"]["final"] = targets["q1"]["cids"]["final"][:-1]
    joblib.dump(targets, tmp_path / "targets.joblib")
    with pytest.raises(ValueError, match="other molecules"):
        load_scalers("q1", frames, graph_inputs.dragon_columns, Q1_BLEND_ALPHA)


# ds-inspect: the checks pass on the tables that the models train on.
def test_the_inspector_passes_on_the_real_inputs(capsys):
    from scripts.preprocessing import dataset_inspector

    dataset_inspector.results.clear()
    dataset_inspector.run_checks()

    failed = [name for name, passed in dataset_inspector.results if not passed]
    assert not failed, failed
    assert len(dataset_inspector.results) >= 10
