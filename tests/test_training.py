"""Tests for the shared training loop and the custom graph model."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from scripts.core import data as data_module
from scripts.core.data import (
    ATOM_FEATURE_COUNT,
    EDGE_FEATURE_COUNT,
    INTERACTION_COLUMNS,
    RDKIT_COLUMNS,
    MoleculeRows,
    apply_std_curve,
    blend_q1_training_targets,
    collate_molecules,
    fit_feature_scaler,
    fit_mean_shrinkage,
    fit_std_curve,
    intensity_output_columns,
    neighbour_features,
    neighbour_labels,
    neighbour_statistics,
    restrict_intensity_to_standard_dilution,
    similar_participant_means,
    std_curve_values,
    transform_features,
    transform_names,
    transform_structure,
)
from scripts.core.data import graphs as graphs_module
from scripts.core.metrics import OFFICIAL_NULL_SD, OFFICIAL_Q1_NULL_SD, TARGET_COUNT, TARGETS
from scripts.core.train import (
    calibration_loss,
    challenge_scores,
    dream_score_loss,
    graph_forward,
    inverse_target_transform,
    output_groups,
)
from scripts.preprocessing import feature_selection
from scripts.preprocessing.enrich_molecules import (
    DRAGON_FAMILY_STARTS,
    EXTERNAL_DRAGON_COLUMN,
    dragon_feature_groups,
)
from scripts.training import custom_model as custom_model_module
from scripts.training.custom_model import GraphDescriptorModel


def test_inverse_transform_clips_to_the_rating_range():
    from sklearn.preprocessing import StandardScaler

    ratings = np.random.default_rng(0).uniform(0, 100, size=(20, TARGET_COUNT * 2))
    scaler = StandardScaler().fit(ratings)
    extreme = np.full((2, TARGET_COUNT * 2), 1e6)

    restored = inverse_target_transform(extreme, scaler)

    assert restored.max() <= 100.0
    assert restored.min() >= 0.0


# The custom model is a graph network, so it is exercised on small synthetic molecules.
DRAGON_GROUPS = {"alpha": [0, 1, 2], "beta": [3, 4]}
DRAGON_COUNT = 5
NUMERIC_COUNT = 9
NAME_COUNT = 6
STRUCTURE_COUNT = 7


def synthetic_batch(atom_counts=(5, 8, 3, 6)) -> dict:
    generator = torch.Generator().manual_seed(0)
    items = []
    for atom_count in atom_counts:
        edges = torch.tensor(
            [[i, j] for i in range(atom_count) for j in range(atom_count) if i != j]
        ).T
        items.append(
            {
                "atoms": torch.randn(atom_count, ATOM_FEATURE_COUNT, generator=generator),
                "edges": edges,
                "edge_features": torch.randn(
                    edges.shape[1], EDGE_FEATURE_COUNT, generator=generator
                ),
                "dragon": torch.randn(DRAGON_COUNT, generator=generator),
                "numeric": torch.randn(NUMERIC_COUNT, generator=generator),
                "name_features": torch.rand(NAME_COUNT, generator=generator),
                "structure_features": torch.rand(
                    STRUCTURE_COUNT, generator=torch.Generator().manual_seed(atom_count)
                ),
                "targets": torch.randn(TARGET_COUNT * 2, generator=generator),
                "cid": 1,
            }
        )
    return collate_molecules(items)


def graph_model(output_count: int = TARGET_COUNT * 2) -> GraphDescriptorModel:
    return GraphDescriptorModel(
        DRAGON_GROUPS, NUMERIC_COUNT, NAME_COUNT, output_count, structure_count=STRUCTURE_COUNT
    )


def test_collate_offsets_edges_into_one_batch_graph():
    batch = synthetic_batch((2, 3))

    assert batch["atoms"].shape[0] == 5
    assert batch["graph_ids"].tolist() == [0, 0, 1, 1, 1]
    # The second molecule's atoms are numbered after the first molecule's two atoms.
    assert int(batch["edges"][:, 2:].min()) == 2


def test_graph_model_predicts_means_and_deviations_for_every_molecule():
    batch = synthetic_batch()
    model = graph_model().eval()

    with torch.no_grad():
        output = graph_forward(model, batch, torch.device("cpu"))

    assert output.shape == (4, TARGET_COUNT * 2)
    assert torch.isfinite(output).all()


def test_graph_model_can_predict_every_participant():
    batch = synthetic_batch()
    model = graph_model(TARGET_COUNT * 49).eval()

    with torch.no_grad():
        output = graph_forward(model, batch, torch.device("cpu"))

    assert output.shape == (4, TARGET_COUNT * 49)


def test_graph_model_trains_with_the_dream_score_loss():
    batch = synthetic_batch()
    model = graph_model()

    loss = dream_score_loss(graph_forward(model, batch, torch.device("cpu")), batch["targets"])
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_dream_score_loss_rewards_correlated_predictions():
    targets = torch.randn(20, TARGET_COUNT * 2, generator=torch.Generator().manual_seed(1))

    perfect = dream_score_loss(targets, targets)
    constant = dream_score_loss(torch.zeros_like(targets), targets)

    # Every group has a correlation of 1, and the weights keep the scale of the official loss.
    expected = -1.0 / float(np.mean(list(OFFICIAL_NULL_SD.values())))
    assert torch.isclose(perfect, torch.tensor(expected), rtol=1e-4)
    assert float(constant) == 0.0


def test_dragon_columns_are_grouped_by_family_order():
    columns = [EXTERNAL_DRAGON_COLUMN]
    for start, _, _ in DRAGON_FAMILY_STARTS:
        columns.extend([start, f"{start}_next"])

    groups = dragon_feature_groups(columns)

    assert list(groups) == [
        "composition_properties",
        "three_d_shape",
        "two_d_local_patterns",
        "two_d_topology",
    ]
    assert EXTERNAL_DRAGON_COLUMN not in {c for members in groups.values() for c in members}
    assert sum(map(len, groups.values())) == 2 * len(DRAGON_FAMILY_STARTS)
    assert "dragon_ZM1" in groups["two_d_topology"]
    assert "dragon_G1_next" in groups["three_d_shape"]


# The graph model fits sklearn scalers on training molecules only and saves them as joblib
# artifacts.
SCALER_DRAGON_COLUMNS = [f"dragon_{i}" for i in range(4)]


def scaler_frame(molecules: int = 30, replicates: int = 1) -> pd.DataFrame:
    generator = np.random.default_rng(3)
    columns = [*SCALER_DRAGON_COLUMNS, *RDKIT_COLUMNS, *INTERACTION_COLUMNS]
    frame = pd.DataFrame(generator.lognormal(size=(molecules, len(columns))), columns=columns)
    frame.insert(0, "CID", np.arange(molecules))
    return pd.concat([frame] * replicates, ignore_index=True)


def test_feature_scaler_standardises_the_training_molecules():
    training = scaler_frame()
    scalers = fit_feature_scaler(training, SCALER_DRAGON_COLUMNS)
    block = scalers["rdkit"].transform(training[list(RDKIT_COLUMNS)].to_numpy())

    assert np.allclose(block.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(block.std(axis=0), 1.0, atol=1e-9)


def test_feature_scaler_imputes_missing_and_infinite_descriptors():
    training = scaler_frame()
    training.loc[0, SCALER_DRAGON_COLUMNS[0]] = np.nan
    training.loc[1, SCALER_DRAGON_COLUMNS[0]] = np.inf
    scalers = fit_feature_scaler(training, SCALER_DRAGON_COLUMNS)

    scaled = scalers["dragon"].transform(training[SCALER_DRAGON_COLUMNS].to_numpy())

    assert np.isfinite(scaled).all()


def test_feature_scaler_counts_each_molecule_once():
    single = fit_feature_scaler(scaler_frame(), SCALER_DRAGON_COLUMNS)
    repeated = fit_feature_scaler(scaler_frame(replicates=3), SCALER_DRAGON_COLUMNS)

    probe = scaler_frame()[SCALER_DRAGON_COLUMNS].to_numpy()
    assert np.allclose(single["dragon"].transform(probe), repeated["dragon"].transform(probe))


def test_transform_features_lays_out_the_numeric_block():
    frame = scaler_frame(10)
    frame["morgan_0"] = 1.0
    frame["log10_concentration"] = -3.0
    frame["is_high_concentration"] = 1.0
    scalers = fit_feature_scaler(frame, SCALER_DRAGON_COLUMNS)

    dragon, numeric = transform_features(frame, scalers, SCALER_DRAGON_COLUMNS, ["morgan_0"])

    assert dragon.shape == (10, len(SCALER_DRAGON_COLUMNS))
    assert numeric.shape == (10, len(RDKIT_COLUMNS) + len(INTERACTION_COLUMNS) + 1 + 2)
    assert (numeric[:, -2] == -3.0).all() and (numeric[:, -1] == 1.0).all()


def test_target_scaler_ignores_missing_ratings_and_leaves_an_empty_column_unscaled():
    targets = pd.DataFrame(np.ones((5, 2)), columns=["a", "b"]) * [1.0, 3.0]
    targets.loc[0, "a"] = np.nan
    scaler = data_module.fit_masked_target_scaler(targets, ["a", "b"])
    assert np.allclose(scaler.mean_, [1.0, 3.0])

    targets["a"] = np.nan
    scaler = data_module.fit_masked_target_scaler(targets, ["a", "b"])
    assert scaler.mean_[0] == 0.0 and scaler.scale_[0] == 1.0


def test_graph_scalers_survive_a_save_and_load(tmp_path):
    training = scaler_frame()
    scalers = fit_feature_scaler(training, SCALER_DRAGON_COLUMNS)
    joblib.dump(scalers, tmp_path / "feature_scaler.joblib")

    restored = joblib.load(tmp_path / "feature_scaler.joblib")

    probe = training[SCALER_DRAGON_COLUMNS].to_numpy()
    assert np.allclose(restored["dragon"].transform(probe), scalers["dragon"].transform(probe))


# Subchallenge 1: individual ratings with missing entries.
Q1_OUTPUTS = TARGET_COUNT * 49


def test_q1_loss_ignores_missing_ratings():
    generator = torch.Generator().manual_seed(2)
    target = torch.randn(40, Q1_OUTPUTS, generator=generator)
    prediction = torch.randn(40, Q1_OUTPUTS, generator=generator, requires_grad=True)
    mask = torch.rand(40, Q1_OUTPUTS, generator=generator) > 0.3
    unscale = (torch.full((Q1_OUTPUTS,), 40.0), torch.full((Q1_OUTPUTS,), 18.0))

    loss = dream_score_loss(prediction, target, mask, unscale)
    loss.backward()

    assert torch.isfinite(loss)
    assert (prediction.grad[~mask] == 0).all()
    assert (prediction.grad[mask] != 0).any()


def test_graph_model_trains_for_every_participant_with_the_masked_loss():
    batch = synthetic_batch()
    model = graph_model(Q1_OUTPUTS)
    prediction = graph_forward(model, batch, torch.device("cpu"))
    target = torch.randn(prediction.shape)
    mask = torch.rand(prediction.shape) > 0.3

    loss = dream_score_loss(prediction, target, mask)
    loss.backward()

    assert prediction.shape == (4, Q1_OUTPUTS)
    assert torch.isfinite(loss)


def test_challenge_scores_use_the_metric_of_the_subchallenge():
    generator = np.random.default_rng(0)
    q2_actual = generator.uniform(0, 100, size=(20, TARGET_COUNT * 2))
    q1_actual = generator.uniform(0, 100, size=(20, TARGET_COUNT, 49))

    q2 = challenge_scores("q2", q2_actual, q2_actual)
    q1 = challenge_scores("q1", q1_actual, q1_actual.reshape(20, -1))

    assert list(q2["z_scores"]) == list(OFFICIAL_NULL_SD)
    assert list(q1["z_scores"]) == ["intensity", "valence", "descriptors"]


def test_graph_model_adds_a_token_for_the_structure_features():
    batch = synthetic_batch()
    batch["structure_features"] = torch.rand(4, STRUCTURE_COUNT)
    model = GraphDescriptorModel(
        DRAGON_GROUPS, NUMERIC_COUNT, NAME_COUNT, TARGET_COUNT * 2, structure_count=STRUCTURE_COUNT
    ).eval()

    with torch.no_grad():
        output = graph_forward(model, batch, torch.device("cpu"))
        batch["structure_features"] = torch.zeros(4, STRUCTURE_COUNT)
        changed = graph_forward(model, batch, torch.device("cpu"))

    assert output.shape == (4, TARGET_COUNT * 2)
    assert torch.isfinite(output).all()
    assert not torch.allclose(output, changed)


def test_graph_model_needs_the_structure_features():
    model = GraphDescriptorModel(
        DRAGON_GROUPS, NUMERIC_COUNT, NAME_COUNT, TARGET_COUNT * 2, structure_count=STRUCTURE_COUNT
    )
    batch = synthetic_batch()

    with pytest.raises(TypeError):
        model(
            batch["atoms"],
            batch["edges"],
            batch["edge_features"],
            batch["graph_ids"],
            batch["dragon"],
            batch["numeric"],
            batch["name_features"],
        )


def test_epi_suite_properties_are_scaled_into_the_numeric_block():
    frame = scaler_frame(10)
    frame["epi_00_vapor_pressure"] = np.linspace(1.0, 1e12, 10)
    frame["epi_01_log_kow"] = np.linspace(-1.0, 4.0, 10)
    frame.loc[0, "epi_01_log_kow"] = np.nan
    frame["morgan_0"] = 1.0
    frame["log10_concentration"] = -3.0
    frame["is_high_concentration"] = 1.0
    scalers = fit_feature_scaler(frame, SCALER_DRAGON_COLUMNS)

    _, numeric = transform_features(frame, scalers, SCALER_DRAGON_COLUMNS, ["morgan_0"])

    assert "epi" in scalers
    assert numeric.shape == (10, len(RDKIT_COLUMNS) + len(INTERACTION_COLUMNS) + 2 + 1 + 2)
    assert np.isfinite(numeric).all()
    assert np.abs(numeric[:, :-3]).max() < 10.0  # the huge vapor pressures are tamed


@pytest.fixture(autouse=True)
def without_feature_manifest(monkeypatch, tmp_path):
    """These tests use small synthetic tables, so they must not depend on the real manifest."""
    monkeypatch.setattr(feature_selection, "MANIFEST_PATH", tmp_path / "no_manifest.csv")


def concentration_batch(rows: int = 40):
    """Ratings at two concentrations, 0 and 1, that differ by a large constant."""
    generator = torch.Generator().manual_seed(11)
    groups = (torch.arange(rows) % 2).float()
    signal = torch.randn(rows, TARGET_COUNT * 2, generator=generator)
    target = signal + 10.0 * groups[:, None]  # the concentration shifts every rating
    return signal, target, groups


def test_the_concentration_label_alone_does_not_score_inside_the_groups():
    _, target, groups = concentration_batch()
    label_only = 10.0 * groups[:, None].expand_as(target)  # knows the concentration, nothing else

    across = dream_score_loss(label_only, target)
    inside = dream_score_loss(label_only, target, groups=groups)

    assert float(across) < -1.0  # across the whole batch the label looks like real skill
    assert abs(float(inside)) < 1e-3  # inside each concentration it carries none


def test_tracking_the_molecules_still_scores_inside_the_groups():
    signal, target, groups = concentration_batch()
    prediction = signal + 10.0 * groups[:, None]

    assert float(dream_score_loss(prediction, target, groups=groups)) < -1.0


def test_one_group_is_the_same_as_no_groups():
    signal, target, _ = concentration_batch()
    prediction = signal + 0.5 * torch.randn_like(signal)

    plain = dream_score_loss(prediction, target)
    grouped = dream_score_loss(prediction, target, groups=torch.zeros(len(target)))

    assert torch.allclose(plain, grouped)


def test_the_grouped_loss_trains_and_keeps_gradients_finite():
    _, target, groups = concentration_batch()
    prediction = torch.randn_like(target, requires_grad=True)

    loss = dream_score_loss(prediction, target, groups=groups)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()


NAME_WINDOWS = [f"dream_odor_name_{index:05d}" for index in range(6)]


def name_frame(molecules: int = 30) -> pd.DataFrame:
    """The table the scalers need, plus name windows of three kinds of name."""
    generator = np.random.default_rng(8)
    topics = generator.random((molecules, 3)) < 0.4
    frame = scaler_frame(molecules)
    for index, column in enumerate(NAME_WINDOWS):
        frame[column] = (topics[:, index % 3] & (generator.random(molecules) < 0.9)).astype(float)
    return frame


def test_name_windows_are_used_as_they_are_with_missing_ones_as_zero():
    frame = name_frame()
    frame.loc[0, NAME_WINDOWS[0]] = np.nan

    names = transform_names(frame)

    assert names.shape == (30, len(NAME_WINDOWS))
    assert names[0, 0] == 0.0
    assert np.allclose(names[1:], frame.loc[1:, NAME_WINDOWS].to_numpy())


def test_a_table_without_name_windows_has_no_name_features():
    assert transform_names(scaler_frame(10)).shape == (10, 0)


def test_the_name_windows_are_what_the_dataset_hands_to_the_model(monkeypatch):
    frame = name_frame(10)
    names = transform_names(frame)
    rows = frame.assign(
        rdkit_canonical_smiles="CCO",
        rdkit_gasteiger_atom_charges="[0.0, 0.1, -0.2]",
        rdkit_gasteiger_implicit_hydrogen_charges="[0.0, 0.0, 0.0]",
    )
    monkeypatch.setattr(graphs_module, "load_conformers", lambda: {})

    dataset = MoleculeRows(
        rows,
        np.zeros((10, 2)),
        np.zeros((10, 3)),
        np.zeros((10, TARGET_COUNT * 2)),
        names=names,
    )

    assert dataset.name_features.shape == (10, len(NAME_WINDOWS))
    assert np.allclose(dataset.name_features.numpy(), names)


def structure_frame(molecules: int = 30) -> pd.DataFrame:
    """The table the scalers need, plus a few columns of each structure family."""
    generator = np.random.default_rng(9)
    frame = scaler_frame(molecules)
    for index in range(8):
        frame[f"nspdk_{index:05d}"] = (generator.random(molecules) < 0.4).astype(float)
    for index in range(3):
        frame[f"nsim_{index:04d}"] = generator.random(molecules) * 0.1
        frame[f"msim_{index:04d}"] = generator.random(molecules)
    return frame


STRUCTURE_WINDOWS = (
    [f"nspdk_{index:05d}" for index in range(8)]
    + [f"nsim_{index:04d}" for index in range(3)]
    + [f"msim_{index:04d}" for index in range(3)]
)


def test_structure_columns_are_used_as_they_are():
    frame = structure_frame()

    structure = transform_structure(frame)

    assert structure.shape == (30, len(STRUCTURE_WINDOWS))
    assert np.allclose(structure, frame[STRUCTURE_WINDOWS].to_numpy())


def test_a_table_without_structure_columns_has_no_structure_features():
    assert transform_structure(scaler_frame(10)).shape == (10, 0)


def test_the_structure_columns_are_what_the_dataset_hands_to_the_model(monkeypatch):
    frame = structure_frame(10)
    structure = transform_structure(frame)
    rows = frame.assign(
        rdkit_canonical_smiles="CCO",
        rdkit_gasteiger_atom_charges="[0.0, 0.1, -0.2]",
        rdkit_gasteiger_implicit_hydrogen_charges="[0.0, 0.0, 0.0]",
    )
    monkeypatch.setattr(graphs_module, "load_conformers", lambda: {})

    dataset = MoleculeRows(
        rows,
        np.zeros((10, 2)),
        np.zeros((10, 3)),
        np.zeros((10, TARGET_COUNT * 2)),
        structure=structure,
    )

    assert dataset.structure_features.shape == (10, len(STRUCTURE_WINDOWS))
    assert dataset.structure_columns == STRUCTURE_WINDOWS
    assert np.allclose(dataset.structure_features.numpy(), structure)


@pytest.mark.parametrize(
    "challenge, outputs", [("q1", TARGET_COUNT * 49), ("q2", TARGET_COUNT * 2)]
)
def test_every_output_column_belongs_to_exactly_one_score_group(challenge, outputs):
    groups = output_groups(challenge)

    columns = sorted(column for members in groups.values() for column in members)
    assert columns == list(range(outputs))


def test_the_score_groups_are_named_like_the_z_scores():
    q1 = challenge_scores(
        "q1",
        np.random.default_rng(0).random((6, 21, 49)),
        np.random.default_rng(1).random((6, 21 * 49)),
    )
    q2 = challenge_scores(
        "q2", np.random.default_rng(0).random((6, 42)), np.random.default_rng(1).random((6, 42))
    )

    assert set(output_groups("q1")) == set(q1["z_scores"])
    assert set(output_groups("q2")) == set(q2["z_scores"])


def test_token_dropout_changes_training_outputs_but_not_evaluation_outputs():
    batch = synthetic_batch()
    model = GraphDescriptorModel(
        DRAGON_GROUPS,
        NUMERIC_COUNT,
        NAME_COUNT,
        TARGET_COUNT * 2,
        structure_count=STRUCTURE_COUNT,
        token_dropout=0.5,
    )
    cpu = torch.device("cpu")

    model.eval()
    with torch.no_grad():
        first, second = graph_forward(model, batch, cpu), graph_forward(model, batch, cpu)
    model.train()
    torch.manual_seed(0)
    with torch.no_grad():
        dropped = [graph_forward(model, batch, cpu) for _ in range(3)]

    assert torch.equal(first, second)
    assert any(not torch.allclose(first, output) for output in dropped)
    assert all(torch.isfinite(output).all() for output in dropped)


def test_a_molecule_always_keeps_at_least_one_token_even_when_all_are_dropped():
    model = GraphDescriptorModel(
        DRAGON_GROUPS,
        NUMERIC_COUNT,
        NAME_COUNT,
        TARGET_COUNT * 2,
        structure_count=STRUCTURE_COUNT,
        token_dropout=1.0,
    )
    model.train()

    visible = model.visible_tokens(torch.zeros(6, 7, 4))

    assert visible.shape == (6, 7)
    assert (visible.sum(dim=1) == 1).all()


def test_token_dropout_must_be_a_probability():
    with pytest.raises(ValueError, match="between 0 and 1"):
        GraphDescriptorModel(
            DRAGON_GROUPS,
            NUMERIC_COUNT,
            NAME_COUNT,
            TARGET_COUNT * 2,
            structure_count=STRUCTURE_COUNT,
            token_dropout=1.5,
        )


def test_balanced_loss_counts_every_group_equally_per_point_of_correlation():
    ratings = torch.rand(30, Q1_OUTPUTS, generator=torch.Generator().manual_seed(4)) * 100
    unscale = (torch.zeros(Q1_OUTPUTS), torch.ones(Q1_OUTPUTS))
    sds = torch.tensor(list(OFFICIAL_Q1_NULL_SD.values()))

    perfect = dream_score_loss(ratings, ratings, None, unscale)

    # Every group has a correlation of 1, and the weights keep the scale of the official loss.
    assert torch.isclose(perfect, -(1.0 / sds.mean()), rtol=1e-4)


def test_balanced_loss_still_trains_every_group():
    generator = torch.Generator().manual_seed(6)
    target = torch.rand(40, Q1_OUTPUTS, generator=generator) * 100
    unscale = (torch.zeros(Q1_OUTPUTS), torch.ones(Q1_OUTPUTS))
    prediction = (target + 30 * torch.randn(target.shape, generator=generator)).requires_grad_()

    dream_score_loss(prediction, target, None, unscale).backward()

    for columns in (slice(0, 49), slice(49, 98), slice(98, None)):
        assert float(prediction.grad[:, columns].abs().sum()) > 0.0


def test_intensity_ratings_are_kept_only_at_the_standard_dilution():
    mask = np.ones((4, TARGET_COUNT * 49), dtype=bool)
    numeric = np.zeros((4, 5))
    numeric[:, -2] = [-3.0, -1.0, -3.0, -5.0]  # the log10 concentration of every row
    numeric[:, -1] = [1.0, 1.0, 0.0, 1.0]  # whether it is the high concentration

    restricted = restrict_intensity_to_standard_dilution(mask, numeric)

    assert restricted[:, :49].any(axis=1).tolist() == [True, False, True, False]
    assert restricted[:, 49:].all()  # every other attribute keeps all of its ratings
    assert mask.all()  # the input is not changed


def test_the_standard_dilution_changes_only_the_concentration_the_model_sees():
    batch = synthetic_batch()
    batch["numeric"][:, -2] = -1.0
    batch["numeric"][:, -1] = 0.0
    model = graph_model().eval()
    cpu = torch.device("cpu")
    standard = dict(batch)
    standard["numeric"] = batch["numeric"].clone()
    standard["numeric"][:, -2] = -3.0
    standard["numeric"][:, -1] = 1.0

    with torch.no_grad():
        actual = graph_forward(model, batch, cpu)
        as_standard = graph_forward(model, batch, cpu, standard_dilution=True)
        expected = graph_forward(model, standard, cpu)

    assert torch.allclose(as_standard, expected)
    assert not torch.allclose(actual, as_standard)
    assert float(batch["numeric"][0, -2]) == -1.0  # the batch itself is not changed


def test_name_columns_list_our_windows_first_and_then_the_organizers():
    frame = pd.DataFrame(
        columns=["orgname_00001", "dream_odor_name_00002", "dream_odor_name_00001", "orgname_00000"]
    )

    assert data_module.name_feature_columns(frame) == [
        "dream_odor_name_00001",
        "dream_odor_name_00002",
        "orgname_00000",
        "orgname_00001",
    ]


def sample_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CID": [1, 1, 2, 3, 3],
            "replicate_key": ["", "", "", "", ""],
            "condition": ["high", "low", "high", "high", "low"],
            "log10_concentration": [-3.0, -5.0, -1.0, -3.0, -5.0],
            "is_high_concentration": [1.0, 0.0, 1.0, 1.0, 0.0],
            "mean_sweet": [40.0, 20.0, 50.0, 30.0, 10.0],
            "mean_intensity_strength": [70.0, 30.0, 60.0, 50.0, 10.0],
            "q1_sweet_subject_1": [60.0, 10.0, 80.0, 30.0, np.nan],
            "q1_intensity_strength_subject_1": [80.0, 40.0, 60.0, 50.0, 20.0],
        }
    )


def test_the_spread_is_zero_at_both_ends_and_largest_in_the_middle():
    means = np.array([0.0, 10.0, 50.0, 90.0, 100.0])

    spread = std_curve_values(means, 0.7, 1.1)

    assert spread[0] == pytest.approx(0.0)
    assert spread[-1] == pytest.approx(0.0, abs=1e-9)
    assert spread.argmax() == 2
    assert (spread >= 0).all()


def test_the_curve_of_the_spread_is_recovered_from_ratings():
    generator = np.random.default_rng(0)
    mean = generator.uniform(2, 98, size=200)
    rows = pd.DataFrame({"condition": "high"}, index=range(200))
    for target in TARGETS:
        rows[f"mean_{target}"] = mean
        rows[f"std_{target}"] = std_curve_values(mean, 0.7, 1.1)

    curve = fit_std_curve(rows)

    assert curve["k0"].shape == curve["k1"].shape == (TARGET_COUNT,)
    assert np.allclose(curve["k0"], 0.7, atol=0.02)
    assert np.allclose(curve["k1"], 1.1, atol=0.02)


def test_the_curve_is_fitted_on_high_concentration_rows_only():
    rows = pd.DataFrame({"condition": ["high"] * 50 + ["low"] * 50})
    generator = np.random.default_rng(1)
    mean = generator.uniform(5, 95, size=100)
    for target in TARGETS:
        rows[f"mean_{target}"] = mean
        rows[f"std_{target}"] = np.where(
            rows["condition"] == "high", std_curve_values(mean, 0.7, 1.1), 99.0
        )

    curve = fit_std_curve(rows)

    assert np.allclose(curve["k0"], 0.7, atol=0.02)


def test_the_predicted_spread_mixes_in_the_curve_of_the_predicted_mean():
    curve = {"k0": np.full(TARGET_COUNT, 0.7), "k1": np.full(TARGET_COUNT, 1.1)}
    predictions = np.vstack([np.concatenate([np.full(21, 40.0), np.full(21, 5.0)])])
    from_mean = std_curve_values(40.0, 0.7, 1.1)

    assert np.allclose(apply_std_curve(predictions, curve, 1.0)[:, 21:], from_mean)
    assert np.allclose(apply_std_curve(predictions, curve, 0.0), predictions)
    half = apply_std_curve(predictions, curve, 0.5)
    assert np.allclose(half[:, 21:], 0.5 * from_mean + 0.5 * 5.0)
    assert np.allclose(half[:, :21], 40.0)  # the means are not touched
    assert np.allclose(predictions[:, 21:], 5.0)  # the input is not changed


def test_the_intensity_outputs_of_each_subchallenge():
    assert intensity_output_columns("q1") == list(range(49))
    assert intensity_output_columns("q2") == [0, TARGET_COUNT]


def test_restricting_intensity_can_name_the_columns_of_subchallenge_2():
    mask = np.ones((3, TARGET_COUNT * 2), dtype=bool)
    numeric = np.zeros((3, 4))
    numeric[:, -2] = [-3.0, -1.0, -5.0]

    restricted = restrict_intensity_to_standard_dilution(
        mask, numeric, intensity_output_columns("q2")
    )

    assert restricted[:, [0, TARGET_COUNT]].any(axis=1).tolist() == [True, False, False]
    assert restricted[:, 1:TARGET_COUNT].all()


def test_every_attribute_can_have_its_own_weight_for_the_curve():
    curve = {"k0": np.full(TARGET_COUNT, 0.7), "k1": np.full(TARGET_COUNT, 1.1)}
    predictions = np.concatenate([np.full(21, 40.0), np.full(21, 5.0)])[None, :]
    weights = np.zeros(TARGET_COUNT)
    weights[0] = 1.0  # only intensity takes its spread from the curve
    from_mean = std_curve_values(40.0, 0.7, 1.1)

    mixed = apply_std_curve(predictions, curve, weights)

    assert mixed[0, TARGET_COUNT] == pytest.approx(from_mean)
    assert np.allclose(mixed[0, TARGET_COUNT + 1 :], 5.0)


def test_the_spread_weights_cover_the_21_attributes_and_give_pleasantness_almost_none():
    weights = np.array(data_module.STD_FROM_MEAN_WEIGHTS)

    assert weights.shape == (TARGET_COUNT,)
    assert weights[0] == 1.0 and weights[1] < 0.1
    assert ((weights >= 0) & (weights <= 1)).all()


def test_the_calibration_loss_pins_the_level_that_the_score_loss_ignores():
    generator = torch.Generator().manual_seed(2)
    target = torch.randn(40, TARGET_COUNT * 2, generator=generator)
    prediction = target + 0.3 * torch.randn(target.shape, generator=generator)

    shifted = prediction + 5.0

    # The score loss compares molecules only, so a shift of every prediction costs nothing.
    assert torch.isclose(
        dream_score_loss(prediction, target), dream_score_loss(shifted, target), atol=1e-4
    )
    # The calibration loss sees the shift.
    assert calibration_loss(shifted, target) > 10 * calibration_loss(prediction, target)


def test_the_calibration_loss_ignores_ratings_that_do_not_exist():
    target = torch.zeros(4, TARGET_COUNT * 2)
    prediction = torch.ones(4, TARGET_COUNT * 2)
    mask = torch.zeros(4, TARGET_COUNT * 2, dtype=torch.bool)
    mask[:, 0] = True

    assert calibration_loss(prediction, target, mask).item() == 1.0
    assert calibration_loss(prediction, target, torch.zeros_like(mask)).item() == 0.0
    target[0, 0] = float("nan")
    assert torch.isfinite(calibration_loss(prediction, target, mask))


def test_intensity_is_asked_at_1_1000_with_the_flag_of_the_other_rating_of_the_pair():
    batch = synthetic_batch()
    batch["numeric"][:, -2] = torch.tensor([-1.0, -5.0, -1.0, -5.0])
    batch["numeric"][:, -1] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    model = graph_model().eval()
    cpu = torch.device("cpu")
    expected_batch = dict(batch)
    expected_batch["numeric"] = batch["numeric"].clone()
    expected_batch["numeric"][:, -2] = -3.0
    expected_batch["numeric"][:, -1] = torch.tensor([0.0, 1.0, 0.0, 1.0])

    with torch.no_grad():
        asked = graph_forward(model, batch, cpu, standard_dilution=True)
        expected = graph_forward(model, expected_batch, cpu)

    assert torch.allclose(asked, expected)


def similar_ratings(rows: int = 40) -> pd.DataFrame:
    generator = np.random.default_rng(3)
    base = generator.uniform(0, 100, size=rows)
    return pd.DataFrame(
        {
            "q1_sweet_subject_1": base,
            "q1_sweet_subject_2": base + generator.normal(0, 1, rows),  # agrees with 1
            "q1_sweet_subject_3": 100 - base,  # disagrees with 1 and 2
        }
    )


def test_a_participant_is_smoothed_towards_the_people_who_agree_with_them():
    ratings = similar_ratings()

    means = similar_participant_means(ratings, ("sweet",), (1, 2, 3))["sweet"]

    # Participant 1 agrees with participant 2 only, so the mean is the rating of participant 2.
    assert np.allclose(means[:, 0], ratings["q1_sweet_subject_2"], atol=1e-9)
    assert np.allclose(means[:, 1], ratings["q1_sweet_subject_1"], atol=1e-9)
    # Participant 3 disagrees with both, so nobody has a weight and the mean is missing.
    assert np.isnan(means[:, 2]).all()


def test_a_participants_own_rating_and_missing_ratings_do_not_enter_their_mean():
    ratings = similar_ratings()
    ratings.loc[0, "q1_sweet_subject_2"] = np.nan

    means = similar_participant_means(ratings, ("sweet",), (1, 2, 3))["sweet"]

    assert np.isnan(means[0, 0])  # the only person who agrees did not rate this row
    assert not np.isnan(means[1:, 0]).any()


def test_blended_targets_use_the_people_who_agree_with_the_participant():
    generator = np.random.default_rng(5)
    count = 30
    rows = pd.DataFrame(index=range(count))
    for target in TARGETS:
        base = generator.uniform(0, 100, size=count)
        rows[f"mean_{target}"] = base
        for subject in range(1, 50):
            rows[f"q1_{target}_subject_{subject}"] = base + generator.normal(0, 15, count)
    column = "q1_sweet_subject_1"

    personal = blend_q1_training_targets(rows, 0.8)

    # Not the plain population mean: the people who agree with the participant weigh more.
    plain = 0.8 * rows[column] + 0.2 * rows["mean_sweet"]
    assert not np.allclose(personal[column], plain)
    # A participant's own rating always keeps most of the weight.
    assert np.corrcoef(personal[column], rows[column])[0, 1] > 0.95


def test_the_shrinkage_is_the_slope_of_the_truth_on_the_predictions():
    generator = np.random.default_rng(0)
    predicted = generator.normal(50, 20, size=(200, 3))
    observed = 0.5 * (predicted - 50) + 50 + generator.normal(0, 1, size=(200, 3))

    assert fit_mean_shrinkage(predicted, observed) == pytest.approx(0.5, abs=0.03)


def test_the_shrinkage_never_stretches_and_never_collapses():
    predicted = np.random.default_rng(1).normal(size=(50, 2))

    assert fit_mean_shrinkage(predicted, 3 * predicted) == 1.0
    assert fit_mean_shrinkage(predicted, 0.001 * predicted) == 0.1
    assert fit_mean_shrinkage(np.zeros((5, 2)), predicted[:5]) == 1.0


def test_shrinking_the_means_keeps_the_curve_away_from_the_ends():
    curve = {
        "k0": np.full(TARGET_COUNT, 0.7),
        "k1": np.full(TARGET_COUNT, 1.1),
        "center": np.full(TARGET_COUNT, 50.0),
        "shrink": np.full(TARGET_COUNT, 0.5),
    }
    predictions = np.vstack(
        [
            np.concatenate([np.full(21, 2.0), np.full(21, 9.0)]),
            np.concatenate([np.full(21, 98.0), np.full(21, 9.0)]),
        ]
    )

    mixed = apply_std_curve(predictions, curve, 1.0)

    expected_low = std_curve_values(26.0, 0.7, 1.1)  # 2 shrunk half way to 50
    expected_high = std_curve_values(74.0, 0.7, 1.1)
    assert np.allclose(mixed[0, 21:], expected_low)
    assert np.allclose(mixed[1, 21:], expected_high)
    assert np.allclose(mixed[:, :21], predictions[:, :21])  # the means are not changed
    unshrunk = apply_std_curve(predictions, {"k0": curve["k0"], "k1": curve["k1"]}, 1.0)
    assert (unshrunk[:, 21:] < mixed[:, 21:]).all()  # near the ends the curve is much smaller


def test_the_curve_of_intensity_is_fitted_on_the_rows_at_1_1000():
    generator = np.random.default_rng(2)
    mean = generator.uniform(5, 95, size=300)
    at_one_thousandth = np.arange(300) % 2 == 0
    rows = pd.DataFrame(
        {
            "condition": np.where(at_one_thousandth, "low", "high"),  # the pair 1/10 and 1/1000
            "log10_concentration": np.where(at_one_thousandth, -3.0, -1.0),
        }
    )
    for target in TARGETS:
        rows[f"mean_{target}"] = mean
        at_standard = rows["log10_concentration"] == -3.0
        rows[f"std_{target}"] = np.where(
            at_standard, std_curve_values(mean, 0.7, 1.1), std_curve_values(mean, 0.3, 1.1)
        )

    curve = fit_std_curve(rows)

    assert curve["k0"][0] == pytest.approx(0.7, abs=0.03)  # intensity: the rows at 1/1000
    assert curve["k0"][1] == pytest.approx(0.3, abs=0.03)  # the others: the high concentration rows
    assert curve["center"].shape == (TARGET_COUNT,)


def test_the_curve_carries_its_own_weights_when_they_are_given():
    curve = {
        "k0": np.full(TARGET_COUNT, 0.7),
        "k1": np.full(TARGET_COUNT, 1.1),
        "weight": np.zeros(TARGET_COUNT),
    }
    predictions = np.concatenate([np.full(21, 40.0), np.full(21, 5.0)])[None, :]

    assert np.allclose(
        apply_std_curve(predictions, curve), predictions
    )  # weight 0 keeps the spread
    curve["weight"] = np.ones(TARGET_COUNT)
    assert np.allclose(
        apply_std_curve(predictions, curve)[:, 21:], std_curve_values(40.0, 0.7, 1.1)
    )
    # With no weights in the curve, ``STD_FROM_MEAN_WEIGHTS`` are used.
    plain = apply_std_curve(predictions, {"k0": curve["k0"], "k1": curve["k1"]})
    assert plain[0, 21] == pytest.approx(std_curve_values(40.0, 0.7, 1.1))  # intensity: weight 1


def similarity_matrix() -> pd.DataFrame:
    values = np.array(
        [
            [1.0, 0.9, 0.1, 0.2],
            [0.9, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.8],
            [0.2, 0.1, 0.8, 1.0],
        ]
    )
    cids = [10, 20, 30, 40]
    return pd.DataFrame(values, index=cids, columns=cids)


def neighbour_reference() -> pd.DataFrame:
    labels = pd.DataFrame(0.0, index=[10, 20, 30], columns=list(TARGETS))
    labels[TARGETS[1]] = [10.0, 30.0, 90.0]  # molecule 10 is rated 10, 20 is rated 30, 30 is 90
    return labels


def test_a_reference_molecule_never_sees_its_own_label_among_its_neighbours():
    reference = neighbour_reference()
    statistics = {"mean": np.zeros(TARGET_COUNT), "scale": np.ones(TARGET_COUNT)}

    features = neighbour_features(
        np.array([10]), reference, similarity_matrix(), statistics, power=1, count=20
    )

    # Molecule 10 is not its own neighbour, so the mean is over molecules 20 and 30 with weights 0.9, 0.1.
    expected = (0.9 * 30.0 + 0.1 * 90.0) / 1.0
    assert features[0, 1] == pytest.approx(expected, rel=1e-5)


def test_a_new_molecule_gets_the_similarity_weighted_labels_of_its_nearest_references():
    reference = neighbour_reference()
    statistics = {"mean": np.zeros(TARGET_COUNT), "scale": np.ones(TARGET_COUNT)}

    # Molecule 40 is a reference to nobody: it is closest to 30 (0.8), then 10 (0.2), then 20 (0.1).
    all_three = neighbour_features(
        np.array([40]), reference, similarity_matrix(), statistics, power=1, count=3
    )
    nearest_only = neighbour_features(
        np.array([40]), reference, similarity_matrix(), statistics, power=1, count=1
    )

    assert all_three[0, 1] == pytest.approx((0.8 * 90 + 0.2 * 10 + 0.1 * 30) / 1.1, rel=1e-5)
    assert nearest_only[0, 1] == pytest.approx(90.0)  # only the closest molecule counts


def test_neighbour_features_are_standardised_and_every_row_of_a_molecule_agrees():
    reference = neighbour_reference()
    statistics = neighbour_statistics(reference)

    features = neighbour_features(
        np.array([40, 40, 30]), reference, similarity_matrix(), statistics, power=2, count=20
    )

    assert features.shape == (3, TARGET_COUNT)
    assert np.array_equal(features[0], features[1])
    assert np.isfinite(features).all()
    assert features.dtype == np.float32


def test_a_label_that_no_neighbour_has_is_filled_with_the_average():
    reference = neighbour_reference()
    reference.loc[[10, 20, 30], TARGETS[2]] = np.nan  # nobody rated this attribute
    reference.loc[[10, 20], TARGETS[3]] = [10.0, 30.0]
    reference.loc[30, TARGETS[3]] = np.nan
    statistics = {"mean": np.full(TARGET_COUNT, 50.0), "scale": np.full(TARGET_COUNT, 10.0)}

    features = neighbour_features(
        np.array([40]), reference, similarity_matrix(), statistics, power=1, count=20
    )

    assert features[0, 2] == pytest.approx(0.0)  # the average, which standardises to zero
    assert features[0, 3] == pytest.approx(
        (np.average([10.0, 30.0], weights=[0.2, 0.1]) - 50.0) / 10.0
    )


def test_the_labels_of_the_neighbours_take_the_intensity_at_1_1000_for_training_molecules():
    rows = pd.DataFrame(
        {
            "CID": [1, 1, 2],
            "condition": ["high", "low", "high"],
            "log10_concentration": [-1.0, -3.0, -3.0],
        }
    )
    for target in TARGETS:
        rows[f"mean_{target}"] = [60.0, 20.0, 40.0]

    labels = neighbour_labels(rows, intensity_at_standard=True)
    held_out = neighbour_labels(rows, intensity_at_standard=False)

    assert labels.loc[1, TARGETS[0]] == 20.0  # molecule 1: the row at 1/1000 is the low row
    assert labels.loc[1, TARGETS[1]] == 60.0  # the others: the high rows
    assert labels.loc[2, TARGETS[0]] == 40.0
    assert held_out.loc[1, TARGETS[0]] == 60.0  # held out rows already give the scored intensity


def test_the_model_takes_a_neighbour_token_only_when_asked_to():
    batch = synthetic_batch()
    batch["neighbour_features"] = torch.randn(4, TARGET_COUNT)
    with_neighbours = GraphDescriptorModel(
        DRAGON_GROUPS,
        NUMERIC_COUNT,
        NAME_COUNT,
        TARGET_COUNT * 2,
        structure_count=STRUCTURE_COUNT,
        neighbour_count=TARGET_COUNT,
    ).eval()
    plain = graph_model().eval()
    cpu = torch.device("cpu")

    with torch.no_grad():
        shown = graph_forward(with_neighbours, batch, cpu)
        changed = dict(batch)
        changed["neighbour_features"] = batch["neighbour_features"] + 3.0
        moved = graph_forward(with_neighbours, changed, cpu)
        ignored = graph_forward(
            plain, batch, cpu
        )  # the plain model has no such token and ignores it

    assert shown.shape == ignored.shape == (4, TARGET_COUNT * 2)
    assert not torch.allclose(shown, moved)
    assert "neighbour_branch.0.weight" in with_neighbours.state_dict()
    assert not any(name.startswith("neighbour") for name in plain.state_dict())
    broken = dict(batch)
    del broken["neighbour_features"]
    with pytest.raises(ValueError, match="neighbour features"):
        graph_forward(with_neighbours.eval(), broken, cpu)


def expert_model(seed: int = 0, outputs: int = TARGET_COUNT * 2) -> GraphDescriptorModel:
    torch.manual_seed(seed)
    return GraphDescriptorModel(
        DRAGON_GROUPS,
        NUMERIC_COUNT,
        NAME_COUNT,
        outputs,
        structure_count=STRUCTURE_COUNT,
        neighbour_count=TARGET_COUNT,
        experts=True,
    )


def expert_batch() -> dict:
    batch = synthetic_batch()
    batch["structure_features"] = torch.rand(4, STRUCTURE_COUNT)
    batch["neighbour_features"] = torch.randn(4, TARGET_COUNT)
    return batch


def test_the_experts_each_read_their_own_view_and_all_read_the_numeric_token():
    model = expert_model()

    names = model.token_names
    views = [{names[position] for position in members} for members in model.expert_tokens]

    assert model.experts and len(model.expert_tokens) == 4
    assert all("numeric" in view for view in views)
    assert {"alpha", "beta"} <= views[0] and "graph" not in views[0]
    assert {"graph", "names"} <= views[1]
    assert "structure" in views[2] and "neighbours" in views[3]


def test_the_experts_give_finite_answers_that_depend_on_every_view():
    batch = expert_batch()
    model = expert_model().eval()
    cpu = torch.device("cpu")

    with torch.no_grad():
        base = graph_forward(model, batch, cpu)
        for name in ("structure_features", "neighbour_features", "name_features"):
            changed = dict(batch)
            changed[name] = batch[name] + 2.0
            assert not torch.allclose(base, graph_forward(model, changed, cpu)), name

    assert base.shape == (4, TARGET_COUNT * 2)
    assert torch.isfinite(base).all()


def test_the_experts_stay_finite_when_tokens_and_experts_are_dropped_hard(monkeypatch):
    monkeypatch.setattr(custom_model_module, "EXPERT_DROPOUT", 0.9)
    batch = expert_batch()
    model = expert_model()
    model.token_dropout = 0.9
    model.train()

    for _ in range(15):
        output = graph_forward(model, batch, torch.device("cpu"))
        assert torch.isfinite(output).all()
    loss = dream_score_loss(output, batch["targets"])
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_the_experts_train_every_expert_and_the_gate():
    batch = expert_batch()
    model = expert_model()
    loss = dream_score_loss(graph_forward(model, batch, torch.device("cpu")), batch["targets"])
    loss.backward()

    assert float(model.gate.weight.grad.abs().sum()) > 0.0
    for position in range(4):
        assert float(model.expert_heads[position][0][-1].weight.grad.abs().sum()) > 0.0


def test_the_experts_do_not_exist_in_the_subchallenge_1_model():
    plain = GraphDescriptorModel(
        DRAGON_GROUPS, NUMERIC_COUNT, NAME_COUNT, TARGET_COUNT * 49, structure_count=STRUCTURE_COUNT
    )
    asked = GraphDescriptorModel(
        DRAGON_GROUPS,
        NUMERIC_COUNT,
        NAME_COUNT,
        TARGET_COUNT * 49,
        structure_count=STRUCTURE_COUNT,
        experts=True,
    )

    assert not plain.experts and not asked.experts  # participants have their own attention instead
    assert not any(name.startswith(("expert", "gate")) for name in asked.state_dict())
    assert set(plain.state_dict()) <= set(asked.state_dict())


def test_a_model_with_experts_restores_from_its_state():
    batch = expert_batch()
    model = expert_model(1).eval()
    restored = expert_model(2).eval()
    restored.load_state_dict(model.state_dict())

    with torch.no_grad():
        first = graph_forward(model, batch, torch.device("cpu"))
        second = graph_forward(restored, batch, torch.device("cpu"))

    assert torch.allclose(first, second)


def test_the_smoothed_epoch_ignores_a_single_lucky_peak():
    from scripts.core.train import smoothed_best_epochs

    scores = [1.0, 1.1, 1.2, 3.0, 1.1, 1.0, 1.0, 2.0, 2.1, 2.2, 2.1, 2.0, 2.0]
    assert smoothed_best_epochs(scores, 1) == 4  # the lucky spike
    assert smoothed_best_epochs(scores, 5) == 10  # the steady stretch
    assert smoothed_best_epochs([3.0, 3.0, 3.0], 3) == 1  # ties go to the earliest epoch
