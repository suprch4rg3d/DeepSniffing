"""Subchallenge 2: predict the mean and the spread of the panel's ratings of each molecule.

``train_q2`` is the whole run. It chooses an epoch for each score group (the one whose score, averaged over 7
epochs around it, is highest), fits the calibration of the predicted spread on the validation predictions (the
spread of a rating follows its mean along a curve, which is mixed with the network's own spread), retrains on the
407 labeled molecules and scores the test set once. The settings below belong to Subchallenge 2 alone, so nothing
done for it can change Subchallenge 1."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.core.data import (
    ATOM_FEATURE_COUNT,
    DISTANCE_CUTOFF,
    DISTANCE_RBF_COUNT,
    EDGE_FEATURE_COUNT,
    INTERACTION_COLUMNS,
    REFERENCE_PREDICTIONS_PATH,
    STANDARD_LOG10_DILUTION,
    STD_FROM_MEAN_WEIGHTS,
    apply_std_curve,
    fit_mean_shrinkage,
    intensity_output_columns,
    load_graph_inputs,
    prepare_graph_data,
)
from scripts.core.metrics import (
    TARGETS,
    dream_challenge_metrics,
    prediction_frame,
    prediction_matrix_from_submission,
    write_metrics,
)
from scripts.core.splits import PROJECT_ROOT
from scripts.core.train.loop import (
    MIN_SCORE_DELTA,
    TrainConfig,
    TrainingResult,
    challenge_scores,
    output_groups,
    predict_graph,
    resolve_device,
    seed_everything,
    train_graph_epoch,
)
from scripts.preprocessing.community_features import NAME_PREFIX
from scripts.training.custom_model import (
    GRAPH_HIDDEN_SIZE,
    build_graph_model,
)

Q2_OUTPUT_DIR = PROJECT_ROOT / "output/q2"


# In single runs of the experts setup, a learning rate of 0.001 scored 9.55 on validation and 9.98 on test,
# against 9.42 and 9.66 for 0.0005. 0.0003 and 0.002 were both worse, and adding a weight decay of 0.003 or
# 10 neighbours to 0.001 did not help.
Q2_TRAIN_CONFIG = TrainConfig(
    epochs=300,
    patience=100,
    batch_size=32,
    learning_rate=0.001,
    weight_decay=0.001,
    gradient_clip=5.0,
)


# The score loss only compares molecules, so it does not care where the predictions sit or how far they
# spread: a shift or a stretch of every prediction leaves it unchanged, and the level drifts during training.
# Subchallenge 2 needs real ratings, because the spread of a rating follows its mean along a curve on the
# scale of 0 to 100, and a model that predicted intensity below zero gave 16 of 69 test molecules the same
# spread of 0. This weight adds the squared error of the standardised predictions to the loss, which pins
# their level and scale.
Q2_CALIBRATION_WEIGHT = 2.0


# The epoch of each score group is where the score, averaged over this many epochs around it, is highest,
# and not where a single epoch peaked by luck.
Q2_EPOCH_WINDOW = 7


# The constants that the weight of the curve in the spread of each group is chosen from, next to the fixed
# weights of ``STD_FROM_MEAN_WEIGHTS``.
SPREAD_WEIGHT_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0)


def smoothed_best_epochs(scores: np.ndarray, window: int) -> int:
    """The 1-based epoch where the scores, averaged over ``window`` epochs around it, peak.

    Near the ends the average uses the epochs that exist. A window of 1 gives the epoch of the
    highest score, and the earliest epoch wins a tie.
    """
    scores = np.asarray(scores, dtype=np.float64)
    half = window // 2
    averaged = np.array(
        [scores[max(0, i - half) : i + half + 1].mean() for i in range(len(scores))]
    )
    return int(np.argmax(averaged)) + 1


def train_q2(
    config: TrainConfig = Q2_TRAIN_CONFIG,
    inputs=None,
    output_dir: Path | None = Q2_OUTPUT_DIR,
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> TrainingResult:
    """Train, calibrate and score the Subchallenge 2 model. ``output_dir=None`` saves nothing."""
    # 1. The data
    device = resolve_device(device)
    data = prepare_graph_data("q2", config.batch_size, inputs=inputs or load_graph_inputs())
    groups = output_groups("q2")
    intensity_columns = intensity_output_columns("q2")
    if verbose:
        print(
            f"Using {device}; training CIDs {len(data.train_cids)}; "
            f"validation CIDs {len(data.validation_cids)}; official test CIDs {len(data.test_cids)}"
        )

    # 2. Train on the 338 training molecules and choose the epoch of each score group on the validation ones
    seed_everything(config.seed)
    model = build_graph_model(data).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: dict[str, list] = {"train_loss": [], "validation_loss": [], "validation_metrics": []}
    best_scores = {name: float("-inf") for name in groups}
    best_epochs = {name: 0 for name in groups}
    best_predictions: dict[str, np.ndarray] = {}
    every_prediction: list[np.ndarray] = []
    patience_left = config.patience
    for epoch in range(1, config.epochs + 1):
        train_objective = train_graph_epoch(
            model, data.train_loader, optimizer, device, config, None, Q2_CALIBRATION_WEIGHT
        )
        raw_prediction = predict_graph(
            model,
            data.validation_loader,
            data.validation_target_scaler,
            device,
            data.validation_cids,
            intensity_columns,
        )
        # The spread is the one the curve of the mean gives, mixed with the network's own.
        prediction = apply_std_curve(raw_prediction, data.std_curve)
        metrics = challenge_scores("q2", data.raw_targets["validation"], prediction)
        score = metrics["dream_challenge_score"]
        every_prediction.append(raw_prediction)
        history["train_loss"].append(float(train_objective))
        history["validation_loss"].append(float(-score))
        history["validation_metrics"].append(metrics)
        if verbose:
            print(
                f"Epoch {epoch:03d}/{config.epochs} | train DREAM score estimate "
                f"{-train_objective:.4f} | validation DREAM score {score:.4f}"
            )
        improved = False
        for name in groups:
            if metrics["z_scores"][name] > best_scores[name] + MIN_SCORE_DELTA:
                best_scores[name] = metrics["z_scores"][name]
                best_epochs[name] = epoch
                best_predictions[name] = raw_prediction
                improved = True
        if improved:
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left == 0:
                if verbose:
                    print(f"Early stopping after {config.patience} epochs without improvement")
                break
    if min(best_epochs.values()) == 0:
        raise RuntimeError("Training did not produce a model checkpoint.")
    # The epoch of each group is the peak of its smoothed score, and not the single best epoch.
    for name in groups:
        series = [m["z_scores"][name] for m in history["validation_metrics"]]
        epoch = smoothed_best_epochs(series, Q2_EPOCH_WINDOW)
        best_epochs[name] = epoch
        best_scores[name] = series[epoch - 1]
        best_predictions[name] = every_prediction[epoch - 1]
    group_epochs = best_epochs
    validation_score = float(np.mean(list(best_scores.values())))

    # 3. Calibrate the spread on the validation predictions, each group at its chosen epoch
    truth = data.raw_targets["validation"]
    count = len(TARGETS)
    # How much to pull the predicted means of each score group towards their centre: one number for
    # intensity, one for pleasantness and one for the descriptors.
    shrink = np.ones(
        len(groups["intensity_mean"])
        + len(groups["valence_mean"])
        + len(groups["descriptors_mean"])
    )
    for name in ("intensity_mean", "valence_mean", "descriptors_mean"):
        columns = groups[name]
        shrink[columns] = fit_mean_shrinkage(best_predictions[name][:, columns], truth[:, columns])
    curve = {**data.std_curve, "shrink": shrink}
    # How much of the spread of each group comes from the curve. The network has a spread output of its own,
    # so for the three spread groups the fixed weights and five constants are scored on the validation
    # molecules, and the best of them is kept. The fixed weights win a tie.
    predictions = np.zeros((len(data.validation_cids), 2 * count))
    for name, columns in groups.items():
        predictions[:, columns] = best_predictions[name][:, columns]
    chosen = np.array(STD_FROM_MEAN_WEIGHTS, dtype=np.float64)
    for name in ("intensity_std", "valence_std", "descriptors_std"):
        attributes = [column - count for column in groups[name]]
        candidates = [chosen[attributes].copy()] + [
            np.full(len(attributes), value) for value in SPREAD_WEIGHT_CANDIDATES
        ]
        best_score, best_weights = -np.inf, candidates[0]
        for candidate in candidates:
            trial = chosen.copy()
            trial[attributes] = candidate
            mixed = apply_std_curve(predictions, {**curve, "weight": trial})
            score = challenge_scores("q2", truth, mixed)["z_scores"][name]
            if score > best_score + 1e-9:
                best_score, best_weights = score, candidate
        chosen[attributes] = best_weights
    curve["weight"] = chosen
    if verbose:
        summary = {
            name: round(float(curve["shrink"][groups[name][0]]), 2)
            for name in ("intensity_mean", "valence_mean", "descriptors_mean")
        }
        print(f"Predicted means are pulled towards the centre by {summary} before the curve")
        weights = {
            name: round(float(np.mean(curve["weight"][[c - count for c in groups[name]]])), 2)
            for name in ("intensity_std", "valence_std", "descriptors_std")
        }
        print(f"The curve makes up this much of the spread (average): {weights}")

    # 4. Train again from scratch on the 407 labeled molecules, keeping a copy at each chosen epoch
    wanted = set(group_epochs.values())
    last_epoch = max(wanted)
    seed_everything(config.seed)
    final_model = build_graph_model(data).to(device)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    if verbose:
        print(
            f"Final model: training on {len(data.full_cids)} molecules (training and validation) "
            f"for {last_epoch} epochs; group epochs {group_epochs}"
        )
    snapshots: dict[int, dict] = {}
    for epoch in range(1, last_epoch + 1):
        objective = train_graph_epoch(
            final_model,
            data.full_loader,
            final_optimizer,
            device,
            config,
            None,
            Q2_CALIBRATION_WEIGHT,
        )
        if epoch in wanted:
            snapshots[epoch] = {
                key: value.detach().cpu().clone() for key, value in final_model.state_dict().items()
            }
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == last_epoch):
            print(
                f"Final model epoch {epoch:03d}/{last_epoch}: DREAM score estimate {-objective:.4f}"
            )

    # 5. Predict the test molecules once, each score group from the copy at its own epoch
    test_prediction = None
    group_states = {}
    for epoch in sorted(wanted):
        final_model.load_state_dict(snapshots[epoch])
        prediction = predict_graph(
            final_model,
            data.test_loader,
            data.target_scaler,
            device,
            data.test_cids,
            intensity_columns,
            curve,
        )
        if test_prediction is None:
            test_prediction = np.zeros_like(prediction)
        for name, columns in groups.items():
            if group_epochs[name] == epoch:
                test_prediction[:, columns] = prediction[:, columns]
                group_states[name] = snapshots[epoch]
    # The model that the result carries is the copy that serves the largest group.
    largest = max(groups, key=lambda name: len(groups[name]))
    final_model.load_state_dict(group_states[largest])
    result = TrainingResult(
        model=final_model,
        history=history,
        best_epoch=last_epoch,
        test_predictions=test_prediction,
        test_metrics=challenge_scores("q2", data.raw_targets["test"], test_prediction),
        group_epochs=group_epochs,
        group_states=group_states,
        std_curve=curve,
        validation_score=validation_score,
    )
    if verbose:
        print(
            f"Selected epochs {group_epochs}, validation DREAM score {validation_score:.4f}\n"
            f"Official DREAM Q2 test Z-score: {result.test_metrics['dream_challenge_score']:.3f}"
        )
    if output_dir is None:
        return result

    # 6. Save the scores, the predictions and the model
    checkpoints, metrics_dir = (output_dir / name for name in ("checkpoints", "metrics"))
    for path in (checkpoints, metrics_dir):
        path.mkdir(parents=True, exist_ok=True)

    test_metrics = dict(result.test_metrics)
    test_metrics["evaluation_set"] = "DREAM official TestSet.txt (69 molecules)"
    test_metrics["prediction_targets"] = "21 high-intensity means and 21 standard deviations"
    test_metrics["final_model_training"] = (
        f"{len(data.full_cids)} molecules (TrainSet and leaderboard), trained from scratch "
        "for the selected epoch count"
    )
    test_metrics["selected_epoch"] = result.best_epoch
    test_metrics["selected_epochs"] = result.group_epochs
    test_metrics["validation_score_at_selected_epoch"] = validation_score
    reference_metrics = None
    if REFERENCE_PREDICTIONS_PATH.is_file():
        reference_metrics = dream_challenge_metrics(
            data.raw_targets["test"],
            prediction_matrix_from_submission(REFERENCE_PREDICTIONS_PATH, data.test_cids),
        )
        test_metrics["stored_reference_prediction_score"] = reference_metrics[
            "dream_challenge_score"
        ]

    prediction_frame(data.test_cids, result.test_predictions, TARGETS).to_csv(
        metrics_dir / "custom_model_test_predictions.tsv", sep="\t", index=False
    )
    write_metrics(metrics_dir / "dream_challenge_test_metrics.json", test_metrics)
    pd.DataFrame(
        [
            {
                "epoch": index + 1,
                "train_loss": result.history["train_loss"][index],
                "dream_challenge_score": values["dream_challenge_score"],
                **{
                    f"pearson_{key}": value for key, value in values["pearson_correlations"].items()
                },
                **{f"z_{key}": value for key, value in values["z_scores"].items()},
            }
            for index, values in enumerate(result.history["validation_metrics"])
        ]
    ).to_csv(metrics_dir / "validation_dream_challenge_metrics.csv", index=False)

    # Inference scores a molecule at high concentration, so keep the typical high value. The last two
    # numeric inputs are the concentration and the flag.
    numeric_inputs = data.full_loader.dataset.numeric
    high_concentration = float(numeric_inputs[numeric_inputs[:, -1] == 1.0, -2].median())
    joblib.dump(data.feature_scaler, checkpoints / "feature_scaler.joblib")
    joblib.dump(data.target_scaler, checkpoints / "target_scaler.joblib")
    torch.save(
        {
            "model_name": "GraphDescriptorModel",
            "challenge": "q2",
            "blend_alpha": data.blend_alpha,
            "high_log10_concentration": high_concentration,
            "state_dict": {
                key: value.detach().cpu() for key, value in result.model.state_dict().items()
            },
            "target_columns": data.target_columns,
            "target_names": list(TARGETS),
            "dragon_columns": data.dragon_columns,
            "dragon_feature_groups": data.dragon_groups,
            "morgan_columns": data.morgan_columns,
            "interaction_columns": list(INTERACTION_COLUMNS),
            "name_ngram_vocabulary": data.name_vocabulary,
            "name_count": data.name_count,
            "structure_count": data.structure_count,
            "community_columns": {
                "name": [
                    column
                    for column in data.train_loader.dataset.name_columns
                    if column.startswith(NAME_PREFIX)
                ],
                "structure": data.train_loader.dataset.structure_columns,
            },
            "numeric_count": data.numeric_count,
            "hidden_size": GRAPH_HIDDEN_SIZE,
            "atom_feature_count": ATOM_FEATURE_COUNT,
            "edge_feature_count": EDGE_FEATURE_COUNT,
            "distance_cutoff_angstrom": DISTANCE_CUTOFF,
            "distance_rbf_count": DISTANCE_RBF_COUNT,
            "selected_epoch": result.best_epoch,
            "group_epochs": result.group_epochs,
            "experts": bool(getattr(result.model, "experts", False)),
            "neighbours": {
                "cids": data.neighbours["labels"].index.to_numpy(),
                "labels": data.neighbours["labels"].to_numpy(),
                "mean": data.neighbours["mean"],
                "scale": data.neighbours["scale"],
                "power": data.neighbours["power"],
                "count": data.neighbours["count"],
                "similarities": data.neighbours["similarities"],
                "width": data.neighbours["width"],
            },
            "std_curve": result.std_curve,
            "intensity_standard_log10_dilution": STANDARD_LOG10_DILUTION,
            "group_columns": groups,
            "group_state_dicts": {
                name: {key: value.detach().cpu() for key, value in state.items()}
                for name, state in result.group_states.items()
            },
            "validation_score": validation_score,
            "training_cids": data.full_cids,
            "validation_cids": data.validation_cids,
        },
        checkpoints / "custom_model.pt",
    )
    if verbose:
        print(f"Artifacts: {output_dir}")
    return result
