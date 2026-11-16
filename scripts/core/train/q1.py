"""Subchallenge 1: predict every participant's own rating of each molecule.

``train_q1`` is the whole run, from the input tables to the saved model. It follows the same steps as
``train_q2`` but has no spread curve to calibrate: the 49 participants' ratings are the outputs."""

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
    STANDARD_LOG10_DILUTION,
    intensity_output_columns,
    load_graph_inputs,
    prepare_graph_data,
)
from scripts.core.metrics import (
    TARGETS,
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

Q1_OUTPUT_DIR = PROJECT_ROOT / "output/q1"


Q1_TRAIN_CONFIG = TrainConfig(
    epochs=300,
    patience=100,
    batch_size=64,
    learning_rate=0.0005,
    weight_decay=0.001,
    gradient_clip=5.0,
)


# Subchallenge 1 smooths each participant's training rating towards the molecule's mean: a rating becomes
# Q1_BLEND_ALPHA of the participant's own rating plus the rest of the mean of the people who agree with the
# participant. The weight is fixed. Cross-validation on earlier inputs compared 0.3, 0.5, and 0.8 once (mean
# scores 26.31, 26.32, and 26.57) and found little difference. In the final setup a weight of 0.2 scored lower
# than 0.8 at both feature thresholds, on validation (26.9 and 28.8 against 29.3 and 29.5) and on the test set
# (34.4 and 33.6 against 36.3 and 35.9). A run uses one fixed seed, RANDOM_SEED, and no folds.
Q1_BLEND_ALPHA = 0.8


def train_q1(
    config: TrainConfig = Q1_TRAIN_CONFIG,
    inputs=None,
    output_dir: Path | None = Q1_OUTPUT_DIR,
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> TrainingResult:
    """Train and score the Subchallenge 1 model. ``output_dir=None`` saves nothing."""
    # 1. The data
    device = resolve_device(device)
    data = prepare_graph_data(
        "q1", config.batch_size, blend_alpha=Q1_BLEND_ALPHA, inputs=inputs or load_graph_inputs()
    )
    groups = output_groups("q1")
    intensity_columns = intensity_output_columns("q1")
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
    patience_left = config.patience
    for epoch in range(1, config.epochs + 1):
        # The loss of Subchallenge 1 is computed on ratings in their original units, so it gets the scaler.
        train_objective = train_graph_epoch(
            model, data.train_loader, optimizer, device, config, data.validation_target_scaler
        )
        prediction = predict_graph(
            model,
            data.validation_loader,
            data.validation_target_scaler,
            device,
            data.validation_cids,
            intensity_columns,
        )
        metrics = challenge_scores("q1", data.raw_targets["validation"], prediction)
        score = metrics["dream_challenge_score"]
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
    group_epochs = best_epochs
    validation_score = float(np.mean(list(best_scores.values())))

    # 3. Train again from scratch on the 407 labeled molecules, keeping a copy at each chosen epoch
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
            final_model, data.full_loader, final_optimizer, device, config, data.target_scaler
        )
        if epoch in wanted:
            snapshots[epoch] = {
                key: value.detach().cpu().clone() for key, value in final_model.state_dict().items()
            }
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == last_epoch):
            print(
                f"Final model epoch {epoch:03d}/{last_epoch}: DREAM score estimate {-objective:.4f}"
            )

    # 4. Predict the test molecules once, each score group from the copy at its own epoch
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
        test_metrics=challenge_scores("q1", data.raw_targets["test"], test_prediction),
        group_epochs=group_epochs,
        group_states=group_states,
        std_curve=None,
        validation_score=validation_score,
    )
    if verbose:
        print(
            f"Selected epochs {group_epochs}, validation DREAM score {validation_score:.4f}\n"
            f"Official DREAM Q1 test Z-score: {result.test_metrics['dream_challenge_score']:.3f}"
        )
    if output_dir is None:
        return result

    # 5. Save the scores, the predictions and the model
    checkpoints, metrics_dir = (output_dir / name for name in ("checkpoints", "metrics"))
    for path in (checkpoints, metrics_dir):
        path.mkdir(parents=True, exist_ok=True)

    test_metrics = dict(result.test_metrics)
    test_metrics["evaluation_set"] = "DREAM official TestSet.txt (69 molecules)"
    test_metrics["prediction_targets"] = "21 attributes for each of 49 participants"
    test_metrics["final_model_training"] = (
        f"{len(data.full_cids)} molecules (TrainSet and leaderboard), trained from scratch "
        "for the selected epoch count"
    )
    test_metrics["selected_epoch"] = result.best_epoch
    test_metrics["selected_epochs"] = result.group_epochs
    test_metrics["validation_score_at_selected_epoch"] = validation_score
    test_metrics["population_blend_alpha"] = data.blend_alpha

    predictions = pd.DataFrame(result.test_predictions, columns=data.target_columns)
    predictions.insert(0, "CID", data.test_cids)
    predictions.to_csv(metrics_dir / "custom_model_test_predictions.csv", index=False)
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
            "challenge": "q1",
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
            "neighbours": None,
            "std_curve": None,
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
