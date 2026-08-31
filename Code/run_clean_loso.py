import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_dataset
from evaluation import evaluate_sklearn_model, evaluate_torch_model
from factory import build_model
from preprocessing import ChannelStandardizer
from splits import make_loso_splits
from training import make_loader, train_model
from utils import (
    ensure_dir,
    get_device,
    json_dumps,
    load_yaml,
    set_seed,
)


MODELS = [
    "csp_lda",
    "eegnet",
    "shallowconvnet",
    "atcnet",
]


def append_result(path: Path, row: dict) -> None:
    ensure_dir(path.parent)

    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def save_predictions(
    directory: Path,
    model_name: str,
    seed: int,
    subject: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
) -> None:
    ensure_dir(directory)

    data = {
        "model": model_name,
        "seed": seed,
        "test_subject": subject,
        "trial_in_subject": np.arange(len(y_true)),
        "y_true": y_true,
        "y_pred": y_pred,
    }

    if probabilities is not None:
        for class_idx in range(probabilities.shape[1]):
            data[f"prob_class_{class_idx}"] = probabilities[:, class_idx]

    output_path = (
        directory
        / f"{model_name}_seed{seed}_subject{subject}.csv"
    )

    pd.DataFrame(data).to_csv(
        output_path,
        index=False,
    )


def run_fold(
    dataset,
    split,
    model_name: str,
    seed: int,
    config: dict,
    device,
    smoke_epochs: int | None,
    output_path: Path,
    predictions_dir: Path,
) -> None:
    set_seed(seed)

    train_idx = split.train_idx
    val_idx = split.val_idx
    test_idx = split.test_idx

    X_train = dataset.X[train_idx]
    y_train = dataset.y[train_idx]

    X_val = dataset.X[val_idx]
    y_val = dataset.y[val_idx]

    X_test = dataset.X[test_idx]
    y_test = dataset.y[test_idx]

    # Fit normalization on TRAINING SUBJECTS ONLY.
    standardizer = ChannelStandardizer()

    X_train = standardizer.fit_transform(X_train)
    X_val = standardizer.transform(X_val)
    X_test = standardizer.transform(X_test)

    model_config = config["models"][model_name]

    model = build_model(
        model_name,
        model_config,
        dataset.n_classes,
        dataset.n_channels,
        dataset.n_samples,
    )

    best_epoch = None
    best_val_loss = None

    if model_name == "csp_lda":
        model.fit(
            X_train,
            y_train,
        )

        metrics, y_pred, probabilities = evaluate_sklearn_model(
            model,
            X_test,
            y_test,
        )

    else:
        training_config = config["training"]

        train_loader = make_loader(
            X_train,
            y_train,
            training_config["batch_size"],
            True,
            training_config["num_workers"],
        )

        val_loader = make_loader(
            X_val,
            y_val,
            training_config["batch_size"],
            False,
            training_config["num_workers"],
        )

        checkpoint_path = (
            Path("Results/checkpoints")
            / f"{model_name}_seed{seed}_subject{split.test_subject}.pt"
        )

        max_epochs = (
            smoke_epochs
            if smoke_epochs is not None
            else training_config["max_epochs"]
        )

        patience = min(
            training_config["patience"],
            max_epochs,
        )

        result = train_model(
            model,
            train_loader,
            val_loader,
            device,
            model_config["learning_rate"],
            model_config["weight_decay"],
            max_epochs,
            patience,
            training_config["min_delta"],
            checkpoint_path,
        )

        best_epoch = result.best_epoch
        best_val_loss = result.best_val_loss

        metrics, y_pred, probabilities = evaluate_torch_model(
            result.model,
            X_test,
            y_test,
            device,
        )

    row = {
        "experiment": "clean_loso",
        "model": model_name,
        "seed": seed,
        "fold": split.fold,
        "test_subject": split.test_subject,
        "train_subjects": ",".join(
            map(str, split.train_subjects)
        ),
        "val_subjects": ",".join(
            map(str, split.val_subjects)
        ),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "kappa": metrics["kappa"],
        "macro_auc_ovo": metrics.get("macro_auc_ovo"),
        "confusion_matrix": json_dumps(
            metrics["confusion_matrix"]
        ),
    }

    append_result(
        output_path,
        row,
    )

    save_predictions(
        predictions_dir,
        model_name,
        seed,
        split.test_subject,
        y_test,
        y_pred,
        probabilities,
    )

    print(
        f"{model_name:16s} "
        f"seed={seed:<3d} "
        f"test=S{split.test_subject} "
        f"acc={metrics['accuracy']:.3f} "
        f"f1={metrics['macro_f1']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run clean LOSO evaluation."
    )

    parser.add_argument(
        "--config",
        default="Config/models.yaml",
    )

    parser.add_argument(
        "--model",
        choices=["all"] + MODELS,
        default="all",
    )

    parser.add_argument(
        "--seeds",
        help="Comma-separated seeds, e.g. 42,123,456",
    )

    parser.add_argument(
        "--test-subject",
        type=int,
    )

    parser.add_argument(
        "--smoke-epochs",
        type=int,
    )

    parser.add_argument(
        "--subjects",
        help=(
            "Optional comma-separated MOABB subject IDs, "
            "e.g. 1,2,3"
        ),
    )

    parser.add_argument(
        "--device",
        help="Optional device override, e.g. cuda:0 or cpu",
    )

    parser.add_argument(
        "--output",
        default="Results/clean/clean_loso.csv",
    )

    args = parser.parse_args()

    config = load_yaml(
        args.config
    )

    dataset_config = config.get(
        "dataset",
        {},
    )

    if args.subjects:
        requested_subjects = [
            int(x.strip())
            for x in args.subjects.split(",")
            if x.strip()
        ]
    else:
        requested_subjects = dataset_config.get(
            "subjects"
        )

    dataset = load_dataset(
        subjects=requested_subjects
    )

    splits = make_loso_splits(
        dataset.y,
        dataset.subjects,
        config["experiment"]["validation_strategy"],
    )

    if args.test_subject is not None:
        splits = [
            split
            for split in splits
            if split.test_subject == args.test_subject
        ]

        if not splits:
            raise ValueError(
                f"Subject {args.test_subject} "
                "is not present in the selected dataset."
            )

    models = (
        MODELS
        if args.model == "all"
        else [args.model]
    )

    if args.seeds:
        seeds = [
            int(x.strip())
            for x in args.seeds.split(",")
            if x.strip()
        ]
    else:
        seeds = config["experiment"]["seeds"]

    device = get_device(
        args.device
    )

    output_path = Path(
        args.output
    )

    predictions_dir = Path(
        "Results/predictions"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Dataset: "
        f"{dataset.n_trials} trials | "
        f"{dataset.n_channels} channels | "
        f"{dataset.n_samples} samples | "
        f"{dataset.n_classes} classes | "
        f"{len(np.unique(dataset.subjects))} subjects"
    )

    for model_name in models:

        # CSP-LDA is deterministic, so multiple random seeds
        # provide no additional information.
        model_seeds = (
            [seeds[0]]
            if model_name == "csp_lda"
            else seeds
        )

        for seed in model_seeds:
            for split in splits:
                run_fold(
                    dataset,
                    split,
                    model_name,
                    seed,
                    config,
                    device,
                    args.smoke_epochs,
                    output_path,
                    predictions_dir,
                )


if __name__ == "__main__":
    main()