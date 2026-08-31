import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray | None = None,
) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro")
        ),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
        ).tolist(),
    }

    if y_probability is not None:
        try:
            metrics["macro_auc_ovo"] = float(
                roc_auc_score(
                    y_true,
                    y_probability,
                    multi_class="ovo",
                    average="macro",
                    labels=np.arange(y_probability.shape[1]),
                )
            )
        except ValueError:
            metrics["macro_auc_ovo"] = None

    return metrics