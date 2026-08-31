import numpy as np
import torch

from metrics import compute_classification_metrics
from training import make_loader


def evaluate_torch_model(
    model,
    X,
    y,
    device,
    batch_size=256,
):
    loader = make_loader(
        X,
        y,
        batch_size=batch_size,
        shuffle=False,
    )

    probabilities = []
    predictions = []

    model.eval()

    with torch.no_grad():
        for features, _ in loader:
            features = features.to(device)
            logits = model(features)
            batch_probabilities = torch.softmax(
                logits,
                dim=1,
            )

            probabilities.append(
                batch_probabilities.cpu().numpy()
            )
            predictions.append(
                batch_probabilities.argmax(dim=1).cpu().numpy()
            )

    probabilities = np.concatenate(probabilities)
    predictions = np.concatenate(predictions)

    metrics = compute_classification_metrics(
        y,
        predictions,
        probabilities,
    )

    return metrics, predictions, probabilities


def evaluate_sklearn_model(model, X, y):
    predictions = model.predict(X)

    probabilities = (
        model.predict_proba(X)
        if hasattr(model, "predict_proba")
        else None
    )

    metrics = compute_classification_metrics(
        y,
        predictions,
        probabilities,
    )

    return metrics, predictions, probabilities