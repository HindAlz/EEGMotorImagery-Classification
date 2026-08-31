from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainResult:
    model: nn.Module
    best_epoch: int
    best_val_loss: float
    history: list[dict]


def make_loader(
    X,
    y,
    batch_size,
    shuffle,
    num_workers=0,
):
    features = torch.from_numpy(
        X[:, None, :, :]
    ).float()
    labels = torch.from_numpy(
        np.asarray(y, dtype=np.int64)
    )

    dataset = TensorDataset(features, labels)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _mean_loss(
    model,
    loader,
    criterion,
    device,
):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            predictions = model(features)
            loss = criterion(predictions, labels)

            batch_size = len(labels)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / max(total_samples, 1)


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    learning_rate,
    weight_decay,
    max_epochs,
    patience,
    min_delta=1e-4,
    checkpoint_path=None,
):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    stale_epochs = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            predictions = model(features)
            loss = criterion(predictions, labels)

            loss.backward()
            optimizer.step()

            batch_size = len(labels)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = total_loss / max(total_samples, 1)
        val_loss = _mean_loss(
            model,
            val_loader,
            criterion,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0

            if checkpoint_path is not None:
                path = Path(checkpoint_path)
                path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                torch.save(best_state, path)
        else:
            stale_epochs += 1

            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("No valid checkpoint")

    model.load_state_dict(best_state)

    return TrainResult(
        model=model,
        best_epoch=best_epoch,
        best_val_loss=float(best_val_loss),
        history=history,
    )