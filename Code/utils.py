import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(requested=None):
    device = requested or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return torch.device(device)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_dir(path):
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def json_dumps(value):
    return json.dumps(
        value,
        separators=(",", ":"),
    )