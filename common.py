"""
common.py -- tiny shared helpers: device selection, seeding, JSON I/O.

Kept deliberately small. The one thing worth reading here is `get_device()`:
it makes the whole repo run on Apple Silicon (MPS), NVIDIA (CUDA), or CPU
without touching any other file.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch


def get_device(verbose: bool = True) -> torch.device:
    """Pick the best available backend: CUDA > Apple MPS > CPU.

    On a Mac you get MPS (the Metal GPU). On a server you get CUDA. Nothing
    else in the codebase ever calls torch.cuda directly -- they all ask here.
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    if verbose:
        print(f"[device] using {dev}")
    return dev


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj))


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
