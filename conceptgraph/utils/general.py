from __future__ import annotations

import functools
import json
import time
from pathlib import Path

import numpy as np
import torch


class Timer:
    """Context manager that prints elapsed wall-clock time."""

    def __init__(self, heading: str = "", verbose: bool = True) -> None:
        self.verbose = verbose
        self.heading = heading
        self.interval: float = 0.0

    def __enter__(self) -> Timer:
        if self.verbose:
            self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        if not self.verbose:
            return
        self.interval = time.perf_counter() - self._start
        print(self.heading, f"{self.interval:.4f}s")


def to_numpy(tensor: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert a tensor (or ndarray) to a numpy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def to_tensor(
    array: np.ndarray | torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert a numpy array (or tensor) to a torch tensor."""
    if isinstance(array, torch.Tensor):
        return array
    t = torch.from_numpy(array)
    return t if device is None else t.to(device)


def to_scalar(d: np.ndarray | torch.Tensor | float) -> int | float:
    """Extract a Python scalar from a single-element array or tensor."""
    if isinstance(d, (int, float)):
        return d
    if isinstance(d, np.generic):
        return d.item()
    if isinstance(d, np.ndarray):
        assert d.size == 1
        return d.item()
    if isinstance(d, torch.Tensor):
        assert d.numel() == 1
        return d.item()
    raise TypeError(f"Invalid type for conversion: {type(d)}")


def prjson(input_json: list[dict] | dict) -> None:
    """Pretty-print a JSON-like object to stdout."""
    if not isinstance(input_json, list):
        input_json = [input_json]

    print("[")
    for i, entry in enumerate(input_json):
        print("  {")
        items = list(entry.items())
        for j, (key, value) in enumerate(items):
            sep = "," if j < len(items) - 1 else ""
            if isinstance(value, str):
                formatted = value.replace("\\n", "\n").replace("\\t", "\t")
                print(f'    "{key}": "{formatted}"{sep}')
            else:
                print(f'    "{key}": {value}{sep}')
        trail = "," if i < len(input_json) - 1 else ""
        print(f"  }}{trail}")
    print("]")


def cfg_to_dict(
    input_cfg: list[dict] | dict,
) -> dict | list[dict]:
    """Convert a config object (or list) to a plain dict."""
    if not isinstance(input_cfg, list):
        input_cfg = [input_cfg]

    result: list[dict] = []
    for entry in input_cfg:
        entry_dict = {}
        for key, value in entry.items():
            if isinstance(value, str):
                value = value.replace("\\n", "\n").replace("\\t", "\t")
            entry_dict[key] = value
        result.append(entry_dict)

    return result[0] if len(result) == 1 else result


def measure_time(func):  # noqa: ANN001, ANN201
    """Decorator that prints the execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"Done! Execution time of {func.__name__}: " f"{elapsed:.2f} seconds")
        return result

    return wrapper


def save_hydra_config(hydra_cfg: dict, exp_out_path: Path) -> None:
    """Persist a Hydra config dict as JSON."""
    with open(exp_out_path / "config_params.json", "w") as f:
        json.dump(cfg_to_dict(hydra_cfg), f, indent=2)
