"""
CWRU loading and windowing utilities.

These functions were originally defined inline in the notebooks
(bearing_autoencoder.ipynb cells 19-20). They are reproduced here verbatim in
behaviour so that generic_cwru_asset_specific_experiments.py can be imported
and executed as a standalone module rather than only pasted into a notebook.

Window geometry matches the shared representation's expected input:
1200 samples at 12 kHz (100 ms), non-overlapping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data_paths import cwru_root

BEARING_IDS = ("1730", "1750", "1772", "1797")

WIN = 1200
STEP = 1200

# Filename patterns of the CWRU NumPy conversion, per fault regime.
_FAULT_PATTERNS = {
    "ball": "{bid}_B_*_DE12.npz",
    "inner": "{bid}_IR_*_DE12.npz",
    "outer": "{bid}_OR*@*_DE12.npz",
}


def make_windows(x, win: int = WIN, step: int = STEP) -> np.ndarray:
    """
    Split a 1-D signal into fixed-length, chronological windows.

    Identical to the notebook definition, so every detector is evaluated on
    exactly the same samples.
    """
    x = np.asarray(x, dtype=np.float32).ravel()

    n = (len(x) - win) // step + 1

    if n <= 0:
        raise ValueError(
            f"Signal length ({len(x)}) is smaller than window size ({win})."
        )

    return np.stack(
        [x[i * step : i * step + win] for i in range(n)],
        axis=0,
    )


def _folder(bearing_id: str, data_root: Path | None = None) -> Path:
    root = Path(data_root) if data_root is not None else cwru_root()
    folder = root / f"{bearing_id} RPM"
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Expected CWRU operating-condition directory not found: {folder}\n"
            f"Known operating conditions: {', '.join(BEARING_IDS)}"
        )
    return folder


def load_cwru_signal(
    bearing_id: str,
    regime: str,
    data_root: Path | None = None,
) -> np.ndarray:
    """
    Load the concatenated drive-end channel for one operating condition.

    regime: "healthy" | "ball" | "inner" | "outer"
    """
    folder = _folder(bearing_id, data_root)

    if regime == "healthy":
        path = folder / f"{bearing_id}_Normal.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing healthy recording: {path}")
        return np.load(path)["DE"]

    if regime not in _FAULT_PATTERNS:
        raise ValueError(
            f"Unknown regime {regime!r}; "
            f"expected 'healthy' or one of {tuple(_FAULT_PATTERNS)}"
        )

    pattern = _FAULT_PATTERNS[regime].format(bid=bearing_id)
    files = sorted(folder.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matching {pattern!r} under {folder}"
        )

    return np.concatenate([np.load(f)["DE"] for f in files])


# ---------------------------------------------------------------------
# Names expected by generic_cwru_asset_specific_experiments.py
# ---------------------------------------------------------------------

def load_1797_normal(bearing_id: str) -> np.ndarray:
    return load_cwru_signal(bearing_id, "healthy")


def load_1797_ball(bearing_id: str) -> np.ndarray:
    return load_cwru_signal(bearing_id, "ball")


def load_1797_inner(bearing_id: str) -> np.ndarray:
    return load_cwru_signal(bearing_id, "inner")


def load_1797_outer(bearing_id: str) -> np.ndarray:
    return load_cwru_signal(bearing_id, "outer")
