"""
Online operational-radius estimation and convergence-based commissioning.

Extracted verbatim from the notebook cells that previously defined these
objects inline (bearing_autoencoder_cwru_per_asset.ipynb and
bearing_autoencoder_NASA_per_asset.ipynb), so that CWRU and NASA IMS share a
single definition rather than two copies that can drift apart. Both notebooks
now import from here.

This module is pure NumPy: it has no TensorFlow dependency and requires no
external data, so the commissioning logic can be verified independently of
the representation and of the datasets (see verification/verify_commissioning.py).

Method summary (manuscript Section: Operational-radius convergence)
------------------------------------------------------------------
The quantity shared fleet-wide is the target nominal coverage (98%), not an
absolute threshold. Each asset estimates its own operational radius as the
empirical percentile of its local nominal reconstruction residuals, and stops
collecting once that estimate is stable:

    C(t) = |tau(t) - tau(t-L)| / max(|tau(t-L)|, eps)

Commissioning ends at the first t >= min_index for which C(t) < tolerance
holds for `consecutive` successive updates. The radius is then frozen.
"""

from __future__ import annotations

from collections import deque

import numpy as np


# =====================================================================
# Online operational-radius estimator
# =====================================================================

class AdaptiveOperationalRadius:
    """
    Online empirical-quantile estimator.

    The globally shared quantity is the target nominal coverage
    (e.g. 98%), not a z-score or an absolute threshold.

    The local operational radius is estimated as the empirical
    percentile of the accepted nominal residuals.
    """

    def __init__(
        self,
        percentile=98.0,
        window_size=100,
        min_samples=20,
    ):
        if not 0.0 < percentile < 100.0:
            raise ValueError("percentile must be between 0 and 100.")

        if window_size < 2:
            raise ValueError("window_size must be at least 2.")

        if min_samples < 2:
            raise ValueError("min_samples must be at least 2.")

        if min_samples > window_size:
            raise ValueError(
                "min_samples cannot be larger than window_size."
            )

        self.percentile = float(percentile)
        self.window_size = int(window_size)
        self.min_samples = int(min_samples)

        self.buf = deque(maxlen=self.window_size)

    def update(self, mse_value):
        """
        Add one nominal residual and update the operational radius.

        Returns
        -------
        tau : float
            Current operational radius. NaN until min_samples is reached.
        """
        self.buf.append(float(mse_value))

        if len(self.buf) < self.min_samples:
            return np.nan

        arr = np.asarray(self.buf, dtype=np.float64)

        return float(
            np.percentile(
                arr,
                self.percentile,
            )
        )

    def warmup_trace(self, mse_sequence):
        """
        Process an entire warm-up sequence and return tau(t).
        """
        history_tau = []

        for mse_value in mse_sequence:
            history_tau.append(
                self.update(mse_value)
            )

        return np.asarray(history_tau, dtype=np.float64)


# =====================================================================
# Convergence utilities
# =====================================================================

def operational_radius_convergence_trace(
    tau_history,
    *,
    lookback=10,
    eps=1e-12,
):
    """
    Relative variation of tau over a fixed look-back interval.

    C(t) = |tau(t) - tau(t-L)| / max(|tau(t-L)|, eps)
    """
    tau_history = np.asarray(tau_history, dtype=np.float64)

    convergence = np.full(
        tau_history.shape,
        np.nan,
        dtype=np.float64,
    )

    for t in range(lookback, len(tau_history)):
        current = tau_history[t]
        previous = tau_history[t - lookback]

        if np.isfinite(current) and np.isfinite(previous):
            convergence[t] = (
                abs(current - previous)
                / max(abs(previous), eps)
            )

    return convergence


def find_convergence_index(
    tau_history,
    *,
    lookback=10,
    tolerance=0.01,
    consecutive=10,
    min_index=0,
):
    """
    Return the first zero-based index at which the operational radius
    remains stable for `consecutive` updates.

    Convergence is not evaluated before `min_index`.
    """
    convergence = operational_radius_convergence_trace(
        tau_history,
        lookback=lookback,
    )

    stable_count = 0

    for t, value in enumerate(convergence):

        if t < min_index:
            stable_count = 0
            continue

        if np.isfinite(value) and value < tolerance:
            stable_count += 1

            if stable_count >= consecutive:
                return t
        else:
            stable_count = 0

    return None
