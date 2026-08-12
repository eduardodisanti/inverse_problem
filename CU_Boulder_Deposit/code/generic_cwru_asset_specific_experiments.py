
"""
Generic, comparable CWRU experiment framework.

Purpose
-------
Train one detector per CWRU operating condition ("ideal but non-scalable" baseline)
using exactly the same windows and splits for every detector.

All detectors expose:
    fit(X_train)
    score(X)  -> larger score means "more anomalous"

Supported examples:
    - Conv1D autoencoder
    - One-Class SVM
    - Elliptic Envelope
    - Any custom detector implementing the same interface

Data access
-----------
The loaders and windowing helper are imported from cwru_loader, which resolves
the dataset location through data_paths (environment variable, then
<deposit>/data/CWRU, then the legacy sibling layout). No hard-coded path is
required.

When this file is instead pasted directly into a notebook that has already
defined data_root, load_1797_normal, load_1797_ball, load_1797_inner,
load_1797_outer and make_windows, the import below is skipped and the
notebook's own definitions are used unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Protocol, Any
import random

import numpy as np
import pandas as pd

try:  # standalone module use
    from cwru_loader import (
        make_windows,
        load_1797_normal,
        load_1797_ball,
        load_1797_inner,
        load_1797_outer,
    )
except ImportError:  # pasted into a notebook that already defines them
    pass

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

GLOBAL_SEED = 42


def set_all_seeds(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------
# Common detector interface
# ---------------------------------------------------------------------

class AnomalyDetector(Protocol):
    """Minimal interface required by the generic experiment runner."""

    def fit(self, X_train: np.ndarray, X_val: np.ndarray | None = None) -> "AnomalyDetector":
        ...

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return one anomaly score per sample; larger means more anomalous."""
        ...


# ---------------------------------------------------------------------
# Dataset definition
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class HealthySplit:
    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class CWRUDataset:
    bearing_id: str
    healthy: HealthySplit
    faults: Dict[str, np.ndarray]
    scaler: StandardScaler


def chronological_split(
    X: np.ndarray,
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.20,
) -> HealthySplit:
    """
    Chronological split avoids overlapping or near-duplicate windows leaking
    between training, threshold calibration, and healthy evaluation.
    """
    n = len(X)
    n_train = int(np.floor(n * train_fraction))
    n_cal = int(np.floor(n * calibration_fraction))

    if n_train < 2 or n_cal < 2 or (n - n_train - n_cal) < 2:
        raise ValueError(
            f"Not enough healthy windows ({n}) for train/calibration/test split."
        )

    return HealthySplit(
        train=X[:n_train],
        calibration=X[n_train:n_train + n_cal],
        test=X[n_train + n_cal:],
    )


def build_cwru_dataset(
    bearing_id: str,
    *,
    win: int = 1200,
    step: int = 1200,
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.20,
) -> CWRUDataset:
    """
    Build one asset-specific CWRU dataset.

    Crucially:
    - identical windows for every detector;
    - scaler fitted only on healthy training windows;
    - threshold calibrated only on held-out healthy calibration windows;
    - healthy FAR measured on a separate healthy test split;
    - faults never used in training or calibration.
    """
    signals = {
        "healthy": load_1797_normal(bearing_id),
        "ball": load_1797_ball(bearing_id),
        "inner_race": load_1797_inner(bearing_id),
        "outer_race": load_1797_outer(bearing_id),
    }

    raw_windows = {
        name: make_windows(signal, win=win, step=step).astype(np.float32)
        for name, signal in signals.items()
    }

    healthy_raw_split = chronological_split(
        raw_windows["healthy"],
        train_fraction=train_fraction,
        calibration_fraction=calibration_fraction,
    )

    scaler = StandardScaler()
    scaler.fit(healthy_raw_split.train)

    healthy = HealthySplit(
        train=scaler.transform(healthy_raw_split.train).astype(np.float32),
        calibration=scaler.transform(healthy_raw_split.calibration).astype(np.float32),
        test=scaler.transform(healthy_raw_split.test).astype(np.float32),
    )

    faults = {
        name: scaler.transform(windows).astype(np.float32)
        for name, windows in raw_windows.items()
        if name != "healthy"
    }

    return CWRUDataset(
        bearing_id=bearing_id,
        healthy=healthy,
        faults=faults,
        scaler=scaler,
    )


# ---------------------------------------------------------------------
# Detector implementations
# ---------------------------------------------------------------------

class Conv1DAutoencoderDetector:
    """
    Same Conv1D architecture as the original notebook.

    This is the ideal asset-specific baseline:
    one AE is trained using real healthy data from each CWRU operating condition.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 16,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 100,
        verbose: int = 0,
        seed: int = GLOBAL_SEED,
    ) -> None:
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.seed = seed
        self.model: Model | None = None

    @staticmethod
    def _as_3d(X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D windows, got shape {X.shape}.")
        return X[..., np.newaxis].astype(np.float32)

    def _build(self, sig_len: int) -> Model:
        inp = keras.Input(shape=(sig_len, 1), name="signal_in")

        x = layers.Conv1D(
            32, kernel_size=16, strides=2, padding="same", activation="relu"
        )(inp)
        x = layers.Conv1D(
            64, kernel_size=8, strides=2, padding="same", activation="relu"
        )(x)
        x = layers.Conv1D(
            128, kernel_size=4, strides=2, padding="same", activation="relu"
        )(x)

        conv_shape = tuple(int(v) for v in x.shape[1:])
        x = layers.Flatten()(x)
        latent = layers.Dense(
            self.latent_dim,
            activity_regularizer=keras.regularizers.l1(1.5e-4),
            name="latent",
        )(x)

        y = layers.Dense(conv_shape[0] * conv_shape[1], activation="relu")(latent)
        y = layers.Reshape(conv_shape)(y)
        y = layers.Conv1DTranspose(
            128, kernel_size=4, strides=2, padding="same", activation="relu"
        )(y)
        y = layers.Conv1DTranspose(
            64, kernel_size=8, strides=2, padding="same", activation="relu"
        )(y)
        y = layers.Conv1DTranspose(
            32, kernel_size=16, strides=2, padding="same", activation="relu"
        )(y)
        y = layers.Conv1D(
            1, kernel_size=1, padding="same", activation="linear", name="signal_out"
        )(y)

        out_len = int(y.shape[1])
        if out_len > sig_len:
            y = layers.Cropping1D((0, out_len - sig_len))(y)
        elif out_len < sig_len:
            y = layers.ZeroPadding1D((0, sig_len - out_len))(y)

        model = Model(inp, y, name="asset_specific_autoencoder")
        model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss="mse",
        )
        return model

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray | None = None,
    ) -> "Conv1DAutoencoderDetector":
        set_all_seeds(self.seed)

        X_train_3d = self._as_3d(X_train)
        X_val_3d = self._as_3d(X_val) if X_val is not None else None

        self.model = self._build(X_train.shape[1])

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=10,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-5,
            ),
        ]

        validation_data = (
            (X_val_3d, X_val_3d) if X_val_3d is not None else None
        )

        self.model.fit(
            X_train_3d,
            X_train_3d,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=self.verbose,
        )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Detector must be fitted before scoring.")

        X_3d = self._as_3d(X)
        X_rec = self.model.predict(X_3d, verbose=0)
        return np.mean((X_3d - X_rec) ** 2, axis=(1, 2))


class OneClassSVMDetector:
    """
    Asset-specific OC-SVM trained on real healthy windows.

    score_samples gives larger values for normal points, so we negate it to
    preserve the common convention: larger score = more anomalous.
    """

    def __init__(
        self,
        *,
        nu: float = 0.02,
        gamma: str | float = "scale",
        pca_components: int | None = None,
    ) -> None:
        steps: list[tuple[str, Any]] = []
        if pca_components is not None:
            steps.append(
                (
                    "pca",
                    PCA(
                        n_components=pca_components,
                        random_state=GLOBAL_SEED,
                    ),
                )
            )
        steps.append(("ocsvm", OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)))
        self.model = Pipeline(steps)

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray | None = None,
    ) -> "OneClassSVMDetector":
        self.model.fit(X_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)


class EllipticEnvelopeDetector:
    """
    Robust Gaussian baseline.

    PCA is strongly recommended because raw 1200-D covariance estimation is
    ill-conditioned when the number of healthy windows is limited.
    """

    def __init__(
        self,
        *,
        contamination: float = 0.02,
        pca_components: int = 16,
        support_fraction: float | None = None,
    ) -> None:
        self.model = Pipeline(
            [
                (
                    "pca",
                    PCA(
                        n_components=pca_components,
                        random_state=GLOBAL_SEED,
                    ),
                ),
                (
                    "elliptic",
                    EllipticEnvelope(
                        contamination=contamination,
                        support_fraction=support_fraction,
                        random_state=GLOBAL_SEED,
                    ),
                ),
            ]
        )

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray | None = None,
    ) -> "EllipticEnvelopeDetector":
        self.model.fit(X_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # score_samples: larger = more normal; negate for anomaly convention.
        return -self.model.score_samples(X)


# ---------------------------------------------------------------------
# Generic evaluation
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentResult:
    bearing_id: str
    detector: str
    threshold_percentile: float
    threshold: float
    healthy_far: float
    ball_dr: float
    inner_race_dr: float
    outer_race_dr: float
    n_train: int
    n_calibration: int
    n_healthy_test: int


def evaluate_detector(
    detector: AnomalyDetector,
    dataset: CWRUDataset,
    *,
    detector_name: str,
    threshold_percentile: float = 98.0,
) -> ExperimentResult:
    """
    Fair ideal-per-asset protocol.

    Training:
        healthy.train
    Model-selection / AE early stopping:
        healthy.calibration
    Threshold:
        percentile of detector scores on healthy.calibration
    Final FAR:
        healthy.test only
    Final DR:
        all fault windows
    """
    detector.fit(dataset.healthy.train, dataset.healthy.calibration)

    calibration_scores = detector.score(dataset.healthy.calibration)
    tau = float(np.percentile(calibration_scores, threshold_percentile))

    healthy_scores = detector.score(dataset.healthy.test)
    fault_scores = {
        name: detector.score(X)
        for name, X in dataset.faults.items()
    }

    return ExperimentResult(
        bearing_id=dataset.bearing_id,
        detector=detector_name,
        threshold_percentile=threshold_percentile,
        threshold=tau,
        healthy_far=float(np.mean(healthy_scores > tau)),
        ball_dr=float(np.mean(fault_scores["ball"] > tau)),
        inner_race_dr=float(np.mean(fault_scores["inner_race"] > tau)),
        outer_race_dr=float(np.mean(fault_scores["outer_race"] > tau)),
        n_train=len(dataset.healthy.train),
        n_calibration=len(dataset.healthy.calibration),
        n_healthy_test=len(dataset.healthy.test),
    )


def run_asset_specific_suite(
    model_factories: Dict[str, Callable[[], AnomalyDetector]],
    *,
    bearing_ids: tuple[str, ...] = ("1730", "1750", "1772", "1797"),
    win: int = 1200,
    step: int = 1200,
    threshold_percentile: float = 98.0,
) -> pd.DataFrame:
    """
    Train every model independently for every CWRU operating condition.

    The dataset is built once per bearing_id and reused unchanged for all
    detector factories, guaranteeing direct comparability.
    """
    rows: list[dict[str, Any]] = []

    for bearing_id in bearing_ids:
        dataset = build_cwru_dataset(
            bearing_id,
            win=win,
            step=step,
        )

        print(
            f"\n[{bearing_id} RPM] "
            f"train={len(dataset.healthy.train)}, "
            f"cal={len(dataset.healthy.calibration)}, "
            f"healthy_test={len(dataset.healthy.test)}"
        )

        for model_name, factory in model_factories.items():
            print(f"  Training {model_name}...")
            detector = factory()
            result = evaluate_detector(
                detector,
                dataset,
                detector_name=model_name,
                threshold_percentile=threshold_percentile,
            )
            rows.append(result.__dict__)

    df = pd.DataFrame(rows)

    metric_cols = [
        "healthy_far",
        "ball_dr",
        "inner_race_dr",
        "outer_race_dr",
    ]
    df[metric_cols] = 100.0 * df[metric_cols]
    return df


# ---------------------------------------------------------------------
# Example experiment configuration
# ---------------------------------------------------------------------

MODEL_FACTORIES: Dict[str, Callable[[], AnomalyDetector]] = {
    "Asset-specific AE": lambda: Conv1DAutoencoderDetector(
        latent_dim=16,
        epochs=100,
        batch_size=32,
        verbose=0,
    ),
    "Asset-specific OC-SVM": lambda: OneClassSVMDetector(
        nu=0.02,
        gamma="scale",
        pca_components=32,
    ),
    "Asset-specific Elliptic Envelope": lambda: EllipticEnvelopeDetector(
        contamination=0.02,
        pca_components=16,
    ),
}


# Run:
#
# ideal_results = run_asset_specific_suite(
#     MODEL_FACTORIES,
#     bearing_ids=("1730", "1750", "1772", "1797"),
#     win=1200,
#     step=1200,
#     threshold_percentile=98.0,
# )
#
# display(ideal_results)
#
# Summary across the four CWRU operating conditions:
#
# summary = (
#     ideal_results
#     .groupby("detector")[
#         ["healthy_far", "ball_dr", "inner_race_dr", "outer_race_dr"]
#     ]
#     .agg(["mean", "std"])
#     .round(2)
# )
# display(summary)
