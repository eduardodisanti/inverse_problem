#!/usr/bin/env python3
"""
End-to-end pipeline verification on simulated data only.

Requires TensorFlow but no external dataset.

Stage A is a regression test against numbers recorded in the deposited
notebook. bearing_autoencoder.ipynb stores, as executed output, the simulated
-domain result of the shared nominal representation:

    Threshold tau (p98 healthy MSE) = 0.115022
    Detection rate (MSE > tau):
      Healthy     :   2.0%
      Outer Race  : 100.0%
      Inner Race  : 100.0%
      Ball Fault  :  92.0%

This script rebuilds that experiment from bearing_simulator.py using the same
architecture, seeds and training schedule, and checks the result against those
recorded values. Reproducing them confirms that the simulator, the
architecture and the seeding policy in this deposit are intact.

Stages B and C then exercise the deployment claim itself:

    B. transfer the absolute OEM threshold to a differently-scaled field
       asset. Expected: the healthy false-alarm rate explodes -- the
       representation transfers, its error scale does not.
    C. commission a local operational radius on that field asset from a short
       unlabeled nominal stream. Expected: the false-alarm rate returns to
       roughly the 2% target while fault detection stays high.

Stage D exercises the asset-specific baseline harness from
generic_cwru_asset_specific_experiments.py (Conv1D AE, One-Class SVM, Elliptic
Envelope) so the comparison code path is covered too.

Stages B-D are direction-and-smoke tests on simulated field data, not
reproductions of the published CWRU or NASA IMS figures, which require the
external datasets.

    python verification/verify_pipeline.py
    python verification/verify_pipeline.py --quick   # skip stage D
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

# MUST be set before TensorFlow is imported.
#
# The autoencoder puts an L1 activity regularizer on its latent layer. Under
# Keras 3 that penalty is applied with different scaling and drives the latent
# to zero: the model degenerates to predicting the per-feature mean, val_loss
# plateaus at ~0.95, and every regime scores alike. The failure is silent --
# the healthy false-alarm rate still reads 2%, because a constant predictor
# satisfies its own percentile. Keras 2 is required. See REPRODUCIBILITY.md.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "code"))

from bearing_simulator import CWRU_PARAMS, BearingSignalSimulator  # noqa: E402
from operational_radius import (  # noqa: E402
    AdaptiveOperationalRadius,
    find_convergence_index,
)

SEED = 42
TARGET_PERCENTILE = 98.0
REGIMES = ("healthy", "outer_race", "inner_race", "ball_fault")

# Exactly the configuration recorded in bearing_autoencoder.ipynb.
N_TRAIN, N_VAL, N_TEST = 800, 200, 150
LATENT_DIM = 16
EPOCHS = 100
BATCH_SIZE = 32

# Stage D only needs to prove the baseline harness runs and returns finite
# metrics, so it is deliberately cheaper than stage A.
BASELINE_WINDOWS = 600
BASELINE_EPOCHS = 30

# Values stored as executed output in bearing_autoencoder.ipynb (cell 15).
REFERENCE_TAU = 0.115022
REFERENCE_DETECTION = {
    "healthy": 0.020,
    "outer_race": 1.000,
    "inner_race": 1.000,
    "ball_fault": 0.920,
}
# Tolerances absorb BLAS/backend nondeterminism across platforms. The original
# run used TensorFlow 2.16.2 on Apple Metal.
TAU_RTOL = 0.10
DETECTION_ATOL = 0.05

# Standardized input has unit variance, so a collapsed autoencoder that
# predicts the per-feature mean scores val_loss ~ 1.0. The reference run
# reached 0.066. Anything above this is a failed training run, not a result.
COLLAPSE_VAL_LOSS = 0.50

_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    line = f"  [{status}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    if not condition:
        _failures.append(name)


def section(title: str) -> None:
    print()
    print(title)
    print("-" * 70)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip stage D (asset-specific baseline harness)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PIPELINE VERIFICATION  (simulated data only)")
    print("=" * 70)

    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from tensorflow import keras
    from tensorflow.keras import Model, layers

    keras_version = getattr(keras, "__version__", None)
    if keras_version is None:  # tf_keras exposes it separately
        try:
            import tf_keras

            keras_version = tf_keras.__version__
        except Exception:  # noqa: BLE001
            keras_version = "unknown"
    legacy = os.environ.get("TF_USE_LEGACY_KERAS") == "1"

    print(f"TensorFlow {tf.__version__}, Keras {keras_version}")
    print(f"TF_USE_LEGACY_KERAS={os.environ.get('TF_USE_LEGACY_KERAS')}")
    print("Reference run: TensorFlow 2.16.2 (Apple Metal), Keras 2")

    if str(keras_version).startswith("3.") or not legacy:
        print()
        print("  " + "!" * 66)
        print("  Keras 3 detected. This package requires Keras 2: under Keras 3")
        print("  the latent L1 activity regularizer collapses the autoencoder")
        print("  and every result below is meaningless. Install tf-keras and")
        print("  re-run:")
        print()
        print("      pip install tf-keras")
        print("      TF_USE_LEGACY_KERAS=1 python verification/verify_pipeline.py")
        print("  " + "!" * 66)

    started = time.time()

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    sim = BearingSignalSimulator(CWRU_PARAMS)
    sig_len = len(sim.t)

    # -----------------------------------------------------------------
    section("A. Regression test against the notebook's recorded output")
    # -----------------------------------------------------------------
    healthy = np.array(
        [sim.sample("healthy", mc=True) for _ in range(N_TRAIN + N_VAL)]
    )
    scaler = StandardScaler()
    healthy_scaled = scaler.fit_transform(healthy)

    X_train = healthy_scaled[:N_TRAIN, :, np.newaxis].astype("float32")
    X_val = healthy_scaled[N_TRAIN:, :, np.newaxis].astype("float32")

    test_blocks, labels = [], []
    for i, regime in enumerate(REGIMES):
        block = np.array([sim.sample(regime, mc=True) for _ in range(N_TEST)])
        test_blocks.append(scaler.transform(block))
        labels.append(np.full(N_TEST, i))
    X_test = np.concatenate(test_blocks)[:, :, np.newaxis].astype("float32")
    y_test = np.concatenate(labels)

    print(f"  train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

    def build_autoencoder(length: int, latent_dim: int) -> Model:
        inp = keras.Input(shape=(length, 1), name="signal_in")
        x = layers.Conv1D(32, 16, strides=2, padding="same", activation="relu")(inp)
        x = layers.Conv1D(64, 8, strides=2, padding="same", activation="relu")(x)
        x = layers.Conv1D(128, 4, strides=2, padding="same", activation="relu")(x)
        conv_shape = x.shape[1:]
        x = layers.Flatten()(x)
        latent = layers.Dense(
            latent_dim,
            activity_regularizer=keras.regularizers.l1(1.5e-4),
            name="latent",
        )(x)
        y = layers.Dense(conv_shape[0] * conv_shape[1], activation="relu")(latent)
        y = layers.Reshape(conv_shape)(y)
        y = layers.Conv1DTranspose(128, 4, strides=2, padding="same", activation="relu")(y)
        y = layers.Conv1DTranspose(64, 8, strides=2, padding="same", activation="relu")(y)
        y = layers.Conv1DTranspose(32, 16, strides=2, padding="same", activation="relu")(y)
        y = layers.Conv1D(1, 1, padding="same", activation="linear", name="signal_out")(y)
        model = Model(inp, y, name="bearing_autoencoder")
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        return model

    ae = build_autoencoder(sig_len, LATENT_DIM)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
        ),
    ]
    history = ae.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )
    best_val_loss = float(min(history.history["val_loss"]))
    print(
        f"  trained {len(history.history['loss'])} epochs, "
        f"best val_loss {best_val_loss:.6f}"
    )

    # A collapsed autoencoder predicts the per-feature mean of standardized
    # input, giving val_loss ~ 1.0. The reference run reached 0.066.
    healthy_training = best_val_loss < COLLAPSE_VAL_LOSS
    check(
        "the shared representation actually trained (no latent collapse)",
        healthy_training,
        f"best val_loss {best_val_loss:.4f}, collapse threshold "
        f"{COLLAPSE_VAL_LOSS}, reference 0.066",
    )
    if not healthy_training:
        latent = Model(ae.input, ae.get_layer("latent").output)
        z = latent.predict(X_val, verbose=0)
        print(f"  latent activation sd = {float(z.std()):.2e} (collapsed to zero)")
        print("  This is the Keras 3 failure mode. Everything below is void.")
        print("  Install tf-keras and set TF_USE_LEGACY_KERAS=1.")
        print()
        print("=" * 70)
        print("RESULT: FAIL — training collapsed; see REPRODUCIBILITY.md")
        return 1

    reconstructed = ae.predict(X_test, verbose=0)
    mse = np.mean((X_test - reconstructed) ** 2, axis=(1, 2))
    tau = float(np.percentile(mse[y_test == 0], TARGET_PERCENTILE))

    rel = abs(tau - REFERENCE_TAU) / REFERENCE_TAU
    check(
        "simulated-domain threshold matches the recorded value",
        rel < TAU_RTOL,
        f"tau={tau:.6f} vs recorded {REFERENCE_TAU:.6f} ({rel:.1%} apart)",
    )

    print()
    print(f"  {'regime':<12} {'observed':>9} {'recorded':>9}")
    observed = {}
    for i, regime in enumerate(REGIMES):
        observed[regime] = float(np.mean(mse[y_test == i] > tau))
        print(
            f"  {regime:<12} {observed[regime]:>8.1%} "
            f"{REFERENCE_DETECTION[regime]:>8.1%}"
        )
    for regime in REGIMES:
        check(
            f"{regime}: detection rate matches the recorded value",
            abs(observed[regime] - REFERENCE_DETECTION[regime]) <= DETECTION_ATOL,
            f"delta = {observed[regime] - REFERENCE_DETECTION[regime]:+.1%}",
        )

    # -----------------------------------------------------------------
    section("B. Direct transfer of the absolute threshold to a field asset")
    # -----------------------------------------------------------------
    # A different physical unit: distinct harmonic signature and gain, the
    # unit-to-unit variation the manuscript's second research question targets.
    np.random.seed(SEED + 1)
    field_signature = np.random.uniform(0.7, 1.4, size=5)
    field_gain = 1.35

    def field_draw(regime: str, n: int, seed: int) -> np.ndarray:
        np.random.seed(seed)
        return np.stack(
            [
                field_gain * sim.sample(regime, asset_signature=field_signature)
                for _ in range(n)
            ]
        ).astype(np.float32)

    def residuals(windows: np.ndarray) -> np.ndarray:
        X = scaler.transform(windows)[:, :, np.newaxis].astype("float32")
        return np.mean((X - ae.predict(X, verbose=0)) ** 2, axis=(1, 2))

    field_res = {
        regime: residuals(field_draw(regime, 300, SEED + 10 + i))
        for i, regime in enumerate(REGIMES)
    }

    far_transfer = float(np.mean(field_res["healthy"] > tau))
    print(f"  OEM absolute radius            = {tau:.6f}")
    print(f"  field healthy residual median  = {np.median(field_res['healthy']):.6f}")
    check(
        "absolute threshold transfer inflates the false-alarm rate",
        far_transfer > 0.20,
        f"healthy FAR = {far_transfer:.1%} against a 2% target",
    )
    print("  Reproduces the manuscript's finding that the error scale does not")
    print("  transfer even when the representation does.")

    # -----------------------------------------------------------------
    section("C. Local operational-radius commissioning on the field asset")
    # -----------------------------------------------------------------
    stream = field_res["healthy"][:120]
    estimator = AdaptiveOperationalRadius(
        percentile=TARGET_PERCENTILE, window_size=len(stream), min_samples=20
    )
    tau_trace = estimator.warmup_trace(stream)
    index = find_convergence_index(
        tau_trace, lookback=10, tolerance=0.01, consecutive=10, min_index=39
    )
    check(
        "commissioning converges on the field asset",
        index is not None,
        f"convergence index = {index}",
    )

    tau_local = float(tau_trace[index if index is not None else -1])
    oracle = float(np.percentile(field_res["healthy"], TARGET_PERCENTILE))
    print(f"  local operational radius   = {tau_local:.6f}")
    print(f"  full-data field oracle p98 = {oracle:.6f}")
    print(f"  agreement with oracle      = {abs(tau_local - oracle) / oracle:.2%}")

    check(
        "local radius is closer to the field oracle than the OEM radius is",
        abs(tau_local - oracle) < abs(tau - oracle),
        f"|local-oracle|={abs(tau_local - oracle):.6f} vs "
        f"|OEM-oracle|={abs(tau - oracle):.6f}",
    )

    far_local = float(np.mean(field_res["healthy"] > tau_local))
    check(
        "commissioning restores the false-alarm rate toward target",
        far_local < 0.10,
        f"FAR {far_transfer:.1%} -> {far_local:.1%}",
    )

    print()
    print(f"  {'regime':<12} {'detection':>10} {'vs healthy FAR':>16}")
    detection = {}
    for regime in REGIMES[1:]:
        detection[regime] = float(np.mean(field_res[regime] > tau_local))
        lift = detection[regime] / max(far_local, 1e-9)
        print(f"  {regime:<12} {detection[regime]:>9.1%} {lift:>15.0f}x")
    check(
        "every fault regime is detected far above the healthy false-alarm rate",
        all(v > 10 * far_local for v in detection.values()),
        f"weakest lift = {min(detection.values()) / max(far_local, 1e-9):.0f}x "
        f"over a {far_local:.1%} FAR",
    )
    check(
        "race faults, the strongly impulsive regimes, stay high",
        detection["outer_race"] > 0.60 and detection["inner_race"] > 0.60,
        f"outer={detection['outer_race']:.1%}, inner={detection['inner_race']:.1%}",
    )
    print()
    print("  Ball fault is the weakest regime here, as in the simulated-domain")
    print("  reference (92% vs 100% for the races). This synthetic field asset")
    print("  also applies a 1.35x gain and a randomized harmonic signature, a")
    print("  harsher shift than the CWRU operating conditions, so absolute")
    print("  rates sit below the published CWRU figures.")

    # -----------------------------------------------------------------
    if not args.quick:
        section("D. Asset-specific baseline harness")

        from generic_cwru_asset_specific_experiments import (
            Conv1DAutoencoderDetector,
            CWRUDataset,
            EllipticEnvelopeDetector,
            HealthySplit,
            OneClassSVMDetector,
            chronological_split,
            evaluate_detector,
        )

        np.random.seed(SEED + 50)
        raw = {
            regime: np.stack(
                [sim.sample(regime, mc=True) for _ in range(BASELINE_WINDOWS)]
            ).astype(np.float32)
            for regime in REGIMES
        }
        split = chronological_split(raw["healthy"])
        baseline_scaler = StandardScaler().fit(split.train)

        dataset = CWRUDataset(
            bearing_id="SIM",
            healthy=HealthySplit(
                train=baseline_scaler.transform(split.train).astype(np.float32),
                calibration=baseline_scaler.transform(split.calibration).astype(np.float32),
                test=baseline_scaler.transform(split.test).astype(np.float32),
            ),
            faults={
                "ball": baseline_scaler.transform(raw["ball_fault"]).astype(np.float32),
                "inner_race": baseline_scaler.transform(raw["inner_race"]).astype(np.float32),
                "outer_race": baseline_scaler.transform(raw["outer_race"]).astype(np.float32),
            },
            scaler=baseline_scaler,
        )
        print(
            f"  splits: train={len(dataset.healthy.train)}, "
            f"cal={len(dataset.healthy.calibration)}, "
            f"test={len(dataset.healthy.test)}"
        )

        factories = {
            "Asset-specific AE": lambda: Conv1DAutoencoderDetector(
                latent_dim=LATENT_DIM, epochs=BASELINE_EPOCHS, verbose=0
            ),
            "Asset-specific OC-SVM": lambda: OneClassSVMDetector(
                nu=0.02, gamma="scale", pca_components=32
            ),
            "Asset-specific Elliptic Envelope": lambda: EllipticEnvelopeDetector(
                contamination=0.02, pca_components=16
            ),
        }

        print()
        print(f"  {'detector':<34} {'FAR':>7} {'ball':>7} {'inner':>7} {'outer':>7}")
        for name, factory in factories.items():
            result = evaluate_detector(
                factory(),
                dataset,
                detector_name=name,
                threshold_percentile=TARGET_PERCENTILE,
            )
            print(
                f"  {name:<34} {result.healthy_far:>6.1%} {result.ball_dr:>6.1%} "
                f"{result.inner_race_dr:>6.1%} {result.outer_race_dr:>6.1%}"
            )
            check(
                f"{name}: runs and produces finite metrics",
                all(
                    np.isfinite(v)
                    for v in (
                        result.threshold,
                        result.healthy_far,
                        result.ball_dr,
                        result.inner_race_dr,
                        result.outer_race_dr,
                    )
                ),
            )
        print()
        print("  Absolute rates here are not comparable to the published CWRU")
        print("  table: these are simulated windows, not CWRU recordings. The")
        print("  check is that the harness runs and returns finite metrics.")

    print()
    print("=" * 70)
    print(f"elapsed {time.time() - started:.0f}s")
    if _failures:
        print(f"RESULT: FAIL — {len(_failures)} check(s) failed:")
        for name in _failures:
            print(f"  - {name}")
        return 1
    print("RESULT: PASS — full pipeline runs end to end on simulated data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
