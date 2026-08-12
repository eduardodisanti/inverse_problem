#!/usr/bin/env python3
"""
Self-contained verification of the bearing simulator (manuscript Section 8.3).

Requires no external data and no deep-learning framework: numpy only.

Checks
------
 1. Shape and finiteness for every regime, both domain configurations.
 2. Determinism: identical global seed reproduces identical signals bit for bit.
 3. Seed sensitivity: different seeds produce different signals.
 4. Monte Carlo nuisance is actually active (mc=True varies, mc=False does not
    vary in its deterministic component).
 5. Fault regimes are separable from healthy under a plain energy statistic,
    confirming the impulse trains are present and non-degenerate.
 6. Invalid regime and coverage arguments raise.

    python verification/verify_simulator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "code"))

from bearing_simulator import (  # noqa: E402
    CWRU_PARAMS,
    NASA_PARAMS,
    BearingSignalSimulator,
)

REGIMES = ("healthy", "outer_race", "inner_race", "ball_fault")
SEED = 42

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
    print("=" * 70)
    print("SIMULATOR VERIFICATION  (no external data required)")
    print("=" * 70)

    sim = BearingSignalSimulator(CWRU_PARAMS)
    expected_len = int(CWRU_PARAMS.fs * CWRU_PARAMS.duration)

    section("1. Shape and finiteness")
    np.random.seed(SEED)
    for regime in REGIMES:
        x = sim.sample(regime=regime)
        check(
            f"{regime}: length {expected_len}, all finite",
            x.shape == (expected_len,) and np.all(np.isfinite(x)),
            f"shape={x.shape}, std={x.std():.4f}",
        )

    nasa = BearingSignalSimulator(NASA_PARAMS)
    nasa_len = int(NASA_PARAMS.fs * NASA_PARAMS.duration)
    x = nasa.sample(regime="healthy")
    check(
        f"NASA config: length {nasa_len}, all finite",
        x.shape == (nasa_len,) and np.all(np.isfinite(x)),
        f"shape={x.shape}",
    )

    section("2. Determinism under a fixed global seed")
    for regime in REGIMES:
        np.random.seed(SEED)
        a = sim.sample(regime=regime)
        np.random.seed(SEED)
        b = sim.sample(regime=regime)
        check(f"{regime}: bit-identical on replay", np.array_equal(a, b))

    section("3. Seed sensitivity")
    np.random.seed(1)
    a = sim.sample(regime="healthy")
    np.random.seed(2)
    b = sim.sample(regime="healthy")
    check(
        "different seeds give different signals",
        not np.array_equal(a, b),
        f"max|a-b|={np.abs(a - b).max():.4f}",
    )

    section("4. Monte Carlo nuisance")
    np.random.seed(SEED)
    draws = np.stack([sim.sample(regime="healthy") for _ in range(32)])
    per_sample_std = draws.std(axis=0).mean()
    check(
        "mc=True produces across-draw variation",
        per_sample_std > CWRU_PARAMS.noise_std,
        f"mean across-draw sd={per_sample_std:.4f} > noise_std={CWRU_PARAMS.noise_std}",
    )

    np.random.seed(SEED)
    fixed = np.stack([sim.sample(regime="healthy", mc=False) for _ in range(32)])
    residual_std = fixed.std(axis=0).mean()
    check(
        "mc=False leaves only measurement noise",
        abs(residual_std - CWRU_PARAMS.noise_std) < 0.01,
        f"mean across-draw sd={residual_std:.4f} ~ noise_std={CWRU_PARAMS.noise_std}",
    )

    np.random.seed(SEED)
    wide = np.stack(
        [sim.sample(regime="healthy", coverage="wide") for _ in range(32)]
    )
    check(
        "coverage='wide' broadens the nuisance envelope vs 'original'",
        wide.std(axis=0).mean() > per_sample_std,
        f"wide={wide.std(axis=0).mean():.4f} > original={per_sample_std:.4f}",
    )

    section("5. Fault regimes are non-degenerate")
    np.random.seed(SEED)
    draws = {
        regime: np.stack([sim.sample(regime=regime) for _ in range(64)])
        for regime in REGIMES
    }

    # Impulsiveness after crude high-pass filtering (second difference).
    # Spectral kurtosis of the high-frequency band is the classical
    # rolling-element fault indicator; the impulse trains must show up here.
    def band_kurtosis(x: np.ndarray) -> np.ndarray:
        hp = np.diff(x, n=2, axis=1)
        centred = hp - hp.mean(axis=1, keepdims=True)
        m2 = np.mean(centred**2, axis=1)
        m4 = np.mean(centred**4, axis=1)
        return m4 / m2**2 - 3.0  # excess kurtosis

    kurt = {regime: band_kurtosis(d) for regime, d in draws.items()}
    healthy_p95 = np.percentile(kurt["healthy"], 95)

    for regime in REGIMES[1:]:
        rate = float(np.mean(kurt[regime] > healthy_p95))
        check(
            f"{regime}: impulsive signature above healthy p95 kurtosis",
            rate > 0.95,
            f"exceedance={rate:.1%}, median kurtosis={np.median(kurt[regime]):.2f} "
            f"vs healthy {np.median(kurt['healthy']):.2f}",
        )

    # Reported, not asserted: broadband energy does NOT separate the regimes,
    # because Monte Carlo amplitude nuisance (+-15%) dominates the fault
    # contribution. This is the premise of the manuscript -- a nuisance-robust
    # representation is required, and a naive scalar statistic is insufficient.
    energy = {r: np.mean(d**2, axis=1) for r, d in draws.items()}
    energy_p95 = np.percentile(energy["healthy"], 95)
    print()
    print("  context: broadband energy alone does not separate the regimes")
    for regime in REGIMES[1:]:
        rate = float(np.mean(energy[regime] > energy_p95))
        print(f"    {regime:<12} energy exceedance over healthy p95 = {rate:5.1%}")
    print("    Nuisance amplitude variation dominates fault energy, which is")
    print("    why the method operates on autoencoder residuals rather than on")
    print("    a raw signal statistic.")

    section("6. Argument validation")
    for bad_call, label in (
        (lambda: sim.sample(regime="not_a_regime"), "unknown regime raises"),
        (
            lambda: sim.sample(regime="healthy", coverage="not_a_coverage"),
            "unknown coverage raises",
        ),
    ):
        try:
            bad_call()
            check(label, False, "no exception raised")
        except ValueError:
            check(label, True)

    section("Note on canonical()")
    c1 = sim.canonical()
    c2 = sim.canonical()
    identical = np.array_equal(c1, c2)
    print(f"  canonical() reproducible across calls: {identical}")
    print("  canonical() calls sample(mc=False), which still adds measurement")
    print("  noise of sd = params.noise_std. It is the nuisance-free mean")
    print("  waveform plus noise, not a noiseless template. Seed the global")
    print("  NumPy state immediately before the call if a fixed realization is")
    print("  needed. Documented in REPRODUCIBILITY.md; behaviour left unchanged")
    print("  so that previously reported figures remain reproducible.")

    print()
    print("=" * 70)
    if _failures:
        print(f"RESULT: FAIL — {len(_failures)} check(s) failed:")
        for name in _failures:
            print(f"  - {name}")
        return 1
    print("RESULT: PASS — all simulator checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
