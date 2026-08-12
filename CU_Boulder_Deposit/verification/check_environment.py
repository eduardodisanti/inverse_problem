#!/usr/bin/env python3
"""
Environment check.

Reports the interpreter and dependency versions actually present, compares
them against requirements.txt, and reports whether the two external datasets
are resolvable. Exits non-zero only if a required package is missing or
unimportable; dataset absence is reported but not treated as failure, since
the self-contained verification does not need it.

    python verification/check_environment.py
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "code"))

# import name -> distribution name used in requirements.txt
REQUIRED = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "tensorflow": "tensorflow",
}


def main() -> int:
    print("=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)
    print(f"Python      {sys.version.split()[0]}  ({platform.python_implementation()})")
    print(f"Platform    {platform.platform()}")
    print(f"Package     {PACKAGE_ROOT}")
    print()

    failures = []
    print("Dependencies")
    print("-" * 70)
    for import_name, dist_name in REQUIRED.items():
        try:
            module = importlib.import_module(import_name)
        except Exception as exc:  # noqa: BLE001 - report any import failure
            print(f"  {dist_name:<16} MISSING  ({type(exc).__name__}: {exc})")
            failures.append(dist_name)
            continue
        version = getattr(module, "__version__", "unknown")
        print(f"  {dist_name:<16} {version}")

    print()
    print("Keras version  (CRITICAL)")
    print("-" * 70)
    try:
        import tensorflow as tf

        version = getattr(tf.keras, "__version__", None)
        if version is None:
            try:
                import tf_keras

                version = tf_keras.__version__
            except Exception:  # noqa: BLE001
                version = "unknown"

        legacy_flag = os.environ.get("TF_USE_LEGACY_KERAS")
        print(f"  keras in use     {version}")
        print(f"  TF_USE_LEGACY_KERAS  {legacy_flag or '(unset)'}")

        try:
            import tf_keras  # noqa: F401

            tf_keras_installed = True
        except Exception:  # noqa: BLE001
            tf_keras_installed = False
        print(f"  tf-keras installed   {tf_keras_installed}")

        if str(version).startswith("2."):
            print()
            print("  OK — Keras 2, matching the environment that produced the")
            print("  published results.")
        else:
            print()
            print("  PROBLEM — Keras 3 is active. This package requires Keras 2.")
            print("  Under Keras 3 the L1 activity regularizer on the latent")
            print("  layer drives it to zero, the autoencoder degenerates to")
            print("  predicting the mean, and no fault is detected. The failure")
            print("  is silent: the healthy false-alarm rate still reads 2%.")
            print()
            print("  Fix:")
            print("      pip install tf-keras")
            print("      export TF_USE_LEGACY_KERAS=1")
            print()
            print("  See REPRODUCIBILITY.md, Known Limitations item 1.")
            failures.append("keras-2")
    except Exception as exc:  # noqa: BLE001
        print(f"  unavailable ({exc})")

    print()
    print("External datasets")
    print("-" * 70)
    try:
        import data_paths

        print("  " + data_paths.describe().replace("\n", "\n  "))
    except Exception as exc:  # noqa: BLE001
        print(f"  data_paths unavailable ({exc})")
        failures.append("data_paths")

    print()
    if failures:
        print(f"RESULT: FAIL — {', '.join(failures)}")
        print("Install with:  pip install -r requirements.txt")
        return 1

    print("RESULT: PASS — all required packages importable.")
    print("Dataset absence above is expected; the self-contained")
    print("verification scripts do not need external data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
