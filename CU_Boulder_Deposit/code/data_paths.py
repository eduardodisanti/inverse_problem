"""
Central resolution of external dataset locations.

Neither the CWRU nor the NASA IMS raw data is redistributed with this
deposit (see data/README.md). This module is the single place where their
location is resolved, so that no notebook or script needs a hard-coded
relative path.

Resolution order for each dataset, first hit wins:

    1. Environment variable  (CWRU_DATA_ROOT / NASA_IMS_DATA_ROOT)
    2. In-package location   (<deposit>/data/CWRU  /  <deposit>/data/NASA_IMS)
    3. Legacy sibling layout (../CWRU_Bearing_NumPy-main/Data
                              ../NASA_Bearing/IMS)

The legacy entries reproduce the paths used while the experiments were
originally run, so an existing working tree keeps functioning unchanged.

Usage
-----
    from data_paths import cwru_root, nasa_ims_root
    data_root = cwru_root()          # raises with guidance if absent
    data_root = cwru_root(required=False)   # returns None instead
"""

from __future__ import annotations

import os
from pathlib import Path

# <deposit>/code/data_paths.py  ->  <deposit>
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

CWRU_ENV_VAR = "CWRU_DATA_ROOT"
NASA_ENV_VAR = "NASA_IMS_DATA_ROOT"

CWRU_CANDIDATES = (
    PACKAGE_ROOT / "data" / "CWRU",
    PACKAGE_ROOT.parent / "CWRU_Bearing_NumPy-main" / "Data",
    PACKAGE_ROOT / "code" / ".." / ".." / "CWRU_Bearing_NumPy-main" / "Data",
)

NASA_CANDIDATES = (
    PACKAGE_ROOT / "data" / "NASA_IMS",
    PACKAGE_ROOT.parent / "NASA_Bearing" / "IMS",
)

_CWRU_HELP = f"""
CWRU data not found.

This deposit does not redistribute the CWRU Bearing Data Center archive.
The loaders expect the NumPy (.npz) conversion of the Drive-End data, laid
out one directory per operating condition:

    <root>/1730 RPM/1730_Normal.npz
    <root>/1730 RPM/1730_B_*_DE12.npz
    <root>/1730 RPM/1730_IR_*_DE12.npz
    <root>/1730 RPM/1730_OR*@*_DE12.npz
    <root>/1750 RPM/...
    <root>/1772 RPM/...
    <root>/1797 RPM/...

Each .npz must expose the key "DE" (drive-end channel).

Point the package at your copy in either way:

    export {CWRU_ENV_VAR}=/path/to/CWRU_Bearing_NumPy-main/Data

or place/symlink it at:

    {PACKAGE_ROOT / "data" / "CWRU"}

See data/README.md for the download and conversion instructions.
""".strip()

_NASA_HELP = f"""
NASA IMS data not found.

This deposit does not redistribute the NASA Prognostics Data Repository IMS
archive. The loader expects the extracted experiment directories:

    <root>/2nd_test/2004.02.12.10.32.39
    <root>/2nd_test/2004.02.12.10.42.39
    ...

Point the package at your copy in either way:

    export {NASA_ENV_VAR}=/path/to/NASA_Bearing/IMS

or place/symlink it at:

    {PACKAGE_ROOT / "data" / "NASA_IMS"}

See data/README.md for the download instructions.
""".strip()


def _resolve(env_var, candidates, help_text, required):
    override = os.environ.get(env_var)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(
                f"{env_var} is set to {path}, which is not a directory.\n\n"
                f"{help_text}"
            )
        return path

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved

    if required:
        raise FileNotFoundError(help_text)
    return None


def cwru_root(required: bool = True) -> Path | None:
    """Directory holding the CWRU .npz conversion, or None if absent."""
    return _resolve(CWRU_ENV_VAR, CWRU_CANDIDATES, _CWRU_HELP, required)


def nasa_ims_root(required: bool = True) -> Path | None:
    """Directory holding the extracted NASA IMS experiments, or None."""
    return _resolve(NASA_ENV_VAR, NASA_CANDIDATES, _NASA_HELP, required)


def describe() -> str:
    """Human-readable summary of what is and is not currently resolvable."""
    lines = [f"Package root: {PACKAGE_ROOT}"]
    for label, fn, env in (
        ("CWRU", cwru_root, CWRU_ENV_VAR),
        ("NASA IMS", nasa_ims_root, NASA_ENV_VAR),
    ):
        found = fn(required=False)
        if found is None:
            lines.append(f"{label:<9} NOT FOUND  (set {env} or populate data/)")
        else:
            lines.append(f"{label:<9} {found}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
