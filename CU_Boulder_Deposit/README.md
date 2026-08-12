# Fleet-Wide Condition Monitoring via Shared Nominal Representations and Local Radius Calibration

**Enabling OEM-Centralized, Fleet-Deployed CBM**

Eduardo Di Santi · Department of Applied Mathematics · University of Colorado Boulder
<eduardo.disanti@colorado.edu>

> Working paper (preprint). Not peer reviewed.

---

## Overview

Condition monitoring is normally deployed one asset at a time: collect data from
each unit, fit a detector to it, engineer its thresholds. Effort therefore grows
with fleet size. This work asks whether representation learning can be decoupled
from deployment — **one representation shared fleet-wide, each asset identifying
only its own local operational radius at commissioning.**

For each asset family a canonical nominal representation, a convolutional
autoencoder, is trained once offline on simulated healthy signals and
transferred unchanged to every deployed unit. Each unit then estimates a single
scalar — its operational radius — as an empirical residual quantile from a short
unlabeled nominal stream, stopped automatically by a convergence criterion.

The contribution is a **deployment methodology**, not a new anomaly-detection
architecture. The transferable artifact is a validated shared nominal
representation; the only asset-specific quantity estimated at deployment is a
scalar radius.

---

## Core finding

> The representation transfers. Its error scale does not.

Transferring the absolute simulator threshold to CWRU flagged **every** healthy
window as anomalous. The same representation, paired with a locally commissioned
radius, works:

| Result | Value |
|---|---|
| Radius agreement with the complete-data oracle | within **0.83%** |
| Commissioning length | **55 windows** on average |
| Healthy false-alarm rate | **2.60%** |
| Ball-fault detection | **99.57%** |
| Inner- and outer-race detection | **100%** |
| Repeated CWRU commissioning realizations that converged | **20 / 20** |
| NASA IMS: lead time before documented end of run | **45.4 h** on average |

Edge cost: the shared model has 847,601 trainable parameters (3.39 MB float32),
median CPU inference latency of 6.26 ms per 1,200-sample window against a 100 ms
acquisition interval, and under 1 kB of per-asset calibration state. No
per-asset training and no fault labels are required after installation.

---

## Method

The forward deployment problem is that a detector calibrated on one unit does
not transfer to another, because nominal residual scale is unit-specific even
when nominal *structure* is shared.

**Shared stage (once, offline, at the OEM).** Train a Conv1D autoencoder on
simulated healthy signals only, with Monte Carlo nuisance covering amplitude,
phase, shaft speed, offset, envelope and dropout variation. Reconstruction error
becomes an implicit membership score for the nominal manifold.

**Local stage (once, per asset, at commissioning).** The quantity shared
fleet-wide is the target nominal coverage (98%), not a threshold. Each asset
accumulates residuals from its own unlabeled nominal stream and tracks the
empirical percentile:

```
tau_k(t) = empirical 98th percentile of the local nominal residuals
C(t)     = |tau_k(t) - tau_k(t-L)| / max(|tau_k(t-L)|, eps)
```

Commissioning stops the first time `C(t) < tolerance` holds for `consecutive`
successive updates. The radius is then frozen and monitoring begins. An
observation is anomalous when its residual exceeds the frozen radius; a
persistent departure is declared when at least 8 of the last 10 recordings do.

---

## Repository structure

```
.
├── README.md                  this file
├── REPRODUCIBILITY.md         what can be verified, how, and known limitations
├── resources.txt              full deposit manifest
├── requirements.txt           Python dependencies
├── LICENSE                    MIT, covering code/
│
├── manuscript/
│   ├── manuscript_draft.pdf   the item of record, 22 pp., 1 figure, 8 tables
│   ├── manuscript_draft.tex   LaTeX source
│   ├── trbunofficial.cls      document class
│   ├── references.bib         bibliography
│   ├── cwru_tau_convergence.pdf   Figure 1
│   └── *.png                  supplementary graphics, not used by the .tex
│
├── code/
│   ├── bearing_simulator.py   physics-motivated signal generator
│   ├── operational_radius.py  online radius estimator + convergence rule
│   ├── data_paths.py          external dataset resolution
│   ├── cwru_loader.py         CWRU loading and windowing
│   ├── generic_cwru_asset_specific_experiments.py   baseline detector harness
│   ├── bearing_autoencoder.ipynb                    shared representation
│   ├── bearing_autoencoder_cwru_per_asset.ipynb     CWRU commissioning
│   ├── bearing_autoencoder_NASA_per_asset.ipynb     NASA IMS run-to-failure
│   └── generic_cwru_asset_specific_experiments.ipynb    baseline driver
│
├── verification/
│   ├── check_environment.py       dependency and dataset report
│   ├── verify_simulator.py        simulator checks, NumPy only
│   ├── verify_commissioning.py    radius and convergence checks, NumPy only
│   └── verify_pipeline.py         end-to-end, needs TensorFlow
│
└── data/
    └── README.md              external dataset acquisition instructions
```

---

## Quick start

> **Keras 2 is required.** Under Keras 3 the L1 activity regularizer on the
> autoencoder's latent layer drives it to zero: the model degenerates to
> predicting the mean and detects nothing, *while still reporting a 2% healthy
> false-alarm rate*. TensorFlow 2.16+ defaults to Keras 3. See
> `REPRODUCIBILITY.md` Section 0.

```bash
pip install -r requirements.txt        # includes tf-keras
export TF_USE_LEGACY_KERAS=1

python verification/check_environment.py      # confirms Keras 2 is active
python verification/verify_simulator.py       # ~5 s,   no external data
python verification/verify_commissioning.py   # ~10 s,  no external data
python verification/verify_pipeline.py        # ~2-3 min, no external data
```

None of these require a download, a dataset licence or a GPU. The last one
retrains the shared representation from the simulator and checks it against the
threshold and detection rates recorded in `bearing_autoencoder.ipynb`, then
reproduces the threshold-transfer failure and the commissioning fix on a
simulated field asset. Add `--quick` to skip the baseline-harness stage.

To reproduce the CWRU and NASA IMS results you must obtain both datasets
yourself; see `data/README.md`, then point the package at them:

```bash
export CWRU_DATA_ROOT=/path/to/CWRU_Bearing_NumPy-main/Data
export NASA_IMS_DATA_ROOT=/path/to/NASA_Bearing/IMS
jupyter lab code/
```

Run `bearing_autoencoder.ipynb` first: it produces the shared representation the
other two notebooks consume. Every notebook opens with an environment guard
cell — run it before anything else.

---

## Data availability

Neither dataset is redistributed here. Both are third-party, publicly available,
and remain governed by their original providers' terms.

- **CWRU Bearing Data Center**, Drive-End data, four operating conditions
  (1730, 1750, 1772, 1797 RPM) treated as independent deployment environments.
  <https://engineering.case.edu/bearingdatacenter>
- **NASA IMS bearing run-to-failure**, Set No. 2, four independently monitored
  bearings, 984 chronological recordings each at 10-minute intervals.
  <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>

All calibration and evaluation operate on exported one-dimensional residual
streams once the shared representation has produced them, so raw data is needed
only once, at the residual-extraction step.

---

## Citation

```bibtex
@techreport{diSanti2026Fleet,
  author      = {Di Santi, Eduardo},
  title       = {Fleet-Wide Condition Monitoring via Shared Nominal
                 Representations and Local Radius Calibration:
                 Enabling OEM-Centralized, Fleet-Deployed CBM},
  institution = {University of Colorado Boulder,
                 Department of Applied Mathematics},
  type        = {Working paper},
  year        = {2026}
}
```

## Related work by the author

- *Anomaly Detection as an Inverse Problem on Quotient Spaces.* Technical
  Report, CU Boulder, 2026. <https://scholar.colorado.edu/concern/reports/7h149s02s>
  — develops the quotient-space formalization this applied paper does not assume.
- *Self-Supervision, Manifold Saturation, and Emergent Operational Classes.*
  Technical Report, CU Boulder, 2026. <https://scholar.colorado.edu/concern/reports/fn107102t>
- *The Blueprints of Intelligence.* arXiv:2512.05089.

## License

Code in `code/` and `verification/` is released under the MIT License; see
`LICENSE`. The manuscript text and figures are governed by the terms selected in
the CU Scholar deposit record.
