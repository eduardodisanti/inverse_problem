# Reproducibility Statement

**Fleet-Wide Condition Monitoring via Shared Nominal Representations and Local
Radius Calibration — Enabling OEM-Centralized, Fleet-Deployed CBM**

Eduardo Di Santi, Department of Applied Mathematics, University of Colorado
Boulder.

## Status

This is a **working paper / preprint**, not a peer-reviewed publication.

A companion technical report develops the quotient-space formalization that this
paper deliberately does not assume: *Anomaly Detection as an Inverse Problem on
Quotient Spaces*, deposited separately at CU Boulder
(<https://scholar.colorado.edu/concern/reports/7h149s02s>) and cited in the
manuscript. Numerical results there and here may differ where calibration has
since been refined; the specific declared sources of difference are listed under
Known Limitations below.

---

## 0. Read this first — Keras 2 is required

The autoencoder applies an L1 activity regularizer to its latent layer. **Under
Keras 3 that penalty drives the latent to zero.** The model degenerates to
predicting the per-feature mean, validation loss plateaus near 0.95 instead of
0.066, and no fault is detected.

The failure is **silent**. A collapsed autoencoder still reports a ~2% healthy
false-alarm rate, because a constant predictor trivially satisfies its own
percentile. Only the detection rates give it away, by collapsing to the
false-alarm rate. Measured in a Keras 3 environment:

| | Keras 2 | Keras 3 |
|---|---|---|
| best val_loss | 0.0636 | 0.9490 |
| latent activation sd | normal | 1.4 × 10⁻⁴ |
| threshold τ | 0.115474 | 2.872169 |
| healthy FAR | 2.0% | 2.0% |
| outer-race detection | 100% | 1.3% |
| inner-race detection | 100% | 2.7% |
| ball-fault detection | 91.3% | 2.7% |

TensorFlow 2.16 and later default to Keras 3, so on any current TensorFlow:

```bash
pip install -r requirements.txt      # includes tf-keras
export TF_USE_LEGACY_KERAS=1
python verification/check_environment.py    # confirms Keras 2 is active
```

Three safeguards are now in place:

- `verification/check_environment.py` fails if Keras 3 is active.
- `verification/verify_pipeline.py` sets `TF_USE_LEGACY_KERAS=1` itself, and
  aborts with a diagnosis if the representation collapses anyway.
- Each notebook opens with an **environment guard cell** that sets the variable
  and raises if Keras 2 is not active. Run it first, before any other cell.

This was found while preparing the deposit and is the single most likely reason
a third party would fail to reproduce the results.

---

## 1. What can be verified without any external data

Three scripts run against the physics-motivated simulator and synthetic
residual streams. They require no download, no dataset licence and no GPU.

| Script | Needs | Runtime | Covers |
|---|---|---|---|
| `verification/check_environment.py` | none | < 5 s | dependency versions, dataset resolution |
| `verification/verify_simulator.py` | NumPy | ~5 s | simulator shape, determinism, nuisance model, fault signatures (Section 8.3) |
| `verification/verify_commissioning.py` | NumPy | ~10 s | operational-radius estimator and convergence stopping rule |
| `verification/verify_pipeline.py` | TensorFlow | several minutes | shared representation, threshold transfer, local commissioning, baseline harness |

```bash
pip install -r requirements.txt
python verification/check_environment.py
python verification/verify_simulator.py
python verification/verify_commissioning.py
python verification/verify_pipeline.py
```

Each exits 0 on success and non-zero on failure, and prints every quantity it
checked rather than only a verdict.

### The regression test that matters most

`bearing_autoencoder.ipynb` carries, as stored executed output, the
simulated-domain result of the shared nominal representation:

```
Threshold tau (p98 healthy MSE) = 0.115022
Detection rate (MSE > tau):
  Healthy     :   2.0%
  Outer Race  : 100.0%
  Inner Race  : 100.0%
  Ball Fault  :  92.0%
```

`verify_pipeline.py` stage A rebuilds that experiment from
`bearing_simulator.py` — same architecture, same seeds, same training schedule —
and checks the outcome against those recorded values. Re-running it on a
different operating system, a different CPU vendor, no GPU, and a TensorFlow
five minor versions newer (with Keras 2 via `tf-keras`) reproduced:

```
best val_loss = 0.063561     (reference run: 0.066)
tau           = 0.115474     (0.4% from the recorded 0.115022)
  healthy     :   2.0%   (recorded   2.0%)
  outer_race  : 100.0%   (recorded 100.0%)
  inner_race  : 100.0%   (recorded 100.0%)
  ball_fault  :  91.3%   (recorded  92.0%)
```

Three of four detection rates match exactly, the fourth to within one test
window; the threshold differs by 0.4%, consistent with BLAS and backend
nondeterminism.

Stages B and C then reproduce the manuscript's two central claims on simulated
field data:

```
B  absolute threshold transferred to a rescaled field asset
     healthy false-alarm rate = 100.0%   (against a 2% target)
     -- matching the paper's finding that transferring the absolute
        simulator threshold flagged every healthy CWRU window as anomalous

C  local operational radius commissioned on the same asset
     converged after 67 windows
     agreement with the complete-data field oracle = 0.53%
     healthy false-alarm rate 100.0% -> 2.0%
     -- the paper reports convergence after 55 windows on average and
        0.83% agreement with the oracle on CWRU
```

---

## 2. What requires external data

The CWRU and NASA IMS results in the manuscript cannot be reproduced from this
deposit alone. Neither dataset is redistributed here; both are third-party and
remain governed by their providers' terms. See `data/README.md` for download and
placement, then:

```bash
export CWRU_DATA_ROOT=/path/to/CWRU_Bearing_NumPy-main/Data
export NASA_IMS_DATA_ROOT=/path/to/NASA_Bearing/IMS
```

or place the data at `data/CWRU/` and `data/NASA_IMS/`. `code/data_paths.py`
resolves both, in that order, falling back to the sibling layout used during the
original runs. No path is hard-coded in any notebook.

Notebook order: `bearing_autoencoder.ipynb` first — it produces the shared
representation (`shared_ae`, `shared_scaler`, `shared_sig_len`, `shared_tau_mc`)
that `bearing_autoencoder_cwru_per_asset.ipynb` and
`bearing_autoencoder_NASA_per_asset.ipynb` consume. Both raise an explicit
`RuntimeError` listing the missing objects if run out of order.

Once the shared representation exists, calibration and evaluation operate on
exported one-dimensional residual streams without retraining, so the raw data is
needed only once, at the residual-extraction step.

---

## 3. Environments

**Original** — the environment that produced the reported figures. Recorded in
the notebooks' own stored output.

```
Python       3.11.1
tensorflow   2.16.2   with tensorflow-metal, Apple M2 Pro (16 GB)
```

The numpy, pandas, matplotlib and scikit-learn versions of that run were not
recorded. This is a gap: only the TensorFlow version and the kernel's Python
version were captured.

**Verified** — an independent environment in which this package was re-run end
to end while preparing the deposit.

```
Python       3.10.12   Linux x86-64, CPU only
numpy        2.2.6
pandas       2.3.3
matplotlib   3.10.9
scikit-learn 1.7.2
tensorflow   2.21.0
tf-keras     2.21.0    with TF_USE_LEGACY_KERAS=1
```

No code change was needed to move across five TensorFlow minor versions, two
NumPy majors and from Apple Metal to CPU — **provided Keras 2 is active**. See
Section 0. `requirements.txt` specifies floors admitting both environments and
pulls in `tf-keras`.

---

## 4. Determinism

Seeding is explicit throughout: `set_all_seeds(seed)` in
`generic_cwru_asset_specific_experiments.py` seeds `random`, `numpy` and
`tensorflow` together; the notebooks seed `numpy` and `tensorflow` at the top;
commissioning-stability experiments use the fixed seed set
`[7, 21, 42, 84, 126]`.

Two caveats, stated rather than silently patched:

1. **`bearing_simulator.py` has no internal seed.** It draws from the global
   NumPy random state, so it is deterministic only once the caller has seeded
   that state. Every notebook and script in this package does so.
   `verify_simulator.py` check 2 confirms bit-identical replay under a fixed
   seed.

2. **`BearingSignalSimulator.canonical()` is not a noiseless template.** It calls
   `sample(mc=False)`, which disables the Monte Carlo nuisance draws but still
   adds measurement noise of standard deviation `params.noise_std`. Two
   successive calls therefore differ. The method is referenced only in a
   markdown cell and is never called in executed code, so no reported result
   depends on it. Behaviour was left unchanged so that previously produced
   figures remain reproducible.

GPU-backend floating-point nondeterminism means bit-level agreement across
platforms is not expected; see the tolerances in `verify_pipeline.py`.

---

## 5. Known limitations — declared, not silently fixed

**1. The code produces a degenerate model under Keras 3.** Fully described in
Section 0. Guard cells now prevent running under Keras 3, but the notebooks
still contain no assertion that *training itself* succeeded, so any other cause
of collapse would look like "the method did not work" rather than "training
failed". Recommended additional fix, not yet applied: assert
`min(history.history['val_loss']) < 0.5` after every `fit`, as
`verify_pipeline.py` does.

**2. Commissioning length is below the floor implied by a 98th percentile.**
`MIN_CALIBRATION_SIZE = 40` in both per-asset notebooks. An empirical 98th
percentile needs at least `ceil(1/0.02) = 50` observations before the estimate
stops being pinned to the sample maximum. Below that floor the radius is biased
low, which inflates the false-alarm rate relative to the 2% target.
`verify_commissioning.py` check 8 quantifies this on lognormal residuals:

| n | relative bias of the p98 estimate |
|---|---|
| 20 | −14.8% |
| 40 | −9.8% |
| 50 | −9.5% |
| 99 | −5.6% |
| 200 | −2.7% |
| 400 | −1.7% |

The manuscript reports 0.83% agreement with the field oracle on CWRU because
CWRU reconstruction residuals are far lighter-tailed than this stress case; the
*sign* of the bias is the general property, its magnitude is tail-dependent.
Re-running with `MIN_CALIBRATION_SIZE >= 99` is planned and has not been applied
to the notebooks.

**3. The convergence rule detects a stable estimate, not a stationary process.**
A monotonically drifting nominal stream still satisfies the stopping criterion,
because the relative change of a cumulative percentile over a 10-step look-back
falls below 1% once enough history accumulates. `verify_commissioning.py`
check 5 demonstrates this and reports the shortfall: under a 6× drift the frozen
radius sits 37.5% below the drifted level. Operationally this is the intended
behaviour for run-to-failure monitoring — it is precisely why commissioning must
occur on verified early-life nominal data. Degradation that begins *during*
commissioning would be absorbed into the radius. The manuscript states the
related caveat directly: the criterion "should not be interpreted as proof that
the population 98th percentile has been recovered exactly."

**4. Two NASA IMS protocol conventions exist in the same notebook.**
`bearing_autoencoder_NASA_per_asset.ipynb` contains both:

| | cell 7 (earlier, exploratory) | cells 19/22/25 (reported path) |
|---|---|---|
| recording aggregation | 95th percentile of window residuals | **mean** of window residuals |
| persistence rule | 3 of 5 | **8 of 10** |

The results reported in the manuscript come from the second path: cell 25 sets
`PERSISTENCE_WINDOW = 10, PERSISTENCE_REQUIRED = 8`, matching the manuscript's
"10-recording persistence window", and cell 19 aggregates by mean. The cell-7
constants belong to a superseded earlier pipeline and are not used for the
reported figures. **Confirm which path a cell belongs to before comparing
against reported numbers.** Removing the dead constants is planned.

**5. The compiled PDF predates two source additions.** `manuscript_draft.tex`
now contains a *Data and Code Availability* section, an *Acknowledgment of
Prior Dissemination* note, and a `\hypersetup` block carrying the PDF document
properties. `manuscript_draft.pdf` in this package has **not** been rebuilt
from that source: it is the 30 July 2026 build, with its metadata patched in
place (see Section 6). The two new sections therefore appear in the `.tex` but
not in the PDF.

Rebuild before submitting anywhere:

```bash
cd manuscript
pdflatex manuscript_draft && bibtex manuscript_draft
pdflatex manuscript_draft && pdflatex manuscript_draft
```

The rebuild also resolves what was previously listed here as a separate
limitation: the NASA IMS dataset entry `ims2007` existed in `references.bib`
but was never cited. The new availability section cites both `smith2015` and
`ims2007`, so both dataset providers are now attributed. Expect the page count
to grow by roughly one page.

Requires the `newtx` package (Times-like text and math), which the TRB class
loads. If a build environment lacks it, the manuscript will not compile.

None of the above reflects a data error.

---

## 6. Changes made while preparing this deposit

Recorded for transparency. No scientific behaviour was altered.

- **Data paths.** `../CWRU_Bearing_NumPy-main/Data` and `../NASA_Bearing/IMS`
  were hard-coded in four notebooks, so nothing ran outside the author's
  directory layout. Replaced by `code/data_paths.py`, which resolves the
  environment variable first, then `data/`, then the original sibling paths — so
  an existing working tree keeps functioning unchanged.
- **Loader extraction.** `generic_cwru_asset_specific_experiments.py` called
  `make_windows`, `load_1797_normal` and siblings that were only ever defined in
  notebook cells, so the module could not be imported standalone. Those
  definitions were reproduced verbatim in `code/cwru_loader.py`, which the module
  now imports, with a fallback that preserves the paste-into-notebook workflow.
- **Estimator de-duplication.** `AdaptiveOperationalRadius`,
  `operational_radius_convergence_trace` and `find_convergence_index` were
  defined twice, once in each per-asset notebook, and had already begun to drift
  (a docstring and a blank line differed). Both copies were replaced by an import
  from `code/operational_radius.py`, which holds the definitions verbatim.
- **Stray import removed.** `from unittest import signals` in
  `bearing_autoencoder_NASA_per_asset.ipynb` cell 4 was an accidental
  autocomplete artifact, shadowed immediately by a local assignment.
- **Keras guard cells.** Each notebook gained a first cell that sets
  `TF_USE_LEGACY_KERAS=1` before TensorFlow is imported and raises if Keras 2 is
  not active. Without it, current TensorFlow installs silently train a
  degenerate model; see Section 0.
- **Duplicate `figures/` removed.** Its three PNGs were byte-identical copies of
  files already in `manuscript/`.
- **Filename typos corrected.** `healthy_sim_reconstuction.png` →
  `healthy_sim_reconstruction.png` (not referenced by the `.tex`, so the build
  is unaffected) and `generic_cwru_asset_specific_expedriments.ipynb` →
  `..._experiments.ipynb`.
- **`LICENSE` scope clarified.** The MIT terms cover `code/` and
  `verification/`; the manuscript is governed by the CU Scholar deposit record.
  This resolves the previous contradiction with the old README, which claimed
  "In Copyright — Educational Use Permitted".
- **`data/README.md` corrected.** It instructed the reader to place raw `.mat`
  files under `data/CWRU/raw/`, but the loaders read the `.npz` conversion from
  `<root>/<RPM> RPM/`. The documented layout now matches the code.
- **PDF document properties set.** The deposited PDF had empty Title, Author,
  Subject and Keywords, and no XMP metadata stream — which is what repositories
  and scholarly indexers read. These were written directly into the existing
  file with `pikepdf`, without re-rendering: page count, page size and extracted
  text are unchanged (63,563 characters, identical before and after). The same
  values were added to `manuscript_draft.tex` as a `\hypersetup` block so any
  future rebuild carries them.
- **Manuscript source additions.** A *Data and Code Availability* section and a
  short *Acknowledgment of Prior Dissemination* note were added to
  `manuscript_draft.tex`, before the bibliography. The availability statement is
  written self-referentially — it points at "the reproducibility package
  accompanying this deposit" and contains no URL — so it stays correct
  regardless of where the record eventually lives. See Known Limitation 5: the
  PDF has not been rebuilt from this source.
- **Housekeeping.** Removed two `.DS_Store` files and `__pycache__`; added
  `.gitignore` and `CU_Scholar_deposit_form.txt`.
- **Added:** `requirements.txt`, the four `verification/` scripts, and this
  statement. The previous `README.md` and `REPRODUCIBILITY.md` in this directory
  described a different paper (the MNIST technote and the quotient-space report
  respectively) and have been rewritten for this manuscript.

---

## 7. Citation

Cite the working paper and, where applicable, the companion technical report
deposited at CU Boulder. Full entries in `README.md` and `resources.txt`.
