# Learning from Scarce Observations as Inverse Recovery of Compact Functional Structures

**Self-Supervision, Manifold Saturation, and Emergent Operational Classes**

Eduardo Di Santi · College of Engineering and Applied Science · University of Colorado Boulder

[![arXiv](https://img.shields.io/badge/arXiv-2512.05089-b31b1b.svg)](https://arxiv.org/abs/2512.05089)
[![License](https://img.shields.io/badge/License-In%20Copyright%20Educational%20Use%20Permitted-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)]()

---

## Overview

This repository contains the technical note and accompanying code for:

> *Learning from Scarce Observations as Inverse Recovery of Compact Functional Structures*
> Eduardo Di Santi, 2026.

The central claim is that self-supervised learning and few-shot anomaly detection can be understood as instances of a broader **inverse problem**: the recovery of stable functional structure from partial, scarce, noisy, or indirect observations.

Rather than assuming a predefined class taxonomy, the system observes signals from an unknown physical or functional process and must recover the latent structure that organizes the data. Classes are not given — they **emerge** as stabilized compact manifolds in a suitable representation space. Expert feedback then assigns physical or operational meaning to these manifolds.

This work extends and provides theoretical grounding for [The Blueprints of Intelligence (arXiv:2512.05089v7)](https://arxiv.org/abs/2512.05089).

---

## Core Thesis

> Learning from scarce observations is not primarily a supervised classification problem.
> It is an **inverse recovery problem** over compact functional structures.

The forward process generates observations from a latent regime:

```
θ → x   (physical process)
```

Learning requires the inverse direction:

```
x → I(θ)   (recovery of functional invariants)
```

This inverse problem is generally **ill-posed** (Hadamard, 1932; Tikhonov & Arsenin, 1977), and requires regularization. Self-supervised learning is interpreted here as a mechanism for constructing representation spaces in which the inverse problem becomes geometrically regular.

---

## Key Concepts

### Compact Functional Manifolds

Physically coherent regimes concentrate around compact regions in representation space:

```
φ(X) ≈ ⋃ Mⱼ
```

Each `Mⱼ` is a compact manifold corresponding to a coherent operational regime. A new observation is assigned to a regime by **geometric membership**, not by a supervised decision boundary. If no manifold is sufficiently close, the system abstains:

```
f(x) = unknown / anomaly / new regime
```

### Blueprint Saturation

A regime becomes **learnable** when its functional geometry stabilizes under additional observations:

```
H(n) → H*,   r_max(n) → r*,   V(n) → V*
```

This provides an operational criterion for diagnosing when a compact functional region has been sufficiently recovered from scarce data.

### Emergent Classification via Expert Feedback

Instead of labeling individual samples, the domain expert labels stabilized manifolds:

```
Mⱼ ↦ yⱼ
```

For example: `M₁ ↦ normal`, `M₂ ↦ friction`, `M₃ ↦ heat`, `M₄ ↦ harmless variation`, `M₅ ↦ emerging degradation`.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── paper/
│   └── Learning_from_Scarcity_as_an_Inverse_Problem.pdf
└── experiments/
    └── mnist_manifold_recovery.py       # MNIST proof-of-concept experiment
```

---

## Experiment: MNIST Operational Manifold Coverage

The MNIST experiment illustrates the proposed principle in a controlled setting. MNIST is **not** used as a supervised classification benchmark. Instead, labels are withheld during geometric analysis and introduced only afterwards as expert feedback, playing the role of `Mⱼ ↦ yⱼ`.

The experiment evaluates three quantities as observations accumulate:

1. **Geometric separability** — fraction of observations closer to their own expert-identified manifold than to any competing manifold
2. **Membership margin** `γ(z) = d_out(z) − d_in(z)`
3. **Empirical manifold radius** — mean p95 class radius as a robust estimate of manifold extent

| Observations n | Geometric separability | Median margin | Mean p95 class radius |
|---------------|----------------------|--------------|----------------------|
| 500           | 0.81                 | 1.18         | 8.20                 |
| 1000          | 0.87                 | 1.65         | 8.34                 |
| 5000          | 1.00                 | 6.56         | 8.37                 |

The p95 radius stabilizes early (~n=500–1000), while separability and margin continue to improve as the already-recovered manifolds are densified — consistent with the blueprint saturation criterion.

### Running the experiment

```bash
pip install -r requirements.txt
python experiments/mnist_manifold_recovery.py
```

---

## Requirements

```
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Related Work

This technical note connects to the following prior and concurrent work:

- **The Blueprints of Intelligence** (Di Santi, 2026) — empirical validation of compact perceptual manifolds across five real-world domains: [arXiv:2512.05089](https://arxiv.org/abs/2512.05089)
- **Learning from Examples as an Inverse Problem** (De Vito et al., JMLR 2005) — formal connection between learning theory and ill-posed inverse problems
- **Learning, Regularization and Ill-Posed Inverse Problems** (Rosasco et al., COLT 2004)
- **Solutions of Ill-Posed Problems** (Tikhonov & Arsenin, 1977)
- **Self-Organized Formation of Topologically Correct Feature Maps** (Kohonen, 1982) — historical precedent for unsupervised geometric regularization
- **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture** (Assran et al., CVPR 2023) — modern neural counterpart

---

## Citation

If you use this work, please cite:

```bibtex
@techreport{diSanti2026SelfSupervision,
  author      = {Di Santi, Eduardo},
  title       = {Self-Supervision, Manifold Saturation, and Emergent Operational Classes:
                 Learning from Scarce Observations as Inverse Recovery of Compact Functional Structure},
  institution = {University of Colorado Boulder, College of Engineering and Applied Science},
  type        = {Working paper},
  year        = {2026},
  url         = {https://scholar.colorado.edu/concern/reports/fn107102t}
}
```

For the companion paper:

```bibtex
@misc{diSanti2025Blueprints,
  author  = {Di Santi, Eduardo},
  title   = {The Blueprints of Intelligence: A Functional-Topological 
             Foundation for Perception and Representation},
  year    = {2026},
  note    = {arXiv:2512.05089v7}
}
```

---

## License

In Copyright — Educational Use Permitted.
© 2026 Eduardo Di Santi. This work may be used freely for educational and research purposes.
