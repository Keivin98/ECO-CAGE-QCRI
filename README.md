# Quantized Optimization

This repository provides an implementation of a **low-bit quantization-aware training (QAT) framework** for transformer-based language models, with a focus on efficient optimization under strict precision constraints.

The codebase explores training dynamics in quantized regimes, combining:

- low-bit weight and activation quantization  
- optimizer-level adaptations  
- experimentation

The goal is to **reduce the performance gap between quantized and full-precision training while maintaining scalability**.

---

## Installation

We recommend **Python 3.11**.

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
````

**Tested with:**

* PyTorch 2.6
* CUDA 12.6

---

## Quickstart

The main entry point is:

```bash
bash train.sh
```

This runs a small-scale training job on C4 with a default configuration.

---

## Configuration

All arguments can be passed as:

```bash
bash train.sh --key=value
```

---

### Model Size

```
--model-size-prefix=30M | 50M | 100M | 200M | 430M | 800M | 1700M | 3200M
```

Example:

```bash
bash train.sh --model-size-prefix=200M
```

---

### Quantization

Control precision of weights and activations:

```bash
--w-bits=4
--a-bits=4
```

Examples:

```bash
# Standard low-bit setup
bash train.sh --w-bits=4 --a-bits=4

# More aggressive compression
bash train.sh --w-bits=2 --a-bits=2

# Mixed precision
bash train.sh --w-bits=8 --a-bits=4
```

---

## Code Structure

```
.
├── train.sh                # Entry point
├── models/                 # Model definitions (LLaMA-style)
├── quantization/           # Quantization modules
├── optim/                  # Optimizers and variants
├── data/                   # Dataset handling (C4)
├── utils/                  # Logging, helpers
```

---

## Acknowledgment

This implementation builds upon ideas introduced in:

**CAGE: Curvature-Aware Gradient Estimation for Accurate Quantization-Aware Training**
Soroush Tabesh, Mher Safaryan, Dan Alistarh (2025)

If you use curvature-aware corrections, please cite their work:

```bibtex
@misc{tabesh2025cagecurvatureawaregradientestimation,
  title={CAGE: Curvature-Aware Gradient Estimation For Accurate Quantization-Aware Training},
  author={Soroush Tabesh and Mher Safaryan and Dan Alistarh},
  year={2025},
  eprint={2510.18784},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

---

## Notes

* Default configurations are tuned for **stability over raw speed**
* Extreme low-bit settings (e.g., 2-bit) may require hyperparameter tuning
* Multi-GPU training is recommended for larger models (≥800M)

```
