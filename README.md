# CAGE 

[![arXiv](https://img.shields.io/badge/arXiv-2510.18784-b31b1b.svg)](https://arxiv.org/abs/2510.18784)

**CAGE: Curvature-Aware Gradient Estimation For Accurate Quantization-Aware Training**

Official implementation of CAGE (Curvature-Aware Gradient Estimation), a new QAT method that augments the straight-through estimator (STE) gradient with a curvature-aware correction designed to counteract the loss increase induced by quantization.

## Abstract

Despite significant work on low-bit quantization-aware training (QAT), there is still a large accuracy gap between such techniques and native training. To address this, we introduce CAGE (Curvature-Aware Gradient Estimation), a new QAT method that augments the straight-through estimator (STE) gradient with a curvature-aware correction designed to counteract the loss increase induced by quantization. CAGE is derived from a multi-objective view of QAT that balances loss minimization with adherence to quantization constraints, yielding a principled correction term that depends on local curvature information. On the theoretical side, we introduce the notion of Pareto-optimal solutions for quantized optimization, and establish that CAGE yields strong convergence guarantees in the smooth non-convex setting. In terms of implementation, our approach is optimizer-agnostic, but we provide a highly-efficient implementation that leverages Adam statistics. When pre-training Llama-style models of up to 800M-parameters, CAGE recovers over 10% of the quantization-induced loss increase in the W4A4 regime over outlier-mitigation methods. These results indicate that curvature-aware gradient corrections can bridge the remaining performance gap beyond current outlier-handling methods.

## Quickstart

The entry point to the codebase is `train.sh`. 

Create a virtual environment and install dependencies (we recommend Python 3.11):

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

**Note:** The code has been tested with CUDA 12.6 and PyTorch 2.6.

Run a simple training on the C4 dataset on 30M llama model:
```bash
bash train.sh
```

## Configuration

You can customize training by passing arguments to `train.sh` using the format `--key=value`. Here are the main options:

### Model Sizes

Available model sizes: `30M`, `50M`, `100M`, `200M`, `430M`, `800M`, `1700M`, `3200M`

```bash
# Train a 100M parameter model
bash train.sh --model-size-prefix=100M

# Train an 800M parameter model
bash train.sh --model-size-prefix=800M
```

### Quantization Bits

Control the number of bits for weight and activation quantization:

```bash
# 4-bit weights and 4-bit activations (default)
bash train.sh --w-bits=4 --a-bits=4

# 8-bit weights and 4-bit activations
bash train.sh --w-bits=8 --a-bits=4

# 2-bit weights and 2-bit activations
bash train.sh --w-bits=2 --a-bits=2
```

### CAGE Arguments

Configure CAGE behavior with the following options:

- `--use-cage`: Enable/disable CAGE (default: `True`)
- `--cage-lambda`: Base lambda value for the correction term (default: `10`)
- `--cage-silence-ratio`: Fraction of training steps where CAGE is inactive (default: `0.8`)
- `--cage-schedule`: Schedule type - `linear_ramp` or `constant` (default: `linear_ramp`)
- `--cage-track-stats`: Track statistics for logging (default: `True`)

```bash
# Disable CAGE
bash train.sh --use-cage=False

# Use a higher lambda value
bash train.sh --cage-lambda=20

# Use constant schedule instead of linear ramp
bash train.sh --cage-schedule=constant

# Adjust silence ratio (lower means CAGE is active more often)
bash train.sh --cage-silence-ratio=0.5

# Combine multiple CAGE options
bash train.sh --cage-lambda=15 --cage-silence-ratio=0.7 --cage-schedule=constant
```

### Combined Example

```bash
# Train a 200M model with 4-bit quantization and custom CAGE settings
bash train.sh --model-size-prefix=200M --w-bits=4 --a-bits=4 --cage-lambda=15 --cage-silence-ratio=0.7
```

## Citation

If you use CAGE in your research, please cite:

```bibtex
@misc{tabesh2025cagecurvatureawaregradientestimation,
      title={CAGE: Curvature-Aware Gradient Estimation For Accurate Quantization-Aware Training}, 
      author={Soroush Tabesh and Mher Safaryan and Dan Alistarh},
      year={2025},
      eprint={2510.18784},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.18784}, 
}
```
