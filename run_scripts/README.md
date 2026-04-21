# Run Scripts Organization

All training and experiment scripts organized by purpose.

## Directory Structure

### 📊 `baselines/`
Core training scripts for baseline comparisons at 30M scale.

**SLURM:**
- `test_30M_baselines.sh` - Full suite: FP32, FP16, STE, CAGE, ECO, ECO0

**Local:**
- `run_baseline_local.sh` - Local version of 30M baselines

**Single training scripts:**
- `train.sh` - Generic training template
- `train_eco.sh` - ECO training
- `train_eco0m-rooh.sh` - ECO0 training (our method)
- `train_all.sh` - Multiple methods in sequence

### 📈 `scaling/`
Model size scaling experiments (50M, 100M, 300M, 1B).

**50M/** - 50M parameter experiments
- `test_scaling_50M.sh` - SLURM version (FP16, CAGE, ECO, ECO0)
- `run_scaling_50M_local.sh` - Local version

**100M/** - 100M parameter experiments
- `test_scaling_100M.sh` - SLURM (not yet run)

**300M/** - 300M parameter experiments
- `test_scaling_300M.sh` - SLURM (not yet run)

**1B/** - 1B parameter experiments
- `test_scaling_1B.sh` - SLURM (not yet run)

**Root:**
- `test_scaling_template.sh` - Template for creating new scales

### 🔬 `ablations/`
Ablation studies and hyperparameter tuning.

**percentile/** - FP4 percentile ablation (P85-P100)
- `test_percentile_sweep.sh` - Broad sweep (P85, P90, P95, P99, P100)
- `test_percentile_fine.sh` - Fine-grained sweep
- `test_percentile_30M.sh` - 30M validation (P90, P95, P99)
- `test_percentile_30M_a100.sh` - A100 partition version

**lr_tuning/** - Learning rate tuning
- `run_eco0_lr_ablation_50M.sh` - ECO0 LR sweep at 50M (0.006, 0.0065, 0.007)

**Root:**
- `test_fp4_scale_fix.sh` - FP4 scale recalibration study

### 🚀 `slurm_utils/`
SLURM submission and management utilities.

- `submit_fp4_test_smart.sh` - Smart multi-partition submission (H200 → A100 → gpu-all)
- `submit_multipartition.sh` - Generic multi-partition array job splitter

### 🛠️ `utils/`
General utilities for environment setup and job management.

- `setup_env.sh` - Environment setup
- `rerun_missing.sh` - Rerun failed/incomplete experiments
- `resume_eco.sh` - Resume ECO training from checkpoint

### 🗄️ `deprecated/`
Old experiments and legacy scripts (preserved for reference).

- `run_bash_*.sh` - Old bash-based runs
- `run_slurm_*.sh` - Legacy SLURM scripts
- `train_momentum*.sh` - Deprecated momentum experiments
- `train_ecohm.sh`, `train_eco0m.sh` - Old optimizer variants
- `train_step2.sh`, `run_step2.sh` - Multi-step experiments
- `train_no_activations.sh` - Activation quantization experiments

## Quick Start

### Run 30M Baselines
```bash
# SLURM (recommended)
sbatch run_scripts/baselines/test_30M_baselines.sh

# Local (single method)
./run_scripts/baselines/run_baseline_local.sh 4  # ECO0
```

### Run 50M Scaling
```bash
# SLURM
sbatch run_scripts/scaling/50M/test_scaling_50M.sh

# Local (parallel on 2 machines)
CUDA_VISIBLE_DEVICES=0,1 ./run_scripts/scaling/50M/run_scaling_50M_local.sh 3 &  # ECO
CUDA_VISIBLE_DEVICES=2,3 ./run_scripts/scaling/50M/run_scaling_50M_local.sh 4 &  # ECO0
```

### Run ECO0 LR Ablation
```bash
./run_scripts/ablations/lr_tuning/run_eco0_lr_ablation_50M.sh 1  # LR=0.006
./run_scripts/ablations/lr_tuning/run_eco0_lr_ablation_50M.sh 2  # LR=0.0065
./run_scripts/ablations/lr_tuning/run_eco0_lr_ablation_50M.sh 3  # LR=0.007
```

### Smart SLURM Submission
```bash
# Tries H200 first, falls back to A100, then gpu-all
./run_scripts/slurm_utils/submit_fp4_test_smart.sh
```

## Key Findings

### ✅ Completed Experiments (April 2026)

**30M Baselines:**
- FP16: 27.49 | CAGE: 29.08 | ECO: 32.73 | **ECO0: 30.06**
- **Result:** ECO0 beats ECO by 2.67 points

**50M Scaling:**
- FP16: 3.074 | CAGE: 3.142 | ECO: 3.189 | ECO0 (LR=0.00775): 3.196
- **Problem:** ECO0 fell behind ECO in final steps (late-stage collapse)
- **Action:** Testing lower LRs (0.006, 0.0065, 0.007)

**Percentile Ablation:**
- **P90 is optimal** for FP4 quantization (1.8% better than P99)
- Validated on both tiny and 30M models

**Memory Profiling (50M):**
- ECO0 saves 1 GB vs FP16 (28.28 vs 29.29 GB)
- Activations dominate at small scale (86%)
- Savings scale linearly with params (1 GB @ 50M → 10 GB @ 1B)

## WandB Projects

- `ECO0` - Main experiments
- `ECO0-SCALING` - 50M, 100M, 300M, 1B scaling
- `FP4-PERCENTILE-SWEEP` - Percentile ablations
- `FP4-PERCENTILE-30M` - 30M percentile validation

## Notes

- All scripts use P90 percentile for FP4 quantization (unless testing percentile itself)
- Default batch: 64 × 8 acc_steps = 512 effective
- Default sequence length: 512 tokens
- LR scaling: 1/sqrt(param_ratio) for model scaling
- Use `conda activate cage` before running
