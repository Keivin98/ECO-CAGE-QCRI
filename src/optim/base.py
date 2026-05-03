from contextlib import nullcontext
import copy
from pathlib import Path
import time
import yaml

import numpy as np
import torch
import wandb
from tqdm import tqdm

from logger.logger import DynamicsLogger
from optim.buckets import BUCKETS, build_param_to_bucket
from optim.weight_averaging import (
    WeightAverager,
    eval_ema,
    eval_wa,
    ExponentialWeightAverager,
)
from .utils import (
    eval,
    get_batch,
    load_checkpoint,
    load_worker_state,
    save_checkpoint,
    save_worker_state,
)


# ----------------------------------------------------------------------
# Helpers for per-bucket dynamics logging (paper Section: Mechanism plots)
# ----------------------------------------------------------------------


@torch.no_grad()
def _bucket_aggregate(values_per_param, bucket_map, weighted=False):
    """values_per_param: dict id(p) -> scalar. Returns dict bucket -> aggregate."""
    sums = {b: 0.0 for b in BUCKETS}
    counts = {b: 0 for b in BUCKETS}
    weights = {b: 0.0 for b in BUCKETS}
    for pid, v in values_per_param.items():
        if v is None:
            continue
        b = bucket_map.get(pid, 'other')
        sums[b] += v
        counts[b] += 1
        weights[b] += 1.0
    return {b: (sums[b] / counts[b]) if counts[b] else None for b in BUCKETS}


@torch.no_grad()
def _snapshot_param_norms(model):
    """id(p) -> ||p||."""
    return {id(p): p.detach().float().norm().item() for _, p in model.named_parameters()}


@torch.no_grad()
def _snapshot_grad_norms(model):
    """id(p) -> ||p.grad|| or None if no grad."""
    out = {}
    for _, p in model.named_parameters():
        if p.grad is None:
            out[id(p)] = None
        else:
            out[id(p)] = p.grad.detach().float().norm().item()
    return out


@torch.no_grad()
def _snapshot_param_clones(model):
    """id(p) -> param tensor clone (for |Δθ| diff after opt.step)."""
    return {id(p): p.detach().clone() for _, p in model.named_parameters()}


@torch.no_grad()
def _delta_norms(model, param_clones):
    """id(p) -> ||p_new - p_old||. param_clones is the dict from _snapshot_param_clones."""
    out = {}
    for _, p in model.named_parameters():
        old = param_clones.get(id(p), None)
        if old is None:
            out[id(p)] = None
        else:
            out[id(p)] = (p.detach().float() - old.float()).norm().item()
    return out


@torch.no_grad()
def _per_param_ratio(num_dict, den_dict, eps=1e-12):
    """Element-wise ratio over matching ids. Returns id(p) -> ratio or None."""
    out = {}
    for pid, n in num_dict.items():
        d = den_dict.get(pid, None)
        if n is None or d is None:
            out[pid] = None
        else:
            out[pid] = n / (d + eps)
    return out


@torch.no_grad()
def _log_weight_histograms(model, bucket_map, tag, log_dict, n_bins=100, save_dir=None):
    """Build a matplotlib histogram per bucket and log it as a wandb.Image
    (static panel with proper value-on-x / count-on-y axes).

    For QuantizedLinear params, weights are multiplied by the quantizer's scale,
    so quantized bins land at integers 0, ±1, ..., ±levels and we draw vertical
    reference lines at each integer to make the lattice obvious.

    If save_dir is provided, also save a PNG per bucket (handy for the paper).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collect (rescaled) weights, separately per bucket; remember whether the
    # bucket was quantized so we can draw the integer grid.
    bucket_vals = {b: [] for b in BUCKETS}
    bucket_levels = {b: None for b in BUCKETS}   # max integer level (for grid lines)
    bucket_was_quantized = {b: False for b in BUCKETS}

    for name, p in model.named_parameters():
        b = bucket_map.get(id(p), 'other')
        scale = None
        levels = None
        try:
            mod = model
            for part in name.split('.')[:-1]:
                mod = getattr(mod, part)
            q = getattr(mod, 'weight_quantizer', None)
            if q is not None and hasattr(q, 'scale') and q.scale is not None:
                s = q.scale
                scale = float(s.detach().float().mean().item()) if hasattr(s, 'detach') else float(s)
                # Pull the integer grid range if the quantizer exposes it.
                if hasattr(q, 'levels'):
                    try:
                        levels = float(q.levels)
                    except Exception:
                        levels = None
        except Exception:
            scale = None

        w = p.detach().float().flatten().cpu()
        if scale is not None and scale > 0:
            w = w * scale
            bucket_was_quantized[b] = True
            if levels is not None:
                bucket_levels[b] = max(bucket_levels[b] or 0, levels)
        bucket_vals[b].append(w)

    # Build one figure per non-empty bucket.
    for b in BUCKETS:
        if not bucket_vals[b]:
            continue
        arr = torch.cat(bucket_vals[b]).numpy()
        # Cap range at quantization grid + small margin for clarity; otherwise
        # use empirical 0.1/99.9 percentiles to avoid being dominated by outliers.
        if bucket_was_quantized[b] and bucket_levels[b]:
            L = bucket_levels[b]
            xmin, xmax = -L - 1.0, L + 1.0
            data_for_hist = np.clip(arr, xmin, xmax)
        else:
            xmin = float(np.percentile(arr, 0.1))
            xmax = float(np.percentile(arr, 99.9))
            # add a small symmetric margin
            margin = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
            xmin, xmax = xmin - margin, xmax + margin
            data_for_hist = arr

        fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=120)
        ax.hist(data_for_hist, bins=n_bins, range=(xmin, xmax),
                color='steelblue', edgecolor='none', alpha=0.85)
        if bucket_was_quantized[b] and bucket_levels[b]:
            L = int(bucket_levels[b])
            for k in range(-L, L + 1):
                ax.axvline(k, color='crimson', linewidth=0.4, alpha=0.5)
            ax.set_xlabel(f"weight × scale (INT4 grid: integers in [-{L}, {L}])")
        else:
            ax.set_xlabel("weight value")
        ax.set_ylabel("count")
        n_outliers_below = int((arr < xmin).sum())
        n_outliers_above = int((arr > xmax).sum())
        clipped = n_outliers_below + n_outliers_above
        title = f"{b}  ·  {tag}  ·  N={arr.size:,}"
        if clipped > 0:
            title += f"  ·  clipped {clipped:,}"
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', linewidth=0.3, alpha=0.5)
        fig.tight_layout()

        # Log as a static image to wandb.
        log_dict[f"hist/{b}/{tag}"] = wandb.Image(fig)

        # Optionally save a PNG to disk for paper figures.
        if save_dir is not None:
            try:
                from pathlib import Path
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_dir / f"hist_{b.replace('.', '_')}_{tag}.png",
                            bbox_inches='tight')
            except Exception:
                pass

        plt.close(fig)


def _measure_grad_noise_scale(model, train_reader, type_ctx, distributed_backend, cfg):
    """One-shot McCandlish-style gradient noise scale: do `acc_steps` separate
    microsteps, capture per-microstep gradients, compute the scale.

    NOTE: This does cfg.acc_steps fresh forward+backward passes that advance
    the train_reader; after returning, the next training iter starts a few
    batches later. For a one-shot measurement that's an acceptable cost.
    """
    # Snapshot current grad state (post-zero_grad it should already be None or zero,
    # but be defensive).
    saved_grads = {}
    for _, p in model.named_parameters():
        saved_grads[id(p)] = None if p.grad is None else p.grad.detach().clone()
        if p.grad is not None:
            p.grad.zero_()

    micro_norms_sq = []  # per-microstep ||g_i||^2 (scalar)
    g_sum = None         # running sum of g_i (vector, on GPU)
    with torch.enable_grad():
        for _ in range(cfg.acc_steps):
            x, y = get_batch(train_reader, device=cfg.device)
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(
                    model=model, microstep_idx=0, gradient_accumulation_steps=1
                ):
                    out = model(x, targets=y)
            out["loss"].backward()
            # Stream-pack the just-accumulated grads so we never hold
            # acc_steps * P floats at once.
            sq = 0.0
            flat_chunks = []
            for _, p in model.named_parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach().float()
                sq += float((g * g).sum().item())
                flat_chunks.append(g.flatten())
                p.grad.zero_()
            micro_norms_sq.append(sq)
            if flat_chunks:
                packed = torch.cat(flat_chunks)
                if g_sum is None:
                    g_sum = packed.clone()
                else:
                    g_sum.add_(packed)

    # Restore the original grads.
    for _, p in model.named_parameters():
        sg = saved_grads.get(id(p), None)
        if sg is None:
            p.grad = None
        else:
            if p.grad is None:
                p.grad = sg.clone()
            else:
                p.grad.copy_(sg)

    if not micro_norms_sq or g_sum is None:
        return {}
    n = len(micro_norms_sq)
    g_mean = g_sum / n
    mean_grad_sq = float((g_mean * g_mean).sum().item())  # |E[g]|^2
    mean_sq = sum(micro_norms_sq) / n                      # E[|g|^2]
    noise_scale = (mean_sq - mean_grad_sq) / max(mean_grad_sq, 1e-12)
    return {
        'setup/grad_noise_scale': noise_scale,
        'setup/grad_norm_E[g]': mean_grad_sq ** 0.5,
        'setup/grad_norm_mean_abs_g': mean_sq ** 0.5,
    }



def train(
    model,
    opt,
    datareaders,
    scheduler,
    exp_dir,
    distributed_backend,
    cfg,
    cage=None,
):
    not_compiled_model = model
    if cfg.compile:
        print(f"Compiling model ...")
        model = torch.compile(model)

    if "cuda" in cfg.device:
        type_ctx = torch.amp.autocast(
            device_type="cuda",
            dtype={
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[cfg.dtype],
        )
    else:
        type_ctx = nullcontext()

    if cfg.resume_from:
        # This is a full resume including the model weights, optimizer, state
        # dataloader state, random seed, etc. Not indended for fine tuning or
        # other scenarios where some of these should change.
        print(f"\nResuming Training From {cfg.resume_from}")
        ckpt_dir = Path(cfg.resume_from)
        curr_iter = load_checkpoint(
            model,
            opt,
            scheduler,
            ckpt_dir / "main.pt",
            cfg.device,
        )
        load_worker_state(ckpt_dir)
    else:
        curr_iter = 0

    if cfg.weight_average:
        # This does generally not support resuming training, but will work if
        # cfg.wa_interval perfectly divides the iteration number of the chkpt.
        # Otherwise, the first avg will not be correctly computed, with a bias
        # towards the first sample and missing values for earlier iterations.
        weight_averager = WeightAverager(
            not_compiled_model,
            horizon=cfg.wa_horizon,
            interval=cfg.wa_interval,
            save_dir=None if cfg.wa_use_temp_dir else exp_dir / "avgs",
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
            count=curr_iter,
        )

    if cfg.exponential_moving_average:
        ema = ExponentialWeightAverager(
            not_compiled_model,
            interval=cfg.ema_interval,
            decay=cfg.ema_decay,
            warmup=cfg.warmup_steps if cfg.ema_after_warmup else 0,
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
        )

    if distributed_backend.is_master_process() and cfg.log_dynamics:
        with open(cfg.dynamics_logger_cfg, "r") as f:
            dlcfg = yaml.safe_load(f)

        # Hooks into optimizer
        dlogger = DynamicsLogger(
            model, opt, dlcfg, cfg.results_base_folder, wandb=cfg.wandb
        )
        dlogger.iteration = curr_iter

    substep = curr_iter * cfg.acc_steps
    train_reader, val_reader = datareaders["train"], datareaders["val"]
    train_reader.set_step(substep)
    stats = {"train_loss": [], "val_loss": [], "val_pp": [], "val_acc": []}
    model.train()

    # Per-bucket dynamics logging setup. Build id(p) -> bucket once.
    raw_model = distributed_backend.get_raw_model(model) if hasattr(distributed_backend, "get_raw_model") else model
    bucket_map = build_param_to_bucket(raw_model)
    grad_noise_scale_logged = False

    # One-shot weight histogram at iter 0 (master only).
    hist_save_dir = exp_dir / "figures" / "histograms" if exp_dir is not None else None
    if (
        cfg.wandb
        and distributed_backend.is_master_process()
        and curr_iter == 0
    ):
        hist_log = {"iter": curr_iter}
        _log_weight_histograms(
            raw_model, bucket_map, "weights_iter_0", hist_log, save_dir=hist_save_dir
        )
        wandb.log(hist_log)

    # Initialize the progress bar
    if distributed_backend.is_master_process():
        pbar = tqdm(total=cfg.iterations, desc="Training Progress", position=curr_iter)
    else:
        pbar = None

    while curr_iter <= cfg.iterations:
        # Save permanent checkpoint
        if cfg.permanent_ckpt_interval > 0:
            if curr_iter % cfg.permanent_ckpt_interval == 0:
                ckpt_dir = exp_dir / "ckpts" / str(curr_iter)
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
                save_worker_state(ckpt_dir)

        # Save temporary checkpoint for resuming training
        if cfg.latest_ckpt_interval > 0:
            if curr_iter % cfg.latest_ckpt_interval == 0 or curr_iter == cfg.iterations:
                ckpt_dir = exp_dir / "ckpts" / "latest"
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
                save_worker_state(ckpt_dir)

        ws = distributed_backend.get_world_size()
        tokens = ws * substep * cfg.sequence_length * cfg.batch_size
        epoch = tokens / train_reader.num_tokens
        if (
            curr_iter % cfg.eval_interval == 0
            or curr_iter == cfg.iterations
            or (curr_iter in cfg.full_eval_at)
        ):
            eval_and_log(
                curr_iter,
                epoch,
                model,
                val_reader,
                type_ctx,
                distributed_backend,
                cfg,
                opt,
                full_eval=(curr_iter in cfg.full_eval_at),
            )

            if curr_iter > cfg.wa_interval and cfg.weight_average:
                eval_wa(
                    curr_iter,
                    not_compiled_model,
                    weight_averager,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )
            if cfg.exponential_moving_average:
                eval_ema(
                    curr_iter,
                    not_compiled_model,
                    ema,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )

        if curr_iter == cfg.iterations:
            # Save checkpoints and evaluate at final iteration, but no need to train further
            # Final weight histograms (master only).
            if cfg.wandb and distributed_backend.is_master_process():
                hist_log = {"iter": curr_iter}
                _log_weight_histograms(
                    raw_model, bucket_map, "weights_iter_final", hist_log,
                    save_dir=hist_save_dir,
                )
                wandb.log(hist_log)
            break

        # NOTE: The McCandlish-style gradient noise scale measurement was
        # removed because it ran fwd+bwd on master rank only, which triggers
        # DDP all-reduce collectives that the non-master rank never matches.
        # That permanently desyncs the NCCL collective sequence and triggers
        # a 10-minute watchdog timeout much later in training (causing the
        # 2026-04-28 batch of failures near iter 11440 of 11444).
        # If you want this metric back, run it symmetrically on all ranks
        # (using no_sync context) and average per-rank estimates.
        _ = grad_noise_scale_logged  # silence unused-var linter

        # Train model
        t_start = time.perf_counter_ns()
        for microstep_idx in range(cfg.acc_steps):  # gradient accumulation
            x, y = get_batch(train_reader, device=cfg.device)
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(
                    model=model,
                    microstep_idx=microstep_idx,
                    gradient_accumulation_steps=cfg.acc_steps,
                ):
                    outputs = model(x, targets=y)

            loss = outputs["loss"] / cfg.acc_steps
            loss.backward()
            substep += 1

        if cfg.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if cfg.opt == "SFAdamW":
            opt.train()

        # Decide whether to capture per-bucket dynamics this step.
        log_dyn_this_step = (
            cfg.wandb
            and cfg.log_interval
            and ((curr_iter + 1) % cfg.log_interval == 0)
            and distributed_backend.is_master_process()
        )
        # Snapshots taken pre-step so we can compute |Δθ| and per-bucket grad norms.
        pre_param_clones = _snapshot_param_clones(raw_model) if log_dyn_this_step else None
        pre_grad_norms = _snapshot_grad_norms(raw_model) if log_dyn_this_step else None
        # Tell optimizer to populate dynamics_stats if applicable.
        if hasattr(opt, "log_dynamics"):
            opt.log_dynamics = log_dyn_this_step

        opt.step()
        scheduler.step()
        # Optional CAGE post-step correction using current LR after scheduler update
        if cage is not None:
            lam = cage.step()

        # Compute |Δθ| using the pre-step clones. Must happen before zero_grad
        # so we can also log per-bucket grad norms (already snapshotted).
        if log_dyn_this_step:
            delta_norms_dict = _delta_norms(raw_model, pre_param_clones)
            post_param_norms = _snapshot_param_norms(raw_model)
            eff_step = _per_param_ratio(delta_norms_dict, post_param_norms)
            opt_dyn_stats = getattr(opt, "dynamics_stats", {}) or {}
            # Stash for the wandb log block below.
            _pending_dyn = {
                "delta_norms": delta_norms_dict,
                "post_param_norms": post_param_norms,
                "pre_grad_norms": pre_grad_norms,
                "eff_step": eff_step,
                "opt_dyn": opt_dyn_stats,
            }
        else:
            _pending_dyn = None

        #check for side effects
        opt.zero_grad(set_to_none=True)
        # Free the pre-step clones promptly.
        pre_param_clones = None
        pre_grad_norms = None
            
        if cfg.weight_average:
            weight_averager.step(
                not_compiled_model, distributed_backend.is_master_process()
            )
        if cfg.exponential_moving_average:
            ema.step(not_compiled_model, distributed_backend.is_master_process())
        dt = (time.perf_counter_ns() - t_start) / 1e9

        curr_iter += 1
        if distributed_backend.is_master_process():
            pbar.update(1)

        if (
            cfg.log_interval
            and curr_iter % cfg.log_interval == 0
            and distributed_backend.is_master_process()  # Only log on master rank
        ):
            train_loss = loss.detach().cpu().item() * cfg.acc_steps
            current_lrs = [param_group["lr"] for param_group in opt.param_groups]

            # Track peak GPU memory usage
            peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9

            print(
                f"Train: Iter={curr_iter} ({epoch:0.3f} epochs) "
                f"train_loss={train_loss:.3f} iter_dt={dt:.2e}s "
                f"lr={current_lrs[0]:.2e} peak_mem={peak_memory_gb:.2f}GB"
            )

            if cfg.wandb:
                cage_stats = {}
                if cage is not None:
                    cage_stats_dict = cage.get_stats() if hasattr(cage, "get_stats") else None
                    if cage_stats_dict is not None:
                        cage_stats = {f"cage/{k}": v for k, v in cage_stats_dict.items()}
                
                # Prepare wandb log dict
                log_dict = {
                    "iter": curr_iter,
                    "train/loss": train_loss,
                    "train/perplexity": 2.71828**train_loss,
                    "iter_dt": dt,
                    "memory/peak_gb": peak_memory_gb,
                    **cage_stats
                }
                
                # Log learning rates for all parameter groups   
                for i, lr in enumerate(current_lrs):
                    if i == 0:
                        log_dict["lr"] = lr  # Keep backward compatibility
                    log_dict[f"lr/group_{i}"] = lr
                
                step_stats = getattr(opt, "step_stats", None)
                if step_stats:
                    # keys you care about
                    keys = [
                        "update_meanabs_pre_carry",
                        "update_meanabs_post_carry",
                        "uq_meanabs",
                        "uq_frac_zero",
                        "remainder_meanabs",
                        "carry_meanabs",
                        "carry_frac_zero",
                        "fire_frac",
                        "u_mean",
                        "u_max",
                        "m_meanabs",
                        "scale",
                        "sat_frac",
                    ]

                    for k in keys:
                        vals = [d[k] for d in step_stats if k in d]
                        if not vals:
                            continue

                        log_dict[f"train/{k}/mean"] = sum(vals) / len(vals)
                        log_dict[f"train/{k}/max"]  = max(vals)
                        log_dict[f"train/{k}/min"]  = min(vals)

                    log_dict["train/quant_stats/num_tensors"] = len(step_stats)

                # ------ Per-bucket dynamics logging ------
                if _pending_dyn is not None:
                    delta = _pending_dyn["delta_norms"]
                    pnorms = _pending_dyn["post_param_norms"]
                    gnorms = _pending_dyn["pre_grad_norms"]
                    eff = _pending_dyn["eff_step"]
                    opt_dyn = _pending_dyn["opt_dyn"]

                    for b in BUCKETS:
                        # Per-bucket norms (sum of squares -> sqrt for proper L2 norm).
                        theta_sq = 0.0; grad_sq = 0.0; delta_sq = 0.0; n_in = 0
                        for pid, bb in bucket_map.items():
                            if bb != b:
                                continue
                            n_in += 1
                            v = pnorms.get(pid)
                            if v is not None:
                                theta_sq += v * v
                            v = gnorms.get(pid)
                            if v is not None:
                                grad_sq += v * v
                            v = delta.get(pid)
                            if v is not None:
                                delta_sq += v * v
                        if n_in == 0:
                            continue
                        theta_n = theta_sq ** 0.5
                        grad_n = grad_sq ** 0.5
                        delta_n = delta_sq ** 0.5
                        log_dict[f"dyn/{b}/theta_norm"] = theta_n
                        log_dict[f"dyn/{b}/grad_norm"] = grad_n
                        log_dict[f"dyn/{b}/delta_norm"] = delta_n
                        if theta_n > 0:
                            log_dict[f"dyn/{b}/eff_step_size"] = delta_n / theta_n

                    if opt_dyn:
                        agg_keys = [
                            # variance / norms (all stages)
                            "v_rel_err",
                            "theta_norm",
                            "theta_meanabs",
                            "grad_norm",
                            "update_norm",
                            "eff_step",

                            # quantization residual (stages 2-4)
                            "e_norm",
                            "e_meanabs",
                            "e_over_theta",

                            # stage 2 (residual buffer)
                            "theta_hat_norm",
                            "theta_master_norm",
                            "residual_norm",
                            "residual_meanabs",
                            "residual_over_theta",

                            # stages 3-4 (ECO error feedback)
                            "eco_error_norm",
                            "eco_error_over_theta",
                            "correction_norm",
                            "update_ef_frac",
                            "cos_u_e",

                            # legacy keys (older Adam0 versions)
                            "cos_g_e",
                            "m_norm",
                            "v_norm",
                            "theta_tilde_norm",
                            "eco_correction_norm",
                        ]

                        for k in agg_keys:
                            per_bucket = {b: [] for b in BUCKETS}

                            for pid, stats_dict in opt_dyn.items():
                                v = stats_dict.get(k)
                                if v is None:
                                    continue

                                bb = bucket_map.get(pid, "other")
                                per_bucket[bb].append(v)

                            for b, vals in per_bucket.items():
                                if not vals:
                                    continue

                                log_dict[f"dyn/{b}/{k}/mean"] = sum(vals) / len(vals)
                                log_dict[f"dyn/{b}/{k}/min"] = min(vals)
                                log_dict[f"dyn/{b}/{k}/max"] = max(vals)

                wandb.log(log_dict)
                # log exp_avg_sq means
                # exp_avg_sq_means = []
                # exp_avg_means = []
                # max_smp_numel = 0
                # smpl_exp_avg_sq_value = None
                # smpl_exp_avg_value = None
                # for param_group in opt.param_groups:
                #     for p in param_group["params"]:
                #         exp_avg_sq_means.append(opt.state[p]['exp_avg_sq'].mean().item())
                #         exp_avg_means.append(opt.state[p]['exp_avg'].mean().item())
                #         if opt.state[p]['exp_avg_sq'].numel() > max_smp_numel:
                #             max_smp_numel = opt.state[p]['exp_avg_sq'].numel()
                #             smpl_exp_avg_sq_value = opt.state[p]['exp_avg_sq']
                #             smpl_exp_avg_value = opt.state[p]['exp_avg']
                
                # wandb.log({"exp_avg_sq/mean_hist": wandb.Histogram(exp_avg_sq_means)})
                # wandb.log({"exp_avg_sq/mean": sum(exp_avg_sq_means) / len(exp_avg_sq_means)})
                # wandb.log({"exp_avg_sq/smpl_mean": smpl_exp_avg_sq_value.mean().item()})
                # perc_999 = smpl_exp_avg_sq_value.abs().flatten().kthvalue(int(0.999 * smpl_exp_avg_sq_value.numel()))[0].item()
                # wandb.log({"exp_avg_sq/smpl_abs_999_percentile": perc_999})
                # wandb.log({"exp_avg_sq/smpl_abs_min": smpl_exp_avg_sq_value.abs().min().item()})
                # wandb.log({"exp_avg_sq/smpl_abs_max": smpl_exp_avg_sq_value.abs().max().item()})
                
                # wandb.log({"exp_avg/mean_hist": wandb.Histogram(exp_avg_means)})
                # wandb.log({"exp_avg/mean": sum(exp_avg_means) / len(exp_avg_means)})
                # wandb.log({"exp_avg/smpl_mean": smpl_exp_avg_value.mean().item()})
                # perc_999 = smpl_exp_avg_value.abs().flatten().kthvalue(int(0.999 * smpl_exp_avg_value.numel()))[0].item()
                # wandb.log({"exp_avg/smpl_abs_999_percentile": perc_999})
                # wandb.log({"exp_avg/smpl_abs_min": smpl_exp_avg_value.abs().min().item()})
                # wandb.log({"exp_avg/smpl_abs_max": smpl_exp_avg_value.abs().max().item()})
    return stats


def eval_and_log(
    curr_iter,
    epoch,
    model,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
    opt,
    full_eval=False,
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return

    model.eval()
    if cfg.opt == "SFAdamW":
        opt.eval()

    if curr_iter == cfg.iterations or full_eval:
        max_num_batches = val_reader.num_batches()
    else:
        max_num_batches = cfg.eval_batches

    # to make sure we start from the beginning of the validation set,
    # i.e. repeat the same batches
    val_reader.set_step(0)
    val_acc, val_loss, val_perplexity = eval(
        model,
        val_reader,
        cfg.device,
        max_num_batches=max_num_batches,
        ctx=type_ctx,
        cfg=cfg,
    )

    print(
        f">Eval: Iter={curr_iter} ({epoch:0.3f} epochs) "
        f"val_loss={val_loss:.3f} "
        f"val_pp={val_perplexity:.3f} "
        f"val_acc={val_acc:3f}"
    )

    if cfg.wandb:
        if curr_iter == cfg.iterations or full_eval:
            logs = {
                "iter": curr_iter,
                "final-val/loss": val_loss,
                "final-val/perplexity": val_perplexity,
                "final-val/acc": val_acc,
            }
        else:
            logs = {
                "iter": curr_iter,
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
                "val/acc": val_acc,
            }

        wandb.log(logs)
        if cfg.eval_seq_prefix != "none" and (
            curr_iter % (cfg.eval_interval * 5) == 0 or curr_iter == cfg.iterations
        ):
            text_table = wandb.Table(columns=["itr", "val-pp", "text"])

            out_str = distributed_backend.get_raw_model(model).generate_from_string(
                cfg.eval_seq_prefix,
                max_new_tokens=40,
                temperature=0.9,
                top_k=None,
            )
            text_table.add_data(curr_iter, val_perplexity, out_str)
            # why a copy? see github.com/wandb/wandb/issues/2981
            wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})
    model.train()
