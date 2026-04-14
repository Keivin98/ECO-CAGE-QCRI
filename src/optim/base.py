from contextlib import nullcontext
import copy
from pathlib import Path
import time
import yaml

import torch
import wandb
from tqdm import tqdm

from logger.logger import DynamicsLogger
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
            break

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
        opt.step()
        scheduler.step()
        # Optional CAGE post-step correction using current LR after scheduler update
        if cage is not None:
            lam = cage.step()
        
        #check for side effects
        opt.zero_grad(set_to_none=True)
            
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

            print(
                f"Train: Iter={curr_iter} ({epoch:0.3f} epochs) "
                f"train_loss={train_loss:.3f} iter_dt={dt:.2e}s "
                f"lr={current_lrs[0]:.2e}"
            )

            if cfg.wandb:
                cage_stats = {}
                if cage is not None:
                    stats = cage.get_stats() if hasattr(cage, "get_stats") else None
                    cage_stats = {f"cage/{k}": v for k, v in stats.items()}
                
                # Prepare wandb log dict
                log_dict = {
                    "iter": curr_iter,
                    "train/loss": train_loss,
                    "train/perplexity": 2.71828**train_loss,
                    "iter_dt": dt,
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
