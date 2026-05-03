'''
    This module contains the final version of the vanilla "Stateless Adam" optimizer. 
    While the performance of this optimizer matches that of AdamW, the best learning rate doesn't match to that of AdamW.
    This compatibility will be addressed in another variant of this optimizer.
'''

import copy
import torch
from optim.qcri.grouping import group_parameters_llm_2d1d_only

class _Adam0Staged(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        beta1=0.9,
        eps=1e-8,
        weight_decay=0.0,
        n_attention_heads=None,
        is_vector=False,
        stage=3,
        quantize_on_init=True,
        enter_once=True,
    ):
        if stage not in (1, 2, 3, 4):
            raise ValueError(f"stage must be 1, 2, 3, or 4, got {stage}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not (0.0 < beta1 < 1.0):
            raise ValueError(f"beta1 must be in (0, 1), got {beta1}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if n_attention_heads is None:
            raise ValueError("n_attention_heads must be provided")

        self.stage = int(stage)
        self.quantize_on_init = bool(quantize_on_init)
        self.enter_once = enter_once

        defaults = dict(
            lr=lr,
            beta1=beta1,
            eps=eps,
            weight_decay=weight_decay,
            n_attention_heads=n_attention_heads,
            is_vector=is_vector,
            enter_once=enter_once,
        )

        super().__init__(params, defaults)

        self.log_dynamics = False
        self.dynamics_stats = {}

    @torch.no_grad()
    def zero_grad(self):
        """
        This optimizer intentionally does not zero gradients.

        Instead, p.grad is used as the first-moment/momentum buffer and is
        decayed by beta1 before the next backward pass accumulates the new grad.
        """
        self.decay_grad()

    @torch.no_grad()
    def decay_grad(self):
        for group in self.param_groups:
            beta1 = group["beta1"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("This optimizer does not support sparse gradients")
                p.grad.data.mul_(beta1)

    @staticmethod
    def _has_quantizer(p):
        return hasattr(p, "quantizer") and p.quantizer is not None

    @torch.no_grad()
    def _maybe_calibrate(self, p, x):
        if not self._has_quantizer(p):
            return

        q = p.quantizer
        did_calibrate = getattr(q, "did_calibrate", True)

        if isinstance(did_calibrate, torch.Tensor):
            did_calibrate = bool(did_calibrate.item())
        else:
            did_calibrate = bool(did_calibrate)

        if hasattr(q, "calibrate_scale_") and not did_calibrate:
            q.calibrate_scale_(x)

    @torch.no_grad()
    def _quantize(self, p, x):
        if not self._has_quantizer(p):
            return x
        self._maybe_calibrate(p, x)
        return p.quantizer.hard_quantize(x)

    @torch.no_grad()
    def _quantize_param_in_place_and_get_residual(self, p):
        theta_fp = p.detach()
        theta_hat = self._quantize(p, theta_fp)
        residual = theta_fp - theta_hat
        p.copy_(theta_hat)
        return residual.detach().clone()

    @torch.no_grad()
    def compute_attention_variance_vectorized_efficient(self, m, beta, n_attention_heads):
        d_out, d_in = m.shape
        head_dim = d_out // n_attention_heads

        m_heads = m.view(n_attention_heads, head_dim, d_in)
        m_sq = m_heads * m_heads

        row = m_sq.sum(dim=2, keepdim=True)
        col = m_sq.sum(dim=1, keepdim=True)
        total = torch.sum(row, dim=1, keepdim=True).clamp_min(1e-30)

        v = row * ((1.0 + beta) / (1.0 - beta) / total) * col
        return v.view(d_out, d_in)

    @torch.no_grad()
    def _make_update_and_denom(self, p, grad_buf, lr, beta1, eps, step, n_attention_heads, is_vector):
        """
        Returns:
            update: tensor with same shape as p. This is what should be subtracted from theta.
            denom_param: denominator tensor with same shape as p, used for ECO inverse step.
            v_rel_err: optional logging metric.
        """
        bias_c1 = 1.0 - beta1 ** step

        if grad_buf.ndim > 1 or is_vector:
            if is_vector:
                m_2d = grad_buf[None, :] / bias_c1
            else:
                m_2d = grad_buf / bias_c1

            variance_map = self.compute_attention_variance_vectorized_efficient(
                m_2d, beta1, n_attention_heads
            )
            denom_2d = variance_map.sqrt().add(eps)

            if is_vector:
                m_param = m_2d[0]
                denom_param = denom_2d[0]
            else:
                m_param = m_2d
                denom_param = denom_2d

            update = lr * m_param / denom_param

            v_rel_err = None
            if self.log_dynamics:
                m_sq = m_2d * m_2d
                num = (variance_map - m_sq).norm()
                den = m_sq.norm().clamp_min(1e-12)
                v_rel_err = (num / den).item()

            return update, denom_param, v_rel_err

        else:
            m_col = grad_buf[:, None] / bias_c1

            variance_map = self.compute_attention_variance_vectorized_efficient(
                m_col, beta1, n_attention_heads
            )
            denom_col = variance_map.sqrt().add(eps)

            m_param = m_col[:, 0]
            denom_param = denom_col[:, 0]

            update = lr * m_param / denom_param

            v_rel_err = None
            if self.log_dynamics:
                m_sq = m_col * m_col
                num = (variance_map - m_sq).norm()
                den = m_sq.norm().clamp_min(1e-12)
                v_rel_err = (num / den).item()

            return update, denom_param, v_rel_err


    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.log_dynamics:
            self.dynamics_stats = {}

        for group in self.param_groups:
            beta1 = group["beta1"]
            weight_decay = group["weight_decay"]
            lr = group["lr"]
            eps = group["eps"]
            n_attention_heads = group["n_attention_heads"]
            is_vector = group["is_vector"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad_buf = p.grad.data

                if grad_buf.is_sparse:
                    raise RuntimeError("This optimizer does not support sparse gradients")

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0

                    if self.stage in (2, 3, 4) and self.quantize_on_init:
                        residual0 = self._quantize_param_in_place_and_get_residual(p)

                        if self.stage == 2:
                            state["residual"] = residual0

                    elif self.stage == 2:
                        state["residual"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                state["step"] += 1
                step = state["step"]

                bias_c1 = 1.0 - beta1 ** step

                update, denom, v_rel_err = self._make_update_and_denom(
                    p=p,
                    grad_buf=grad_buf,
                    lr=lr,
                    beta1=beta1,
                    eps=eps,
                    step=step,
                    n_attention_heads=n_attention_heads,
                    is_vector=is_vector,
                )

                if self.stage == 1:
                    if weight_decay != 0.0:
                        p.mul_(1.0 - lr * weight_decay)

                    p.sub_(update)

                    if self.log_dynamics:
                        theta_norm = p.detach().float().norm().clamp_min(1e-12)
                        grad_norm = grad_buf.detach().float().norm()
                        update_norm = update.detach().float().norm()

                        self.dynamics_stats[id(p)] = {
                            "numel": p.numel(),
                            "stage": 1,
                            "step": step,
                            "v_rel_err": v_rel_err,
                            "theta_norm": theta_norm.item(),
                            "theta_meanabs": p.detach().float().abs().mean().item(),
                            "grad_norm": grad_norm.item(),
                            "update_norm": update_norm.item(),
                            "eff_step": (update_norm / theta_norm).item(),
                            "e_norm": None,
                            "e_over_theta": None,
                            "residual_norm": None,
                            "correction_norm": None,
                            "update_ef_frac": None,
                        }

                    continue

                if self.stage == 2:
                    residual = state["residual"]

                    theta_master = p.detach() + residual

                    if weight_decay != 0.0:
                        theta_master = theta_master.mul(1.0 - lr * weight_decay)

                    theta_tilde = theta_master - update
                    theta_hat = self._quantize(p, theta_tilde)
                    residual_next = theta_tilde - theta_hat

                    if not torch.isfinite(residual_next).all():
                        raise FloatingPointError(
                            f"Non-finite residual in stage=2 at step {step}: "
                            f"||residual||={residual_next.detach().float().norm().item()}"
                        )

                    p.copy_(theta_hat)
                    state["residual"] = residual_next.detach().clone()

                    if self.log_dynamics:
                        theta_hat_f = theta_hat.detach().float()
                        residual_f = residual_next.detach().float()
                        grad_f = grad_buf.detach().float()
                        update_f = update.detach().float()

                        theta_norm = theta_hat_f.norm().clamp_min(1e-12)
                        residual_norm = residual_f.norm()
                        update_norm = update_f.norm()
                        grad_norm = grad_f.norm()

                        self.dynamics_stats[id(p)] = {
                            "numel": p.numel(),
                            "stage": 2,
                            "step": step,
                            "v_rel_err": v_rel_err,
                            "theta_norm": theta_norm.item(),
                            "theta_hat_norm": theta_norm.item(),
                            "theta_master_norm": theta_master.detach().float().norm().item(),
                            "theta_meanabs": theta_hat_f.abs().mean().item(),
                            "grad_norm": grad_norm.item(),
                            "update_norm": update_norm.item(),
                            "eff_step": (update_norm / theta_norm).item(),
                            "e_norm": residual_norm.item(),
                            "e_over_theta": (residual_norm / theta_norm).item(),
                            "residual_norm": residual_norm.item(),
                            "residual_meanabs": residual_f.abs().mean().item(),
                            "correction_norm": None,
                            "update_ef_frac": None,
                        }

                    continue

                theta_start = p.detach()

                if weight_decay != 0.0:
                    theta_start = theta_start.mul(1.0 - lr * weight_decay)

                theta_tilde = theta_start - update
                theta_hat = self._quantize(p, theta_tilde)
                e_next = theta_tilde - theta_hat

                if not torch.isfinite(e_next).all():
                    raise FloatingPointError(
                        f"Non-finite quantization error at step {step}: "
                        f"||e_next||={e_next.detach().float().norm().item()}"
                    )

                correction = None

                if self.stage == 3:
                    inv_eta_t = (bias_c1 / lr) * denom
                    correction = inv_eta_t * (1.0 - 1.0 / beta1) * e_next

                    if not torch.isfinite(correction).all():
                        raise FloatingPointError(
                            f"Non-finite ECO correction at step {step}: "
                            f"||correction||={correction.detach().float().norm().item()}"
                        )

                    grad_buf.add_(correction)

                p.copy_(theta_hat)

                if self.log_dynamics:
                    theta_hat_f = theta_hat.detach().float()
                    e_f = e_next.detach().float()
                    grad_f = grad_buf.detach().float()
                    update_f = update.detach().float()

                    theta_norm = theta_hat_f.norm().clamp_min(1e-12)
                    e_norm = e_f.norm()
                    grad_norm = grad_f.norm()
                    update_norm = update_f.norm()

                    if correction is not None:
                        correction_norm = correction.detach().float().norm()
                        update_ef_frac = correction_norm / grad_norm.clamp_min(1e-12)
                    else:
                        correction_norm = None
                        update_ef_frac = None
                    u_flat = update_f.flatten()
                    e_flat = e_f.flatten()
                    cos_u_e = (
                        (u_flat @ e_flat)
                        / (u_flat.norm() * e_flat.norm()).clamp_min(1e-12)
                    )

                    self.dynamics_stats[id(p)] = {
                        "numel": p.numel(),
                        "stage": self.stage,
                        "step": step,
                        "v_rel_err": v_rel_err,
                        "theta_norm": theta_norm.item(),
                        "theta_hat_norm": theta_norm.item(),
                        "theta_meanabs": theta_hat_f.abs().mean().item(),
                        "grad_norm": grad_norm.item(),
                        "update_norm": update_norm.item(),
                        "eff_step": (update_norm / theta_norm).item(),
                        "e_norm": e_norm.item(),
                        "e_meanabs": e_f.abs().mean().item(),
                        "e_over_theta": (e_norm / theta_norm).item(),
                        "eco_error_norm": e_norm.item(),
                        "eco_error_over_theta": (e_norm / theta_norm).item(),
                        "correction_norm": (
                            correction_norm.item() if correction_norm is not None else None
                        ),
                        "update_ef_frac": (
                            update_ef_frac.item() if update_ef_frac is not None else None
                        ),
                        "cos_u_e": cos_u_e.item(),
                    }

        return loss

class _Adam0(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, eps=1e-8, weight_decay=0, n_attention_heads=None, is_vector=False, enter_once=True):
        self.enter_once = enter_once
        defaults = dict(
            lr=lr, beta1=beta1, eps=eps, weight_decay=weight_decay,
            n_attention_heads=n_attention_heads, is_vector=is_vector, enter_once=enter_once
        )
        assert n_attention_heads != None
        super().__init__(params, defaults)
        # Set to True to populate self.dynamics_stats on the next step. Reset by caller.
        self.log_dynamics = False
        # id(p) -> dict of scalar metrics, populated only when log_dynamics is True.
        self.dynamics_stats = {}

    @torch.no_grad()
    def zero_grad(self):
        # decaying gradients instead of zeroing them
        self.decay_grad()

    @torch.no_grad()
    def decay_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("This optimizer does not support sparse gradients")
                grad.mul_(group['beta1'])  # decay the gradient by beta1
    
    @torch.no_grad()
    def compute_attention_variance_vectorized_efficient(self, m, beta, n_attention_heads):
        d_out, d_in = m.shape
        head_dim = d_out // n_attention_heads
        m_heads = m.view(n_attention_heads, head_dim, d_in)           # [H, Hd, Din]
        m_sq = m_heads * m_heads                            # [H, Hd, Din]
        row = m_sq.sum(dim=2, keepdim=True)                 # [H, Hd, 1]
        col = m_sq.sum(dim=1, keepdim=True)                 # [H, 1, Din]
        total = torch.sum(row, dim=1, keepdim=True)         # [H, 1, 1]
        v = row * ((1+beta)/(1-beta)/total) * col
        return v.view(d_out, d_in)

    @staticmethod
    def _has_quantizer(p: torch.Tensor) -> bool:
        return hasattr(p, "quantizer") and p.quantizer is not None

    @staticmethod
    def _hard_quantize_like_param(p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if hasattr(p, "quantizer") and p.quantizer is not None:
            return p.quantizer.hard_quantize(x)
        return x
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.log_dynamics:
            self.dynamics_stats = {}

        for group in self.param_groups:
            beta1 = group['beta1']
            weight_decay = group['weight_decay']
            lr = group['lr']
            eps = group['eps']
            n_attention_heads = group['n_attention_heads']
            is_vector = group['is_vector']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("This optimizer does not support sparse gradients")

                state = self.state[p]

                # Initialize state
                if not 'step' in state:
                    state['step'] = 0

                step = state['step'] = state['step'] + 1  # increment in-place

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                if grad.ndim > 1 or is_vector:           # if grad is assocaited with the linear layer's weights
                    self.weights_m = grad
                    denom, v_rel_err = self.update_weights(p, lr, beta1, eps, step, n_attention_heads, is_vector=is_vector)
                else:                       # if grad is associated with the linear layer's bias
                    denom, v_rel_err = self.update_biases(p, lr, beta1, eps, step, n_attention_heads, is_vector=is_vector)

                theta_tilde = p.clone()

                if hasattr(p, "quantizer") and p.quantizer is not None:
                    theta_hat = p.quantizer.hard_quantize(theta_tilde)
                else:
                    theta_hat = theta_tilde

                e_next = theta_tilde - theta_hat

                # ECO momentum correction
                bias_c1 = 1.0 - beta1 ** step
                coeff = (bias_c1 / lr) * (1.0 - 1.0 / beta1)


                # IT WAS: / (1 - beta1). BUT BE CAREFUL HERE, THIS OPTIMIZER DOESNT MULTIPLY M BY (1 - beta1)
                correction = coeff * denom * e_next
                correction_t = correction.squeeze(0) if correction.ndim > grad.ndim else correction

                if self.log_dynamics:
                    # Everything is already in scope; no extra buffers needed.
                    grad_used_norm = grad.norm()
                    e_norm = e_next.norm()
                    theta_norm = theta_hat.norm() + 1e-12
                    correction_norm = correction_t.norm()
                    # cos(g_t, e_t): use bias-corrected first-moment direction
                    # (= grad / bias_c1), which is what actually drives the update.
                    g_dir = grad / bias_c1
                    e_flat = e_next.flatten()
                    g_flat = g_dir.flatten()
                    cos_g_e = (g_flat @ e_flat) / (g_flat.norm() * e_flat.norm() + 1e-12)
                    self.dynamics_stats[id(p)] = {
                        'numel': p.numel(),
                        'v_rel_err': v_rel_err,
                        'e_norm': e_norm.item(),
                        'theta_norm': theta_norm.item(),
                        'e_over_theta': (e_norm / theta_norm).item(),
                        'cos_g_e': cos_g_e.item(),
                        'update_ef_frac': (correction_norm / (grad_used_norm + 1e-12)).item(),
                        'grad_norm': grad_used_norm.item(),
                        'correction_norm': correction_norm.item(),
                    }

                grad.add_(correction_t)

                p.copy_(theta_hat)

        return loss

    @torch.no_grad()
    def update_weights(self, p, lr, beta1, eps, step, n_attention_heads, is_vector):
        if is_vector:
            m = p.grad.data[None, :] / (1 - beta1 ** step)
        else:
            m = p.grad.data / (1 - beta1 ** step)
        # Reshape into [n_attention_heads, d1, head_dim] without copying
        variance_map = self.compute_attention_variance_vectorized_efficient(m, beta1, n_attention_heads)
        denom = variance_map.sqrt() + eps

        # Variance estimator quality: rel-err between rank-1 factorized v
        # and the true pointwise m^2. Both are already in scope; compute only
        # when caller requested dynamics logging this step.
        v_rel_err = None
        if self.log_dynamics:
            m_sq = m * m
            num = (variance_map - m_sq).norm()
            den = m_sq.norm() + 1e-12
            v_rel_err = (num / den).item()

        if is_vector:
            p.data.addcdiv_(m[0], denom[0], value=-lr)
        else:
            p.data.addcdiv_(m, denom, value=-lr)  # no need to divide by beta1_sq

        return denom, v_rel_err

    @torch.no_grad()
    def update_biases(self, p, lr, beta1, eps, step, n_attention_heads, is_vector):
        if is_vector:
            raise NotImplementedError("Layer norm biases are not implemented")
        m = p.grad.data[:, None] / (1 - beta1 ** step)
        variance_map = self.compute_attention_variance_vectorized_efficient(m, beta1, n_attention_heads)
        denom = variance_map.sqrt() + eps

        v_rel_err = None
        if self.log_dynamics:
            m_sq = m * m
            num = (variance_map - m_sq).norm()
            den = m_sq.norm() + 1e-12
            v_rel_err = (num / den).item()

        p.data.addcdiv_(m[:, 0], denom[:, 0], value=-lr)
        return denom, v_rel_err


class AdamW0(torch.optim.Optimizer):
    def __init__(
        self,
        model,
        lr,
        beta1=0.9,
        eps=1e-8,
        weight_decay=0.0,
        grouping_verbose=False,
        use_fallback_optim=False,
        fallback_optim_kwargs=None,
    ):
        self.optimizers = []
        self._group_metadata = []
        self._param_to_child = {}
        self._use_fallback_optim = bool(use_fallback_optim)
        # Per-step dynamics-logging hooks; base.py toggles log_dynamics each step.
        self.log_dynamics = False
        self.dynamics_stats = {}

        if fallback_optim_kwargs is None:
            fallback_optim_kwargs = {}
        if not isinstance(fallback_optim_kwargs, dict):
            raise TypeError("fallback_optim_kwargs must be a dict.")
        self._fallback_optim_kwargs = copy.deepcopy(fallback_optim_kwargs)
        if (not self._use_fallback_optim) and self._fallback_optim_kwargs:
            raise ValueError("fallback_optim_kwargs was provided but use_fallback_optim=False.")

        param_groups = group_parameters_llm_2d1d_only(
            model,
            verbose=grouping_verbose,
            allow_fallback_for_unsupported=self._use_fallback_optim,
        )

        parent_groups = []

        for group in param_groups:
            group_copy = dict(group)
            rule = group_copy.pop("rule").lower()
            params = list(group_copy.pop("params"))
            sync_lr_from_parent = True
            if rule == "fallback":
                sync_lr_from_parent = "lr" not in self._fallback_optim_kwargs

            parent_groups.append({"params": params})

            child = self._create_child_optimizer(
                rule=rule,
                params=params,
                lr=lr,
                beta1=beta1,
                eps=eps,
                weight_decay=weight_decay,
                extra_kwargs=group_copy,
            )

            child_index = len(self.optimizers)
            self.optimizers.append(child)

            metadata = {
                "rule": rule,
                "config": copy.deepcopy(group_copy),
                "child_index": child_index,
                "sync_lr_from_parent": sync_lr_from_parent,
            }

            if child.param_groups:
                child_group = {
                    key: copy.deepcopy(value)
                    for key, value in child.param_groups[0].items()
                    if key != "params"
                }
                if child_group:
                    metadata["child_param_group"] = child_group

            self._group_metadata.append(metadata)

            for param in params:
                self._param_to_child[param] = child_index

        defaults = {
            "lr": lr,
            "beta1": beta1,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(parent_groups, defaults)

        self._build_param_index()

    def _create_child_optimizer(self, rule, params, lr, beta1, eps, weight_decay, extra_kwargs):
        child_kwargs = dict(extra_kwargs)
        if rule == "adam0":
            child_kwargs["n_attention_heads"] = 1
            child_kwargs["lr"] = lr
            child_kwargs["beta1"] = beta1
            child_kwargs["eps"] = eps
            child_kwargs["weight_decay"] = weight_decay
            return _Adam0(params=params, **child_kwargs)
        elif rule == "adam0_att":
            # n_attention_heads already in extra_kwargs from group_parameters_for_llama
            child_kwargs["lr"] = lr
            child_kwargs["beta1"] = beta1
            child_kwargs["eps"] = eps
            child_kwargs["weight_decay"] = weight_decay
            return _Adam0(params=params, **child_kwargs)        # n_attention_heads is included in the child_kwargs
        elif rule == "fallback":
            return self._create_fallback_optimizer(
                params=params,
                lr=lr,
                beta1=beta1,
                eps=eps,
                weight_decay=weight_decay,
            )
        raise ValueError(f"Unknown rule: {rule}")

    def _create_fallback_optimizer(self, params, lr, beta1, eps, weight_decay):
        if not self._use_fallback_optim:
            raise ValueError("Received fallback group but use_fallback_optim=False.")
        kwargs = copy.deepcopy(self._fallback_optim_kwargs)
        if "lr" not in kwargs:
            kwargs["lr"] = lr
        if "betas" not in kwargs:
            kwargs["betas"] = (beta1, 0.99)
        if "eps" not in kwargs:
            kwargs["eps"] = eps
        if "weight_decay" not in kwargs:
            kwargs["weight_decay"] = weight_decay        
        return torch.optim.AdamW(params=params, **kwargs)

    def _build_param_index(self):
        self._param_list = []
        self._param_index = {}

        for group in self.param_groups:
            for param in group["params"]:
                if param not in self._param_index:
                    index = len(self._param_list)
                    self._param_index[param] = index
                    self._param_list.append(param)

    def _sync_parent_state_from_children(self):
        for param in self._param_list:
            child_idx = self._param_to_child.get(param)
            if child_idx is None:
                continue

            child = self.optimizers[child_idx]
            if param in child.state:
                self.state[param] = child.state[param]
            else:
                self.state[param] = {}

    def _sync_children_from_parent(self):
        for param in self._param_list:
            child_idx = self._param_to_child.get(param)
            if child_idx is None:
                continue

            child = self.optimizers[child_idx]
            source_state = self.state.get(param, {})
            target_state = child.state.setdefault(param, {})
            target_state.clear()
            if source_state:
                target_state.update(source_state)

    def _clear_parent_state(self):
        # Parent state is only a transient mirror for serialization/deserialization.
        # Child optimizers remain the source of truth for runtime state.
        self.state.clear()

    def _sync_lr_to_children(self):
        """Sync learning rate from parent param_groups to child optimizers.
        
        When the scheduler updates parent_group['lr'], this method propagates
        the change to the child optimizers. 
        """
        for idx, parent_group in enumerate(self.param_groups):
            if idx >= len(self._group_metadata):
                continue
            
            metadata = self._group_metadata[idx]
            child_idx = metadata["child_index"]
            child = self.optimizers[child_idx]
            
            if not child.param_groups:
                continue
            
            parent_lr = parent_group.get("lr")
            if parent_lr is None:
                continue
            if not metadata.get("sync_lr_from_parent", True):
                continue
                
            child_group = child.param_groups[0]
            child_group["lr"] = parent_lr

    def state_dict(self):
        self._sync_parent_state_from_children()
        try:
            base_state = super().state_dict()
        finally:
            self._clear_parent_state()

        for idx, group in enumerate(base_state["param_groups"]):
            parent_group = self.param_groups[idx]

            if "lr" not in group:
                group["lr"] = parent_group.get("lr", self.defaults.get("lr"))
            
            if "initial_lr" not in group and "initial_lr" in parent_group:
                group["initial_lr"] = parent_group["initial_lr"]

            if "weight_decay" not in group:
                group["weight_decay"] = parent_group.get("weight_decay", self.defaults.get("weight_decay"))

            if "beta1" not in group:
                group["beta1"] = parent_group.get("beta1", self.defaults.get("beta1"))

            if "eps" not in group:
                group["eps"] = parent_group.get("eps", self.defaults.get("eps"))

            metadata = self._group_metadata[idx]
            nsadam_meta = {
                "rule": metadata["rule"],
                "config": copy.deepcopy(metadata.get("config", {})),
                "sync_lr_from_parent": metadata.get("sync_lr_from_parent", True),
            }
            child_pg = metadata.get("child_param_group")
            if child_pg:
                nsadam_meta["child_param_group"] = copy.deepcopy(child_pg)
            group["nsadam_meta"] = nsadam_meta

        return base_state

    def load_state_dict(self, state_dict):
        base_state = copy.deepcopy(state_dict)
        saved_groups = base_state.get("param_groups", [])

        metadata_payloads = []
        for group in saved_groups:
            metadata_payloads.append(group.pop("nsadam_meta", None))

        super().load_state_dict(base_state)
        self._build_param_index()

        if len(metadata_payloads) != len(self.param_groups):
            metadata_payloads.extend([None] * (len(self.param_groups) - len(metadata_payloads)))

        for idx, meta in enumerate(metadata_payloads):
            metadata = self._group_metadata[idx]
            if meta is None:
                meta = {
                    "rule": metadata["rule"],
                    "config": copy.deepcopy(metadata.get("config", {})),
                }
                if "child_param_group" in metadata:
                    meta["child_param_group"] = copy.deepcopy(metadata["child_param_group"])

            meta_rule = meta.get("rule", metadata["rule"])
            if meta_rule != metadata["rule"]:
                raise ValueError(f"Rule mismatch for parameter group {idx}")

            metadata["config"] = copy.deepcopy(meta.get("config", {}))
            metadata["sync_lr_from_parent"] = bool(meta.get("sync_lr_from_parent", True))

            child_pg = meta.get("child_param_group")
            metadata.pop("child_param_group", None)

            child_idx = metadata["child_index"]
            child = self.optimizers[child_idx]
            if child_pg:
                if child.param_groups:
                    target_group = child.param_groups[0]
                    for key, value in child_pg.items():
                        target_group[key] = value
                metadata["child_param_group"] = copy.deepcopy(child_pg)

            self.param_groups[idx]["nsadam_meta"] = copy.deepcopy(meta)

        self._sync_children_from_parent()
        self._sync_lr_to_children()
        self._clear_parent_state()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._sync_lr_to_children()
        # Propagate log_dynamics flag from wrapper to children before stepping.
        log_dyn = getattr(self, 'log_dynamics', False)
        for opt in self.optimizers:
            opt.log_dynamics = log_dyn
            opt.step()
        # Aggregate dynamics_stats from all children up to the wrapper.
        if log_dyn:
            self.dynamics_stats = {}
            for opt in self.optimizers:
                self.dynamics_stats.update(getattr(opt, 'dynamics_stats', {}) or {})
        return loss

    def zero_grad(self, set_to_none=False):
        for opt in self.optimizers:
            opt.zero_grad()

# Just for the sake of consistency with torch interface, otherwise we don't really implement Adam0 with L2 weight decay
class Adam0(AdamW0):
    pass




class AdamW0Staged(AdamW0):
    """Wrapper that constructs ``_Adam0Staged`` children with a fixed stage.

    Use via ``--opt eco0m-staged --ablation-stage {1,2,3,4}``.
    """

    def __init__(self, *args, stage=3, **kwargs):
        self._staged_stage = int(stage)
        super().__init__(*args, **kwargs)

    def _create_child_optimizer(self, rule, params, lr, beta1, eps, weight_decay, extra_kwargs):
        child_kwargs = dict(extra_kwargs)
        if rule == "adam0":
            child_kwargs["n_attention_heads"] = 1
            child_kwargs["lr"] = lr
            child_kwargs["beta1"] = beta1
            child_kwargs["eps"] = eps
            child_kwargs["weight_decay"] = weight_decay
            return _Adam0Staged(params=params, stage=self._staged_stage, **child_kwargs)
        elif rule == "adam0_att":
            child_kwargs["lr"] = lr
            child_kwargs["beta1"] = beta1
            child_kwargs["eps"] = eps
            child_kwargs["weight_decay"] = weight_decay
            return _Adam0Staged(params=params, stage=self._staged_stage, **child_kwargs)
        elif rule == "fallback":
            return self._create_fallback_optimizer(
                params=params,
                lr=lr,
                beta1=beta1,
                eps=eps,
                weight_decay=weight_decay,
            )
        raise ValueError(f"Unknown rule: {rule}")


class Adam0Staged(AdamW0Staged):
    """Public alias. Use via ``--opt eco0m-staged --ablation-stage {1,2,3,4}``."""
    pass

class AdamEF(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.0,
        decoupled_wd=True,
        stage=3,
        quantize_on_init=True,
        max_eco_correction_norm=None,
        log_dynamics=False,
    ):
        if stage not in (1, 2, 3, 4):
            raise ValueError(f"stage must be 1, 2, 3, or 4, got {stage}")

        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            decoupled_wd=decoupled_wd,
        )

        super().__init__(params, defaults)

        self.stage = int(stage)
        self.quantize_on_init = bool(quantize_on_init)
        self.max_eco_correction_norm = max_eco_correction_norm
        self.log_dynamics = bool(log_dynamics)
        self.dynamics_stats = {}

    @staticmethod
    def _has_quantizer(p):
        return hasattr(p, "quantizer") and p.quantizer is not None

    @torch.no_grad()
    def _maybe_calibrate(self, p, x):
        if not self._has_quantizer(p):
            return

        q = p.quantizer

        did_calibrate = getattr(q, "did_calibrate", True)

        if isinstance(did_calibrate, torch.Tensor):
            did_calibrate = bool(did_calibrate.item())
        else:
            did_calibrate = bool(did_calibrate)

        if hasattr(q, "calibrate_scale_") and not did_calibrate:
            q.calibrate_scale_(x)

    @torch.no_grad()
    def _quantize(self, p, x):
        """
        Returns quantized x. If p has no quantizer, returns x unchanged.
        """
        if not self._has_quantizer(p):
            return x

        self._maybe_calibrate(p, x)
        return p.quantizer.hard_quantize(x)

    @torch.no_grad()
    def _quantize_param_in_place_and_get_residual(self, p):
        """
        Quantizes p.data in-place and returns residual:

            residual = old_fp_value - quantized_value
        """
        theta_fp = p.detach()
        theta_hat = self._quantize(p, theta_fp)
        residual = theta_fp - theta_hat
        p.copy_(theta_hat)
        return residual.detach().clone()

    @torch.no_grad()
    def _clip_like_update(self, correction, reference, max_ratio):
        """
        Optional safeguard for ECO only.

        Clips ||correction|| <= max_ratio * ||reference||.

        This is not part of the ECO paper, so keep this disabled for clean experiments.
        It is useful for debugging catastrophic blowups.
        """
        if max_ratio is None:
            return correction

        correction_norm = correction.norm()
        reference_norm = reference.norm().clamp_min(1e-12)
        max_norm = float(max_ratio) * reference_norm

        if correction_norm > max_norm:
            correction = correction * (max_norm / correction_norm.clamp_min(1e-12))

        return correction

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.log_dynamics:
            self.dynamics_stats = {}

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            wd = group["weight_decay"]
            decoupled_wd = group["decoupled_wd"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()

                if grad.is_sparse:
                    raise RuntimeError("AdamECO does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    if self.stage in (2, 3, 4) and self.quantize_on_init:
                        residual0 = self._quantize_param_in_place_and_get_residual(p)
                        if self.stage == 2:
                            state["residual"] = residual0
                    elif self.stage == 2:
                        state["residual"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                state["step"] += 1
                t = state["step"]

                m = state["m"]
                v = state["v"]

                #
                if wd != 0.0 and not decoupled_wd:
                    if self.stage == 2:
                        theta_for_decay = p.detach() + state["residual"]
                    else:
                        theta_for_decay = p.detach()
                    grad = grad.add(theta_for_decay, alpha=wd)

                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_c1 = 1.0 - beta1 ** t
                bias_c2 = 1.0 - beta2 ** t

                v_hat_sqrt = (v / bias_c2).sqrt()
                denom = v_hat_sqrt.add(eps)

                adam_update = lr * m / (bias_c1 * denom)

                if self.stage == 1:
                    if wd != 0.0 and decoupled_wd:
                        p.mul_(1.0 - lr * wd)

                    p.add_(-adam_update)

                    if self.log_dynamics:
                        self.dynamics_stats[id(p)] = {
                            "stage": 1,
                            "step": t,
                            "theta_norm": p.norm().item(),
                            "grad_norm": grad.norm().item(),
                            "m_norm": m.norm().item(),
                            "v_norm": v.norm().item(),
                            "residual_norm": None,
                            "eco_error_norm": None,
                            "eco_correction_norm": None,
                        }

                    continue

                if self.stage == 2:
                    residual = state["residual"]

                    theta_master = p.detach() + residual

                    if wd != 0.0 and decoupled_wd:
                        theta_master = theta_master.mul(1.0 - lr * wd)

                    theta_tilde = theta_master - adam_update

                    theta_hat = self._quantize(p, theta_tilde)
                    residual_next = theta_tilde - theta_hat

                    if not torch.isfinite(residual_next).all():
                        raise FloatingPointError(
                            f"Non-finite residual in stage=2 at step {t}. "
                            f"residual_norm={residual_next.norm().item()}"
                        )

                    p.copy_(theta_hat)
                    state["residual"] = residual_next.detach().clone()

                    if self.log_dynamics:
                        self.dynamics_stats[id(p)] = {
                            "stage": 2,
                            "step": t,
                            "theta_hat_norm": theta_hat.norm().item(),
                            "theta_master_norm": theta_master.norm().item(),
                            "grad_norm": grad.norm().item(),
                            "m_norm": m.norm().item(),
                            "v_norm": v.norm().item(),
                            "residual_norm": residual_next.norm().item(),
                            "residual_over_theta": (
                                residual_next.norm()
                                / theta_hat.norm().clamp_min(1e-12)
                            ).item(),
                            "eco_error_norm": None,
                            "eco_correction_norm": None,
                        }

                    continue


                theta_start = p.detach()

                if wd != 0.0 and decoupled_wd:
                    theta_start = theta_start.mul(1.0 - lr * wd)

                theta_tilde = theta_start - adam_update
                theta_hat = self._quantize(p, theta_tilde)
                e_next = theta_tilde - theta_hat

                if not torch.isfinite(e_next).all():
                    raise FloatingPointError(
                        f"Non-finite quantization error at step {t}. "
                        f"e_norm={e_next.norm().item()}"
                    )

                if self.stage == 3:
                    inv_eta_t = (bias_c1 / lr) * denom

                    correction = inv_eta_t * (1.0 - 1.0 / beta1) * e_next

                    correction = self._clip_like_update(
                        correction,
                        reference=m,
                        max_ratio=self.max_eco_correction_norm,
                    )

                    if not torch.isfinite(correction).all():
                        raise FloatingPointError(
                            f"Non-finite ECO correction at step {t}. "
                            f"correction_norm={correction.norm().item()}"
                        )

                    m.add_(correction)

                    eco_correction_norm = correction.norm().item()
                else:
                    eco_correction_norm = None

                p.copy_(theta_hat)

                if self.log_dynamics:
                    e_norm = e_next.norm()
                    theta_norm = theta_hat.norm().clamp_min(1e-12)

                    self.dynamics_stats[id(p)] = {
                        "stage": self.stage,
                        "step": t,
                        "theta_hat_norm": theta_hat.norm().item(),
                        "grad_norm": grad.norm().item(),
                        "m_norm": m.norm().item(),
                        "v_norm": v.norm().item(),
                        "residual_norm": None,
                        "eco_error_norm": e_norm.item(),
                        "eco_error_over_theta": (e_norm / theta_norm).item(),
                        "eco_correction_norm": eco_correction_norm,
                    }

        return loss