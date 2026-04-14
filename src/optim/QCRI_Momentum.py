import math
import torch
from torch.optim import Optimizer
import csv
import os

import json, os

class ProbeJSONL:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "a")

    def write(self, obj):
        self.f.write(json.dumps(obj) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()

def select_probes_quantized(param_groups, n_tensors=3, n_per_tensor=4, seed=0):
    gen = torch.Generator().manual_seed(seed)
    params = []
    for group in param_groups:
        for p in group["params"]:
            q = getattr(p, "quantizer", None)
            if p.requires_grad and (q is not None) and hasattr(q, "scale"):
                params.append(p)

    if len(params) == 0:
        raise RuntimeError("No parameters with p.quantizer.scale found. Probes would be meaningless.")

    # pick a mix: smallest among quantized (or remove sorting)
    params = sorted(params, key=lambda x: x.numel())
    chosen = params[:n_tensors]

    probes = []
    for ti, p in enumerate(chosen):
        k = min(n_per_tensor, p.numel())
        idxs = torch.randint(0, p.numel(), (k,), generator=gen).tolist()
        for j, idx in enumerate(idxs):
            probes.append((p, idx, f"qt{ti}_i{j}"))
    return probes


class QCRIECOAdam(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        collect_stats: bool = True,
    ):
        if lr <= 0:
            raise ValueError("Invalid lr")
        b1, b2 = betas
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError("Invalid betas")
        if eps <= 0:
            raise ValueError("Invalid eps")

        defaults = dict(lr=lr, betas=betas, eps=eps, collect_stats=collect_stats)
        super().__init__(params, defaults)
        self.step_stats = []

    @staticmethod
    def _mean_abs(x: torch.Tensor) -> float:
        return x.detach().abs().mean().item()

    @staticmethod
    def _frac_zero(x: torch.Tensor) -> float:
        return (x.detach() == 0).float().mean().item()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_stats = []

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            collect_stats = group.get("collect_stats", True)

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("QCRIECOAdam does not support sparse gradients")

                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)

                m = st["m"]
                v = st["v"]

                st["step"] += 1
                t = st["step"]

                # Adam moments (uncorrected)
                m.mul_(b1).add_(g, alpha=(1.0 - b1))
                v.mul_(b2).addcmul_(g, g, value=(1.0 - b2))

                # Bias correction
                b1t = b1 ** t
                b2t = b2 ** t
                bc1 = 1.0 - b1t
                bc2 = 1.0 - b2t

                m_hat = m / bc1
                v_hat = v / bc2
                denom_hat = v_hat.sqrt().add_(eps)

                theta_tilde = p - lr * (m_hat / denom_hat)

                q = getattr(p, "quantizer", None)
                if q is not None and hasattr(q, "hard_quantize"):
                    theta_hat = q.hard_quantize(theta_tilde)
                    e = theta_tilde - theta_hat

                    eco_mul = (bc1 / lr) * (1.0 - 1.0 / b1)
                    eco_scale = (v / bc2).sqrt().add_(eps)  # sqrt(v_hat) + eps
                    m.add_(eco_mul * eco_scale * e)

                    p.copy_(theta_hat)
                else:
                    p.copy_(theta_tilde)

                if collect_stats:
                    stats = {
                        "m_meanabs": self._mean_abs(m),
                        "v_meanabs": self._mean_abs(v),
                    }
                    if q is not None and hasattr(q, "scale"):
                        s = q.scale.detach()
                        stats["scale"] = float(s.item()) if s.numel() == 1 else float(s.mean().item())
                    if q is not None and hasattr(q, "hard_quantize"):
                        stats["e_meanabs"] = self._mean_abs(e) if "e" in locals() else 0.0
                    self.step_stats.append(stats)

        return loss



class QCRITernaryCarryAdam(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        tau=0.5,
        m_fire_mul=0.0,
        carry_decay=1.0,
        carry_clip=None,
        collect_stats=True,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            tau=tau,
            m_fire_mul=m_fire_mul,
            carry_decay=carry_decay,
            carry_clip=carry_clip,
            collect_stats=collect_stats,
        )
        super().__init__(params, defaults)
        self.step_stats = []

    @staticmethod
    def ternary_round(x, tau):
        sign = torch.sign(x)
        ax = x.abs()
        flo = torch.floor(ax)
        frac = ax - flo
        ri = flo + (frac >= tau).to(ax.dtype)
        return sign * ri.clamp(-1, 1)

    @staticmethod
    def _mean_abs(x):
        return x.detach().abs().mean().item()

    @staticmethod
    def _frac_zero(x):
        return (x.detach() == 0).float().mean().item()

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_stats = []

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            tau = group["tau"]
            m_fire_mul = group["m_fire_mul"]
            carry_decay = group["carry_decay"]
            carry_clip = group["carry_clip"]
            collect_stats = group["collect_stats"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Sparse gradients not supported")

                st = self.state[p]
                if len(st) == 0:
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                    st["carry"] = torch.zeros_like(p)

                m = st["m"]
                v = st["v"]
                carry = st["carry"]

                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)

                denom = v.sqrt().add_(eps)

                u = -lr * m / denom
                u_eff = u + carry

                q = getattr(p, "quantizer", None)
                if q is not None and hasattr(q, "scale"):
                    delta = (1.0 / q.scale).to(p.device, p.dtype)
                else:
                    delta = torch.ones((), device=p.device, dtype=p.dtype)

                z = u_eff / delta
                uq_int = self.ternary_round(z, tau)
                u_q = uq_int * delta

                e = u_eff - u_q

                if carry_clip is not None:
                    e = e.clamp(-carry_clip, carry_clip)

                st["carry"] = carry_decay * e

                p.add_(u_q)

                fired = uq_int != 0

                if m_fire_mul == 0:
                    m.masked_fill_(fired, 0.0)
                else:
                    m[fired] *= m_fire_mul

                if collect_stats:
                    stats = {}

                    stats["update_meanabs_pre_carry"] = self._mean_abs(u)
                    stats["update_meanabs_post_carry"] = self._mean_abs(u_eff)
                    stats["uq_meanabs"] = self._mean_abs(u_q)
                    stats["uq_frac_zero"] = self._frac_zero(u_q)
                    stats["remainder_meanabs"] = self._mean_abs(e)
                    stats["carry_meanabs"] = self._mean_abs(st["carry"])
                    stats["carry_frac_zero"] = self._frac_zero(st["carry"])

                    stats["fire_frac"] = fired.float().mean().item()
                    stats["fire_count"] = int(fired.sum().item())
                    stats["m_meanabs"] = self._mean_abs(m)

                    if q is not None and hasattr(q, "scale"):
                        s = q.scale.detach()
                        stats["scale"] = float(s.item()) if s.numel() == 1 else float(s.mean().item())

                        if hasattr(q, "qmin") and hasattr(q, "qmax"):
                            xi = torch.round(p.detach().float() * s.to(p.device))
                            sat = (xi <= q.qmin) | (xi >= q.qmax)
                            stats["sat_frac"] = sat.float().mean().item()

                    self.step_stats.append(stats)

        return loss

class QCRITernaryAdam(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        tau=0.5,
        m_fire_mul=0.0,
        collect_stats=True,
        probe_path="logs/ternaryadam_probes1.jsonl",
        probe_n_tensors=3,
        probe_n_per_tensor=4,
        probe_seed=0,
        log_every=1,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            tau=tau,
            m_fire_mul=m_fire_mul,
            collect_stats=collect_stats,
        )
        super().__init__(params, defaults)

        self.step_stats = []
        self.global_step = 0

        self.probes = None
        self.probe_logger = ProbeJSONL(probe_path) if probe_path else None
        self.probe_n_tensors = probe_n_tensors
        self.probe_n_per_tensor = probe_n_per_tensor
        self.probe_seed = probe_seed
        self.log_every = int(log_every)

    @staticmethod
    def ternary_round(x, tau):
        sign = torch.sign(x)
        ax = x.abs()
        flo = torch.floor(ax)
        frac = ax - flo
        ri = flo + (frac >= tau).to(ax.dtype)
        return sign * ri.clamp(-1, 1)

    @staticmethod
    def _mean_abs(x):
        return x.detach().abs().mean().item()

    @staticmethod
    def _frac_zero(x):
        return (x.detach() == 0).float().mean().item()

    @staticmethod
    def _scalar_from(t: torch.Tensor | None, idx: int):
        """Return scalar at flat idx; works for scalars and tensors."""
        if t is None:
            return None
        t = t.detach()
        if t.numel() == 1:
            return float(t.item())
        return float(t.flatten()[idx].item())

    @staticmethod
    def _scale_mean(q):
        """Return mean scale as Python float (handles scalar or tensor)."""
        s = q.scale
        if torch.is_tensor(s):
            s = s.detach()
            return float(s.item()) if s.numel() == 1 else float(s.mean().item())
        return float(s)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_stats = []
        self.global_step += 1

        # init probes once (ONLY quantized params)
        if self.probes is None and self.probe_logger is not None:
            self.probes = select_probes_quantized(
                self.param_groups,
                n_tensors=self.probe_n_tensors,
                n_per_tensor=self.probe_n_per_tensor,
                seed=self.probe_seed,
            )

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            tau = group["tau"]
            m_fire_mul = group["m_fire_mul"]
            collect_stats = group["collect_stats"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Sparse gradients not supported")

                st = self.state[p]
                if len(st) == 0:
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)

                m = st["m"]
                v = st["v"]

                m.mul_(1.0).add_(g, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)

                denom = v.sqrt().add_(eps)
                u = -lr * m / denom

                q = getattr(p, "quantizer", None)
                if q is not None and hasattr(q, "scale"):
                    delta = (1.0 / q.scale).to(device=p.device, dtype=p.dtype)
                else:
                    delta = torch.ones((), device=p.device, dtype=p.dtype)

                z = u / delta
                uq_int = self.ternary_round(z, tau)
                u_q = uq_int * delta

                if self.probe_logger is not None:
                    st["_denom"] = denom
                    st["_u"] = u
                    st["_delta"] = delta
                    st["_z"] = z
                    st["_uq_int"] = uq_int
                    st["_u_q"] = u_q

                p.add_(u_q)

                fired = uq_int != 0
                if m_fire_mul == 0.0:
                    m.masked_fill_(fired, 0.0)
                else:
                    m[fired] *= m_fire_mul

                if collect_stats:
                    stats = {
                        "update_meanabs_pre_carry": self._mean_abs(u),
                        "uq_meanabs": self._mean_abs(u_q),
                        "uq_frac_zero": self._frac_zero(u_q),
                        "fire_frac": fired.float().mean().item(),
                        "fire_count": int(fired.sum().item()),
                        "m_meanabs": self._mean_abs(m),
                    }

                    if q is not None and hasattr(q, "scale"):
                        sm = self._scale_mean(q)
                        stats["scale"] = sm

                        if hasattr(q, "qmin") and hasattr(q, "qmax"):
                            s = q.scale.detach()
                            xi = torch.round(p.detach().float() * s.to(p.device))
                            sat = (xi <= q.qmin) | (xi >= q.qmax)
                            stats["sat_frac"] = sat.float().mean().item()

                    self.step_stats.append(stats)

        # write ONE record per step
        if (
            self.probe_logger is not None
            and self.probes is not None
            and (self.global_step % self.log_every == 0)
        ):
            out = {"step": self.global_step, "probes": []}

            for (pp, idx, label) in self.probes:
                stp = self.state.get(pp, {})
                q_probe = getattr(pp, "quantizer", None)
                has_q = q_probe is not None
                has_scale = has_q and hasattr(q_probe, "scale")

                scale_mean = None
                delta_should_be = None
                if has_scale:
                    scale_mean = self._scale_mean(q_probe)
                    delta_should_be = 1.0 / scale_mean

                out["probes"].append({
                    "label": label,
                    "has_q": has_q,
                    "has_scale": has_scale,
                    "scale_mean": scale_mean,
                    "delta_should_be": delta_should_be,

                    "p": self._scalar_from(pp, idx),
                    "g": self._scalar_from(pp.grad, idx) if pp.grad is not None else None,
                    "m": self._scalar_from(stp.get("m"), idx),
                    "v": self._scalar_from(stp.get("v"), idx),
                    "denom": self._scalar_from(stp.get("_denom"), idx),
                    "u": self._scalar_from(stp.get("_u"), idx),
                    "delta": self._scalar_from(stp.get("_delta"), idx),
                    "z": self._scalar_from(stp.get("_z"), idx),
                    "uq_int": int(self._scalar_from(stp.get("_uq_int"), idx)) if stp.get("_uq_int") is not None else None,
                    "u_q": self._scalar_from(stp.get("_u_q"), idx),
                })

            self.probe_logger.write(out)

        return loss