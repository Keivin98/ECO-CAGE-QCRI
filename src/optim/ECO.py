import math
import torch
from torch.optim import Optimizer

# class ECOAdam(Optimizer):
#     def __init__(
#         self,
#         params,
#         lr=1e-3,
#         betas=(0.9, 0.999),
#         eps=1e-8,
#         weight_decay=0.0,
#     ):
#         if lr <= 0:
#             raise ValueError("Invalid learning rate")
#         if eps <= 0:
#             raise ValueError("Invalid eps")

#         beta1, beta2 = betas
#         if not (0.0 < beta1 < 1.0 and 0.0 <= beta2 < 1.0):
#             raise ValueError("Invalid betas")
#         if weight_decay < 0:
#             raise ValueError("Invalid weight_decay")

#         defaults = dict(
#             lr=lr,
#             betas=betas,
#             eps=eps,
#             weight_decay=weight_decay,
#         )
#         super().__init__(params, defaults)

#     @torch.no_grad()
#     def step(self, closure=None):
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         for group in self.param_groups:
#             lr = group["lr"]
#             beta1, beta2 = group["betas"]
#             eps = group["eps"]

#             for p in group["params"]:
#                 if p.grad is None:
#                     continue

#                 grad = p.grad
#                 if grad.is_sparse:
#                     raise RuntimeError("ECOAdam does not support sparse gradients")

#                 state = self.state[p]
#                 if len(state) == 0:
#                     state["step"] = 0
#                     state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                     state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

#                 m = state["exp_avg"]
#                 v = state["exp_avg_sq"]

#                 state["step"] += 1
#                 t = state["step"]

#                 m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
#                 v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

#                 bias_c1 = 1.0 - beta1 ** t
#                 bias_c2 = 1.0 - beta2 ** t

#                 m_hat = m / bias_c1
#                 denom = torch.sqrt(v / bias_c2) + eps

#                 p.addcdiv_(m_hat, denom, value=-lr)
#                 theta_tilde = p.clone()
                
#                 if hasattr(p, "quantizer") and p.quantizer is not None:
#                     theta_hat = p.quantizer.hard_quantize(theta_tilde)
#                 else:
#                     theta_hat = theta_tilde
                    
#                 e_next = theta_tilde - theta_hat

#                 # ECO momentum correction
#                 coeff = (bias_c1 / lr) * (1.0 - 1.0 / beta1)
#                 m.add_(coeff * denom * e_next)

#                 p.copy_(theta_hat)


#         return loss

class ECOAdam(Optimizer):
    """
    AdamW-style ECO optimizer.

    p.data stores quantized weights.
    No master weights.
    No explicit residual buffer.
    Quantization error is injected into exp_avg.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        quantize_on_init=True,
        decoupled_weight_decay=True,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0:
            raise ValueError(f"Invalid eps: {eps}")

        beta1, beta2 = betas
        if not (0.0 < beta1 < 1.0):
            raise ValueError(f"Invalid beta1: {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid beta2: {beta2}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            quantize_on_init=quantize_on_init,
            decoupled_weight_decay=decoupled_weight_decay,
        )
        super().__init__(params, defaults)

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
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            quantize_on_init = group["quantize_on_init"]
            decoupled_wd = group["decoupled_weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()

                if grad.is_sparse:
                    raise RuntimeError("ECOAdam does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                    # Important for true no-master-weight training:
                    # stored parameter starts on the quantization grid.
                    if quantize_on_init and self._has_quantizer(p):
                        theta_hat_0 = self._quantize(p, p.detach())
                        p.copy_(theta_hat_0)

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # Coupled weight decay, if requested.
                if weight_decay != 0.0 and not decoupled_wd:
                    grad = grad.add(p.detach(), alpha=weight_decay)

                # Adam moment updates.
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_c1 = 1.0 - beta1 ** t
                bias_c2 = 1.0 - beta2 ** t

                denom = (v / bias_c2).sqrt().add_(eps)

                # Start from quantized weights.
                theta_start = p.detach()

                # Decoupled AdamW weight decay.
                if weight_decay != 0.0 and decoupled_wd:
                    theta_start = theta_start.mul(1.0 - lr * weight_decay)

                # Temporary high-precision updated weight.
                theta_tilde = theta_start - lr * m / (bias_c1 * denom)

                # Quantize back to grid.
                theta_hat = self._quantize(p, theta_tilde)

                # Quantization residual.
                e_next = theta_tilde - theta_hat

                # ECO correction:
                # inv_eta_t = bias_c1 * denom / lr
                # correction = inv_eta_t * (1 - 1 / beta1) * e_next
                inv_eta_t = (bias_c1 / lr) * denom
                correction = inv_eta_t * (1.0 - 1.0 / beta1) * e_next

                if not torch.isfinite(e_next).all():
                    raise FloatingPointError(
                        f"Non-finite ECO quantization error at step {t}: "
                        f"||e_next||={e_next.norm().item()}"
                    )

                if not torch.isfinite(correction).all():
                    raise FloatingPointError(
                        f"Non-finite ECO correction at step {t}: "
                        f"||correction||={correction.norm().item()}"
                    )

                # Inject ECO correction into first moment.
                m.add_(correction)

                # Store quantized weights.
                p.copy_(theta_hat)

        return loss

class ECOAdamHM(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        adjust_lr=False, 
    ):
        if lr <= 0:
            raise ValueError("Invalid learning rate")
        if eps <= 0:
            raise ValueError("Invalid eps")

        beta1, beta2 = betas
        if not (0.0 < beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("Invalid betas")
        if weight_decay < 0:
            raise ValueError("Invalid weight_decay")

        if adjust_lr:
            lr = lr * math.sqrt((1 - beta1) / (1 + beta1))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            adjust_lr=adjust_lr
        )
        super().__init__(params, defaults)
    
    def zero_grad(self, set_to_none: bool = False):
        self.decay_grad()
    
    @torch.no_grad()
    def decay_grad(self):
        for group in self.param_groups:
            beta1 = group["betas"][0]
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.mul_(beta1)
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

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ECOAdam does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                v = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                m = (1-beta1)*grad
                v.mul_(beta2).addcmul_(grad, grad, value=(1.0 - beta2)*(1-beta1*beta1))

                bias_c1 = 1.0 - beta1 ** t
                bias_c2 = 1.0 - beta2 ** t

                m_hat = m / bias_c1
                denom = torch.sqrt(v / bias_c2) + eps

                p.addcdiv_(m_hat, denom, value=-lr)
                theta_tilde = p.clone()
                
                if hasattr(p, "quantizer") and p.quantizer is not None:
                    theta_hat = p.quantizer.hard_quantize(theta_tilde)
                else:
                    theta_hat = theta_tilde
                    
                e_next = theta_tilde - theta_hat

                # ECO momentum correction
                coeff = (bias_c1 / lr) * (1.0 - 1.0 / beta1)
                
                # m=(1-beta1)*grad
                # m.add_(coeff * denom * e_next)
                grad.add_(coeff * denom * e_next/(1-beta1))
                p.copy_(theta_hat)
        return loss

class ECOAdam0M(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        adjust_lr=False, 
    ):
        if lr <= 0:
            raise ValueError("Invalid learning rate")
        if eps <= 0:
            raise ValueError("Invalid eps")

        beta1, beta2 = betas
        if not (0.0 < beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("Invalid betas")
        if weight_decay < 0:
            raise ValueError("Invalid weight_decay")

        if adjust_lr:
            lr = lr * math.sqrt((1 - beta1) / (1 + beta1))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            adjust_lr=adjust_lr
        )
        super().__init__(params, defaults)
    
    def zero_grad(self, set_to_none: bool = False):
        self.decay_grad()
    
    @torch.no_grad()
    def decay_grad(self):
        for group in self.param_groups:
            beta1 = group["betas"][0]
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.mul_(beta1)
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

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ECOAdam does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                v = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                m = (1-beta1)*grad
                v.mul_(beta2).addcmul_(grad, grad, value=(1.0 - beta2)*(1-beta1*beta1))

                if v.dim()==2:
                    naive_v=m**2
                    R=torch.sum(naive_v,dim=1)
                    C=torch.sum(naive_v,dim=0)
                    s=torch.sum(naive_v)
                    v = torch.outer(R, C) / s
                    # print(v.shape)

                bias_c1 = 1.0 - beta1 ** t
                bias_c2 = 1.0 - beta2 ** t

                m_hat = m / bias_c1
                denom = torch.sqrt(v / bias_c2) + eps

                p.addcdiv_(m_hat, denom, value=-lr)
                theta_tilde = p.clone()
                
                if hasattr(p, "quantizer") and p.quantizer is not None:
                    theta_hat = p.quantizer.hard_quantize(theta_tilde)
                else:
                    theta_hat = theta_tilde
                    
                e_next = theta_tilde - theta_hat

                # ECO momentum correction
                coeff = (bias_c1 / lr) * (1.0 - 1.0 / beta1)
                
                # m=(1-beta1)*grad
                # m.add_(coeff * denom * e_next)
                grad.add_(coeff * denom * e_next/(1-beta1))
                p.copy_(theta_hat)
        return loss