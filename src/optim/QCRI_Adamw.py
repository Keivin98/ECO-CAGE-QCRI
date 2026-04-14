import math
from typing import Callable, Iterable, Optional, Any

import torch
from torch.optim import Optimizer

class NaiveOptimizer(Optimizer):
    #this is an optimizer that does not carry quantization error, it just hard quantizes the update and applies the weight decay
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.update_norms = []
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Any]] = None):
        loss = None
        self.update_norms = []
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("NaiveOptiomizer does not support sparse gradients")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                m, v =  state["exp_avg"], state["exp_avg_sq"]
                
                state["step"] += 1
                t = state["step"]
                m.mul_(b1).add_(grad, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
                bias_c1 = 1.0 - (b1 ** t)
                bias_c2 = 1.0 - (b2 ** t)
                step_size = lr * math.sqrt(bias_c2) / bias_c1
                denom = v.sqrt().add_(eps)
                factor = step_size / denom
                update = -factor * m
                update_norm = update.norm().item()
                self.update_norms.append(update_norm)
                if self._has_quantizer(p):
                    quantizer = p.quantizer
                    has_hard_quantize = hasattr(quantizer, "hard_quantize")
                    if has_hard_quantize:
                        u_q = quantizer.hard_quantize(update)
                        p.add_(u_q)
                        # if wd != 0.0:
                        #     p.mul_(1 - lr * wd)
                        p.copy_(quantizer.hard_quantize(p))
                    else:
                        p.add_(update)
                        p.copy_(quantizer.hard_quantize(p))
                else:
                    p.add_(update)
                    # if wd != 0.0:
                    #     p.mul_(1 - lr * wd)
        return loss

    @staticmethod
    def _has_quantizer(p: torch.Tensor) -> bool:
        return hasattr(p, "quantizer") and p.quantizer is not None

class QCRIAdamWGradAccumulator(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        carry_clip: float = 1e3,
        carry_decay: float = 0.999,
    ):
        if lr <= 0:
            raise ValueError("Invalid lr")
        b1, b2 = betas
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError("Invalid betas")
        if eps <= 0:
            raise ValueError("Invalid eps")
        if weight_decay < 0:
            raise ValueError("Invalid weight_decay")

        self.carry_clip = carry_clip
        self.carry_decay = carry_decay
        # Track update norms for logging
        self.step_stats = [] 
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _has_quantizer(p: torch.Tensor) -> bool:
        return hasattr(p, "quantizer") and p.quantizer is not None
    @staticmethod
    def _mean_abs(x: torch.Tensor) -> float:
        return x.detach().abs().mean().item()

    @staticmethod
    def _frac_zero(x: torch.Tensor) -> float:
        # fraction of elements exactly 0 (after quantization this is meaningful)
        return (x.detach() == 0).float().mean().item()

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Any]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_stats = []
        for group in self.param_groups:

            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("QCRIAdamWGradAccumulator does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["carry"] = torch.zeros_like(p)

                # Add carry from previous step to current gradient
                # if "carry" in state and state["carry"] is not None:
                #     grad = grad + state["carry"]

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # Standard Adam momentum updates (using gradient with carry)
                m.mul_(b1).add_(grad, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)

                # Bias correction
                bias_c1 = 1.0 - (b1 ** t)
                bias_c2 = 1.0 - (b2 ** t)
                step_size = lr * math.sqrt(bias_c2) / bias_c1

                # Compute factor and update
                denom = v.sqrt().add_(eps)
                factor = step_size / denom
                update = -factor * m

                update_pre_carry = update.clone()

                carry_prev = state.get("carry", None)
                if carry_prev is not None:
                    update = update + carry_prev
                update_post_carry = update

                if self._has_quantizer(p):
                    quantizer = p.quantizer
                    has_hard_quantize = hasattr(quantizer, "hard_quantize")
                    if has_hard_quantize:
                        u_q = quantizer.hard_quantize(update_post_carry)
                        
                        p.add_(u_q)
                    
                        p.copy_(quantizer.hard_quantize(p))
                        
                        remainder = update_post_carry - u_q
                        
                        carry = remainder.clamp(-self.carry_clip, self.carry_clip)
                        carry = carry * self.carry_decay
                        
                        state["carry"] = carry
                        
                        self.step_stats.append({
                            "param_shape": tuple(p.shape),
                            "update_meanabs_pre_carry": self._mean_abs(update_pre_carry),
                            "update_meanabs_post_carry": self._mean_abs(update_post_carry),
                            "uq_meanabs": self._mean_abs(u_q),
                            "uq_frac_zero": self._frac_zero(u_q),
                            "remainder_meanabs": self._mean_abs(remainder),
                            "carry_meanabs": self._mean_abs(carry),
                            "carry_frac_zero": self._frac_zero(carry),  # often useful too
                        })
                    else:
                        p.add_(update_post_carry)
                        p.copy_(quantizer.hard_quantize(p))
                else:
                    p.add_(update)
                    # if wd != 0.0:
                    #     p.mul_(1 - lr * wd)

        return loss

class QCRIAdamWTernary(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        carry_clip: float = 1e3,
        carry_decay: float = 0.999,
    ):
        if lr <= 0:
            raise ValueError("Invalid lr")
        b1, b2 = betas
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError("Invalid betas")
        if eps <= 0:
            raise ValueError("Invalid eps")
        if weight_decay < 0:
            raise ValueError("Invalid weight_decay")

        self.carry_clip = carry_clip
        self.carry_decay = carry_decay
        # Track update norms for logging
        self.step_stats = [] 
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _has_quantizer(p: torch.Tensor) -> bool:
        return hasattr(p, "quantizer") and p.quantizer is not None
    @staticmethod
    def _mean_abs(x: torch.Tensor) -> float:
        return x.detach().abs().mean().item()

    @staticmethod
    def _frac_zero(x: torch.Tensor) -> float:
        # fraction of elements exactly 0 (after quantization this is meaningful)
        return (x.detach() == 0).float().mean().item()

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Any]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_stats = []
        for group in self.param_groups:

            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("QCRIAdamWGradAccumulator does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["carry"] = torch.zeros_like(p)

                # Add carry from previous step to current gradient
                # if "carry" in state and state["carry"] is not None:
                #     grad = grad + state["carry"]

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # Standard Adam momentum updates (using gradient with carry)
                m.mul_(b1).add_(grad, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)

                # Bias correction
                bias_c1 = 1.0 - (b1 ** t)
                bias_c2 = 1.0 - (b2 ** t)
                step_size = lr * math.sqrt(bias_c2) / bias_c1

                # Compute factor and update
                denom = v.sqrt().add_(eps)
                factor = step_size / denom
                update = -factor * m

                update_pre_carry = update.clone()

                carry_prev = state.get("carry", None)
                if carry_prev is not None:
                    update = update + carry_prev
                update_post_carry = update

                if self._has_quantizer(p):
                    quantizer = p.quantizer
                    has_hard_quantize = hasattr(quantizer, "hard_quantize")
                    has_ternary_quantize = hasattr(quantizer, "ternary_quantize")
                    if has_hard_quantize and has_ternary_quantize:
                        u_q = quantizer.ternary_quantize(update_post_carry)
                        
                        p.add_(u_q)
                    
                        p.copy_(quantizer.hard_quantize(p))
                        
                        remainder = update_post_carry - u_q
                        
                        carry = remainder.clamp(-self.carry_clip, self.carry_clip)
                        carry = carry * self.carry_decay
                        
                        state["carry"] = carry
                        
                        self.step_stats.append({
                            "param_shape": tuple(p.shape),
                            "update_meanabs_pre_carry": self._mean_abs(update_pre_carry),
                            "update_meanabs_post_carry": self._mean_abs(update_post_carry),
                            "uq_meanabs": self._mean_abs(u_q),
                            "uq_frac_zero": self._frac_zero(u_q),
                            "remainder_meanabs": self._mean_abs(remainder),
                            "carry_meanabs": self._mean_abs(carry),
                            "carry_frac_zero": self._frac_zero(carry),  # often useful too
                        })
                    else:
                        p.add_(update_post_carry)
                        p.copy_(quantizer.hard_quantize(p))
                else:
                    p.add_(update)
                    # if wd != 0.0:
                    #     p.mul_(1 - lr * wd)

        return loss


class QCRIAdamWTernaryStep2(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        carry_clip: float = 1e3,
        carry_decay: float = 0.999,
    ):
        if lr <= 0:
            raise ValueError("Invalid lr")
        b1, b2 = betas
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError("Invalid betas")
        if eps <= 0:
            raise ValueError("Invalid eps")
        if weight_decay < 0:
            raise ValueError("Invalid weight_decay")

        self.carry_clip = carry_clip
        self.carry_decay = carry_decay
        # Track update norms for logging
        self.step_stats = [] 
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _has_quantizer(p: torch.Tensor) -> bool:
        return hasattr(p, "quantizer") and p.quantizer is not None
    @staticmethod
    def _mean_abs(x: torch.Tensor) -> float:
        return x.detach().abs().mean().item()

    @staticmethod
    def _frac_zero(x: torch.Tensor) -> float:
        # fraction of elements exactly 0 (after quantization this is meaningful)
        return (x.detach() == 0).float().mean().item()

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Any]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_stats = []
        for group in self.param_groups:

            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("QCRIAdamWGradAccumulator does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["carry"] = torch.zeros_like(p)

                # Add carry from previous step to current gradient
                # if "carry" in state and state["carry"] is not None:
                #     grad = grad + state["carry"]

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # Standard Adam momentum updates (using gradient with carry)
                m.mul_(b1).add_(grad, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)

                # Bias correction
                bias_c1 = 1.0 - (b1 ** t)
                bias_c2 = 1.0 - (b2 ** t)
                step_size = lr * math.sqrt(bias_c2) / bias_c1

                # Compute factor and update
                denom = v.sqrt().add_(eps)
                factor = step_size / denom
                update = -factor * m

                update_pre_carry = update.clone()

                carry_prev = state.get("carry", None)
                if carry_prev is not None:
                    update = update + carry_prev
                update_post_carry = update

                if self._has_quantizer(p):
                    quantizer = p.quantizer
                    has_hard_quantize = hasattr(quantizer, "hard_quantize")
                    has_ternary_quantize_conditional = hasattr(quantizer, "ternary_quantize_conditional")
                    if has_hard_quantize and has_ternary_quantize_conditional:
                        u_q = quantizer.ternary_quantize_conditional(update_post_carry)
                        
                        p.add_(u_q)
                    
                        p.copy_(quantizer.hard_quantize(p))
                        
                        remainder = update_post_carry - u_q
                        
                        carry = remainder.clamp(-self.carry_clip, self.carry_clip)
                        carry = carry * self.carry_decay
                        carry[u_q != 0] = 0
                        
                        state["carry"] = carry
                        
                        self.step_stats.append({
                            "param_shape": tuple(p.shape),
                            "update_meanabs_pre_carry": self._mean_abs(update_pre_carry),
                            "update_meanabs_post_carry": self._mean_abs(update_post_carry),
                            "uq_meanabs": self._mean_abs(u_q),
                            "uq_frac_zero": self._frac_zero(u_q),
                            "remainder_meanabs": self._mean_abs(remainder),
                            "carry_meanabs": self._mean_abs(carry),
                            "carry_frac_zero": self._frac_zero(carry),  # often useful too
                        })
                    else:
                        p.add_(update_post_carry)
                        p.copy_(quantizer.hard_quantize(p))
                else:
                    p.add_(update)
                    # if wd != 0.0:
                    #     p.mul_(1 - lr * wd)

        return loss


class QCRIAdamW(Optimizer):
    """
    AdamW optimizer with quantization-aware updates.
    
    Carries quantization error in the momentum buffer.
    """
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        carry_clip: float = 1e3,  # Clip to prevent explosion
        carry_decay: float = 0.999,  # Slight decay for stability
    ):
        if lr <= 0:
            raise ValueError("Invalid lr")
        b1, b2 = betas
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError("Invalid betas")
        if eps <= 0:
            raise ValueError("Invalid eps")
        if weight_decay < 0:
            raise ValueError("Invalid weight_decay")

        self.carry_clip = carry_clip
        self.carry_decay = carry_decay
        # Track update norms for logging
        self.update_norms = []
        self.carry_norms = []
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def _has_quantizer(p: torch.Tensor) -> bool:
        return hasattr(p, "quantizer") and p.quantizer is not None

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Any]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Clear update norms from previous step
        self.update_norms = []
        self.carry_norms = []
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("QCRIAdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # Standard Adam momentum updates
                m.mul_(b1).add_(grad, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)

                # Bias correction
                bias_c1 = 1.0 - (b1 ** t)
                bias_c2 = 1.0 - (b2 ** t)
                step_size = lr * math.sqrt(bias_c2) / bias_c1

                # Compute factor and update
                denom = v.sqrt().add_(eps)
                factor = step_size / denom
                update = -factor * m

                # Track update norm for logging
                update_norm = update.norm().item()
                self.update_norms.append(update_norm)

                if self._has_quantizer(p):
                    quantizer = p.quantizer
                    is_fixed_scale = hasattr(quantizer, 'scale')
                    
                    if is_fixed_scale:
                        u_q = quantizer.hard_quantize(update)
                        
                        p.add_(u_q)

                        if wd != 0.0:
                            p.mul_(1 - lr * wd)
                        
                        # assert that p is quantized
                        assert (p == quantizer.hard_quantize(p)).all(), f"p is not quantized: {p} != {quantizer.hard_quantize(p)}"
                        p.copy_(quantizer.hard_quantize(p))
                        
                        remainder = update - u_q
                        remainder = remainder.clamp(-self.carry_clip, self.carry_clip)
                        carry_next = (remainder / factor.clamp(min=eps))
                        
                        #here do i need the (1 - b1) factor?
                        # carry_next.mul_((1 - b1) * self.carry_decay/b1)
                        carry_next.mul_(self.carry_decay/b1)
                        carry_norm = carry_next.norm().item()
                        self.carry_norms.append(carry_norm)
                        m.add_(carry_next)
                    else:
                        p.add_(update)
                        p.copy_(quantizer.hard_quantize(p))
                else:
                    p.add_(update)
                    if wd != 0.0:
                        p.mul_(1 - lr * wd)

        return loss
