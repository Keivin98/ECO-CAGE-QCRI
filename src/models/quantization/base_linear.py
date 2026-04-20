import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
from scipy import integrate
from scipy.stats import norm
import torch._dynamo as dynamo

try:
    from fast_hadamard_transform import hadamard_transform
    HAS_HADAMARD = True
except ImportError:
    HAS_HADAMARD = False
    # Create dummy hadamard_transform that returns the input unchanged
    # This allows Hadamard classes to be defined without error (they won't work if used)
    def hadamard_transform(x, scale=1.0):
        # Return input unchanged (fake Hadamard transform)
        # This is only called during class definition for aux_matrix
        return x * scale


class BaseQuantizer(nn.Module):
    def __init__(self, bits=4, centered=True):
        super().__init__()
        self.bits = bits
        self.n_levels = 2**bits
        self.centered = centered

    def cast(self, x):
        # This method can be inherited to use any casting, e.g. int, fp(e2m1, e1m2,...), optimal gaussian, etc.
        # NOTE: raise_zero should never be used with FP quantization
        return x.round()

    def ste_cast(self, x):
        return (self.cast(x) - x).detach() + x

    def grad_scale(self, x, scale):
        return (x - x * scale).detach() + x * scale

    @torch.no_grad()
    def hard_quantize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("hard_quantize must be implemented for this quantizer.")

    def forward(self, x):
        raise NotImplementedError


class NoQuantizer(BaseQuantizer):
    def __init__(self, **kwargs):
        super().__init__(16)

    def forward(self, x):
        return x

    def hard_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def hard_quantize(self, x: torch.Tensor) -> torch.Tensor:
        return x



OPTIMAL_GAUSSIAN_SCALES = {
    1: 0.7978845587140913,
    1.585: 1.2240089519030855,
    2: 1.4935346200015913,
    3: 2.051068354131873,
    4: 2.513930578568423,
    5: 2.9160938834961225,
    6: 3.276597282593217,
    7: 3.6010497188221655,
    8: 3.884938678807525,
}


class AbsMaxQuantizer(BaseQuantizer):
    def forward(self, x):
        with torch.no_grad():
            scale = torch.max(torch.abs(x), dim=-1, keepdim=True) + 1e-8
            step = scale * 2 / (self.n_levels - 1)
            xq = torch.round(x / step + 1 / 2) * step - step / 2
        return x + (xq - x).detach()


class MSEQuantizer(BaseQuantizer):
    def __init__(self, bits=4, centered=True, clip_scale: float = 1.0):
        super().__init__(bits, centered)
        self.clip_scale = clip_scale

    @torch.no_grad()
    def _quantize_core(self, x: torch.Tensor):
        xd = x.detach()
        scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * torch.sqrt(torch.mean(xd**2, dim=-1, keepdim=True)) + 1e-8
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(xd, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
            mask = (torch.abs(xd) <= scale * self.clip_scale).float()
        else:
            neg_scale = -scale * (self.n_levels - 2)
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(xd, neg_scale, scale)
            xq = torch.round(x_clip / step) * step
            mask = ((neg_scale * self.clip_scale <= xd) & (xd <= scale * self.clip_scale)).float()
        return xq, mask

    @torch.no_grad()
    def hard_quantize(self, x: torch.Tensor) -> torch.Tensor:
        xq, mask = self._quantize_core(x)
        return x.detach() * (1.0 - mask) + xq * mask

    def forward(self, x):
        xq, mask = self._quantize_core(x)
        return x * mask + (xq - x * mask).detach()


def block_rmatmul(x, matrix):
    block_dim = matrix.shape[0]
    x_shape = x.shape
    x_reshaped = x.reshape(*x_shape[:-1], -1, block_dim)
    x_had = x_reshaped @ matrix.to(x.device).to(x.dtype)
    return x_had.reshape(*x_shape)


if HAS_HADAMARD:
    class HadamardMSEQuantizer(MSEQuantizer):
        aux_matrix = hadamard_transform(torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2))
else:
    # Placeholder when hadamard is not available
    class HadamardMSEQuantizer(MSEQuantizer):
        def __init__(self, *args, **kwargs):
            raise ImportError("HadamardMSEQuantizer requires fast_hadamard_transform. Install it with: pip install fast-hadamard-transform")

    def __init__(self, bits=4, centered=True, clip_scale: float = 1.0):
        super().__init__(bits, centered, clip_scale)

    @torch.no_grad()
    def hard_quantize(self, x: torch.Tensor) -> torch.Tensor:
        x_had = block_rmatmul(x, self.aux_matrix)
        hard_had = super().hard_quantize(x_had)
        return block_rmatmul(hard_had, self.aux_matrix.T)

    def forward(self, x):
        x_had = block_rmatmul(x, self.aux_matrix)
        output = super().forward(x_had)
        output = block_rmatmul(output, self.aux_matrix.T)
        return output
    
    @torch.no_grad()
    def quantize_update_like(self, update: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        upd_h = block_rmatmul(update, self.aux_matrix)
        ref_h = block_rmatmul(ref, self.aux_matrix)

        refd = ref_h.detach()
        scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * torch.sqrt(torch.mean(refd**2, dim=-1, keepdim=True)) + 1e-8

        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            upd_clip = torch.clamp(upd_h, -scale, scale)
            uq_h = torch.round(upd_clip / step + 0.5) * step - 0.5 * step
        else:
            neg_scale = -scale * (self.n_levels - 2)
            step = 2 * scale / self.n_levels
            upd_clip = torch.clamp(upd_h, neg_scale, scale)
            uq_h = torch.round(upd_clip / step) * step

        return block_rmatmul(uq_h, self.aux_matrix.T)


# Hadamard-based quantizers (require fast_hadamard_transform)
if HAS_HADAMARD:
    class HalfHadamardTrustQuantizer(BaseQuantizer):
        aux_matrix = hadamard_transform(torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2))

    def __init__(self, bits=4, trust=None):
        super().__init__(bits, True)
        self.matrix = None
        if trust is None:
            trust = OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1)
        self.trust = trust

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x_had, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
            mask = (torch.abs(xq - x_had) <= std * self.trust).float()

        grad_flow_output = x_had * mask
        return grad_flow_output + (xq - grad_flow_output).detach()


class TrustQuantizer(BaseQuantizer):
    def __init__(self, bits=4, centered=True, trust=None):
        super().__init__(bits, centered)

        # in terms of std
        if trust is None:
            trust = OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1)
        self.trust = trust

    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
        scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
        else:
            neg_scale = -scale * (self.n_levels - 2)
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(x, neg_scale, scale)
            xq = torch.round(x_clip / step) * step

        mask = (torch.abs(xq - x) <= std * self.trust).float()
        return x * mask + (xq - x * mask).detach()


class HadamardTrustQuantizer(TrustQuantizer):
    aux_matrix = hadamard_transform(torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2))

    def __init__(self, bits=4, trust=None):
        super().__init__(bits, True, trust)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
            if self.centered:
                step = 2 * scale / (self.n_levels - 1)
                x_clip = torch.clamp(x_had, -scale, scale)
                xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
            else:
                neg_scale = -scale * (self.n_levels - 2)
                step = 2 * scale / self.n_levels
                x_clip = torch.clamp(x_had, neg_scale, scale)
                xq = torch.round(x_clip / step) * step
            mask = (torch.abs(xq - x_had) <= std * self.trust).float()
            xq = xq @ self.matrix.T

        grad_flow_output = (x_had * mask) @ self.matrix.T

        return grad_flow_output + (xq - grad_flow_output).detach()


class GaussianSTEQuantizer(BaseQuantizer):
    def __init__(self, bits=4):
        super().__init__(bits)
        self.register_buffer("levels", self._compute_gaussian_levels())

    def _compute_gaussian_levels(self):
        levels = np.linspace(-3, 3, self.n_levels)
        boundaries = np.zeros(self.n_levels + 1)

        for _ in range(20):
            boundaries[1:-1] = (levels[1:] + levels[:-1]) / 2
            boundaries[0] = -float("inf")
            boundaries[-1] = float("inf")

            new_levels = []
            for i in range(self.n_levels):
                b_left, b_right = boundaries[i], boundaries[i + 1]

                def f(x):
                    return x * norm.pdf(x)

                integral_num = integrate.quad(f, b_left, b_right)[0]
                integral_den = integrate.quad(norm.pdf, b_left, b_right)[0]
                if integral_den > 1e-10:
                    new_levels.append(integral_num / integral_den)
                else:
                    new_levels.append(levels[i])
            levels = np.array(new_levels)
        return torch.tensor(levels, dtype=torch.float32)

    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-8
        x_norm = x / std
        expanded_input = x_norm.unsqueeze(-1)
        distances = torch.abs(expanded_input - self.levels)
        indices = torch.argmin(distances, dim=-1)
        xq_norm = self.levels[indices]
        xq = xq_norm * std

        return x + (xq - x).detach()


class GaussianClipQuantizer(GaussianSTEQuantizer):
    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-8
        x_norm = x / std
        expanded_input = x_norm.unsqueeze(-1)
        distances = torch.abs(expanded_input - self.levels)
        indices = torch.argmin(distances, dim=-1)
        xq_norm = self.levels[indices]
        xq = xq_norm * std

        mask = (x_norm.abs() <= self.levels[-1]).float()
        return x * mask + (xq - x * mask).detach()


class GaussianTrustQuantizer(GaussianSTEQuantizer):
    def __init__(self, bits=4, trust=None):
        super().__init__(bits)
        if trust is None:
            trust = (self.levels[-1] - self.levels[-2]) / 2
        self.trust = trust

    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-8
        x_norm = x / std
        expanded_input = x_norm.unsqueeze(-1)
        distances = torch.abs(expanded_input - self.levels)
        indices = torch.argmin(distances, dim=-1)
        xq_norm = self.levels[indices]
        xq = xq_norm * std

        mask = (torch.abs(xq - x) <= std * self.trust).float()
        return x * mask + (xq - x * mask).detach()


class HalfHadamardGaussianClipQuantizer(GaussianClipQuantizer):
    aux_matrix = hadamard_transform(torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2))

    def __init__(self, bits=4):
        super().__init__(bits)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            x_norm = x_had / std
            expanded_input = x_norm.unsqueeze(-1)
            distances = torch.abs(expanded_input - self.levels)
            indices = torch.argmin(distances, dim=-1)
            xq_norm = self.levels[indices]
            xq = xq_norm * std

            mask = (x_norm.abs() <= self.levels[-1]).float()

        grad_flow_output = x_had * mask

        return grad_flow_output + (xq - grad_flow_output).detach()


class HadamardGaussianClipQuantizer(GaussianClipQuantizer):
    aux_matrix = hadamard_transform(torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2))

    def __init__(self, bits=4):
        super().__init__(bits)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            x_norm = x_had / std
            expanded_input = x_norm.unsqueeze(-1)
            distances = torch.abs(expanded_input - self.levels)
            indices = torch.argmin(distances, dim=-1)
            xq_norm = self.levels[indices]
            xq = xq_norm * std

            xq = xq @ self.matrix.T
            mask = (x_norm.abs() <= self.levels[-1]).float()

        grad_flow_output = (x_had * mask) @ self.matrix.T

        return grad_flow_output + (xq - grad_flow_output).detach()


class HalfHadamardGaussianTrustQuantizer(GaussianTrustQuantizer):
    aux_matrix = hadamard_transform(torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2))

    def __init__(self, bits=4, trust=None):
        super().__init__(bits, trust)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            x_norm = x_had / std
            expanded_input = x_norm.unsqueeze(-1)
            distances = torch.abs(expanded_input - self.levels)
            indices = torch.argmin(distances, dim=-1)
            xq_norm = self.levels[indices]
            xq = xq_norm * std

            mask = (torch.abs(xq - x_had) <= std * self.trust).float()

        grad_flow_output = x_had * mask
        return grad_flow_output + (xq - grad_flow_output).detach()


class HadamardGaussianTrustQuantizer(GaussianTrustQuantizer):
    aux_matrix = hadamard_transform(torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2))

    def __init__(self, bits=4, trust=None):
        super().__init__(bits, trust)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            x_norm = x_had / std
            expanded_input = x_norm.unsqueeze(-1)
            distances = torch.abs(expanded_input - self.levels)
            indices = torch.argmin(distances, dim=-1)
            xq_norm = self.levels[indices]
            xq = xq_norm * std

            mask = (torch.abs(xq - x_had) <= std * self.trust).float()
            xq = xq @ self.matrix.T

        grad_flow_output = (x_had * mask) @ self.matrix.T

        return grad_flow_output + (xq - grad_flow_output).detach()


class Q99IntQuantizer(BaseQuantizer):
    def __init__(
            self, 
            bits=4, 
            centered=True, 
            levels: Optional[float] = None, 
            percentile: float = 99.0, 
            init_scale: float = None,
            tau: float = 0.5,
            calibrate_once: bool = False
        ):
        super().__init__(bits=bits, centered=centered)
        if levels is None:
            levels = 2**(bits - 1)
        self.levels = float(levels)
        self.percentile = percentile
        self.tau = tau
        # print("TAU SET TO: ", self.tau)
        self.calibrate_once = calibrate_once
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("did_calibrate", torch.tensor(False, dtype=torch.bool))
        
    def calibrate_scale(self, x: torch.Tensor):
        computed_scale = self._compute_scale_from_data(x)
        self.register_buffer("scale", torch.tensor(computed_scale, dtype=torch.float32))
        return computed_scale
    
    @dynamo.disable
    @torch.no_grad()
    def calibrate_scale_(self, x: torch.Tensor):
        s = self._compute_scale_from_data(x)
        self.scale.copy_(s)
        self.did_calibrate.fill_(True)

    def _compute_scale_from_data(self, x: torch.Tensor) -> float:
        abs_x = x.abs().flatten()
        abs_x = abs_x[torch.isfinite(abs_x)]
        if abs_x.numel() == 0:
            return self.scale
        
        k = max(1, int(self.percentile / 100 * abs_x.numel()))
        k = min(k, abs_x.numel())
        
        p_val = abs_x.sort().values[k - 1]
        p_val = torch.clamp(p_val, min=1e-6)
        scale = (self.levels / p_val).to(torch.float32)
        return scale

    def forward(self, x):
        if self.calibrate_once and not bool(self.did_calibrate):
            self.calibrate_scale_(x.detach())

        xs = x * self.scale
        xi = self.round_at(xs, self.tau).clamp(-self.levels, self.levels)
        xq = xi / self.scale
        return x + (xq - x).detach()
    
    @torch.no_grad()
    def round_at(self, x: torch.Tensor, tau: float = 0.1):
        sign = torch.sign(x)
        ax = x.abs()
        flo = torch.floor(ax)
        frac = ax - flo
        ri = flo + (frac >= tau).to(ax.dtype)
        return sign * ri
    
    @torch.no_grad()
    def hard_quantize(self, x: torch.Tensor) -> torch.Tensor:
        xs = x * self.scale
        # print("XS: ", xs.mean())
        xi = self.round_at(xs, self.tau).clamp(-self.levels, self.levels)
        return xi / self.scale
    
    @torch.no_grad()
    def ternary_quantize(self, x: torch.Tensor) -> torch.Tensor:
        xs = x * self.scale
        # print("XS: ", xs.mean())
        xi = self.round_at(xs, self.tau).clamp(-self.levels, self.levels)
        xi = xi.clamp(-1, 1)
        return xi / self.scale
    
    @torch.no_grad()
    def ternary_quantize_conditional(self, x: torch.Tensor) -> torch.Tensor:
        xs = x * self.scale
        xi = self.round_at(xs, self.tau).clamp(-self.levels, self.levels)

        xi = xi.clamp(-1, 1)

        xi[(xi != 1) & (xi != -1)] = 0

        return xi / self.scale

class Q99FP4Quantizer(BaseQuantizer):

    def __init__(
        self,
        bits: int = 4,
        centered: bool = True,
        percentile: float = 99.0,
        init_scale: Optional[float] = None,
        calibrate_once: bool = False,
        recalibrate_interval: int = 0,  # NEW: 0 = never recalibrate, N = recalibrate every N calls
        codebook: Optional[torch.Tensor] = None,
    ):
        super().__init__(bits=bits, centered=centered)

        if bits != 4:
            raise ValueError("Q99FP4Quantizer currently expects bits=4.")

        self.percentile = percentile
        self.calibrate_once = calibrate_once
        self.recalibrate_interval = recalibrate_interval  # NEW

        if codebook is None:
            codebook = torch.tensor([
                -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.75,
                0.0,
                0.5,  0.75,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
            ], dtype=torch.float32)

        if codebook.numel() != 2 ** bits:
            raise ValueError(f"FP4 codebook must have exactly {2 ** bits} entries.")

        codebook = torch.sort(codebook.flatten().to(torch.float32)).values
        self.register_buffer("codebook", codebook)

        scale_value = 1.0 if init_scale is None else float(init_scale)
        self.register_buffer("scale", torch.tensor(scale_value, dtype=torch.float32))
        self.register_buffer("did_calibrate", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("quantize_call_count", torch.tensor(0, dtype=torch.long))  # NEW: track calls

    def calibrate_scale(self, x: torch.Tensor):
        computed_scale = self._compute_scale_from_data(x)
        self.register_buffer("scale", torch.tensor(computed_scale, dtype=torch.float32))
        return computed_scale

    @dynamo.disable
    @torch.no_grad()
    def calibrate_scale_(self, x: torch.Tensor):
        s = self._compute_scale_from_data(x)
        self.scale.copy_(s)
        self.did_calibrate.fill_(True)

    def _compute_scale_from_data(self, x: torch.Tensor) -> torch.Tensor:
        abs_x = x.abs().flatten()
        abs_x = abs_x[torch.isfinite(abs_x)]

        if abs_x.numel() == 0:
            return self.scale

        k = max(1, int(self.percentile / 100.0 * abs_x.numel()))
        k = min(k, abs_x.numel())

        p_val = abs_x.sort().values[k - 1]
        p_val = torch.clamp(p_val, min=1e-6)

        # Match the percentile value to the largest magnitude representable value
        max_code = self.codebook.abs().max().clamp_min(1e-6)
        scale = (max_code / p_val).to(torch.float32)
        return scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrate_once and not bool(self.did_calibrate):
            self.calibrate_scale_(x.detach())

        xq = self.hard_quantize(x)
        return x + (xq - x).detach()

    @torch.no_grad()
    def _nearest_codebook(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Quantize scaled tensor xs to nearest FP4 codebook value.
        """
        shape = xs.shape
        flat = xs.reshape(-1).to(torch.float32)

        # Compute distances to all codebook points
        # [N, 1] - [1, K] -> [N, K]
        dist = (flat[:, None] - self.codebook[None, :]).abs()
        idx = dist.argmin(dim=1)

        q = self.codebook[idx]
        return q.view(shape).to(xs.dtype)

    @torch.no_grad()
    def hard_quantize(self, x: torch.Tensor) -> torch.Tensor:
        # NEW: Periodic recalibration logic
        if self.recalibrate_interval > 0:
            self.quantize_call_count += 1
            if self.quantize_call_count % self.recalibrate_interval == 0:
                self.calibrate_scale_(x)

        xs = x * self.scale
        xq_scaled = self._nearest_codebook(xs)
        return xq_scaled / self.scale

    @torch.no_grad()
    def ternary_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Collapse to {-a, 0, +a} where a corresponds to the smallest nonzero
        magnitude available in the current FP4 codebook after rescaling.
        """
        xs = x * self.scale

        nonzero_codes = self.codebook[self.codebook != 0]
        if nonzero_codes.numel() == 0:
            return torch.zeros_like(x)

        a = nonzero_codes.abs().min()
        pos = a
        neg = -a

        out = torch.zeros_like(xs)
        out[xs > 0] = pos
        out[xs < 0] = neg
        return out / self.scale

    @torch.no_grad()
    def ternary_quantize_conditional(self, x: torch.Tensor) -> torch.Tensor:
        """
        First quantize to FP4, then keep only {-a, 0, +a}, zeroing everything else.
        """
        xs = x * self.scale
        xq_scaled = self._nearest_codebook(xs)

        nonzero_codes = self.codebook[self.codebook != 0]
        if nonzero_codes.numel() == 0:
            return torch.zeros_like(x)

        a = nonzero_codes.abs().min()

        out = torch.zeros_like(xq_scaled)
        out[xq_scaled == a] = a
        out[xq_scaled == -a] = -a
        return out / self.scale
    
class SymmetricIntQuantizer(BaseQuantizer):
    def __init__(
            self,
            bits: int = 4,
            centered=True, 
            percentile: float = 99.0,
            init_scale: Optional[float] = None,
            tau: float = 0.5,
            calibrate_once: bool = False,
            eps: float = 1e-6,
        ):
        super().__init__(bits=bits, centered=centered)
        assert bits >= 2
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))          # e.g. -8
        self.qmax = (2 ** (bits - 1)) - 1       # e.g.  7
        self.percentile = float(percentile)
        self.tau = float(tau)
        self.calibrate_once = bool(calibrate_once)
        self.eps = float(eps)

        s0 = 1.0 if init_scale is None else float(init_scale)
        self.register_buffer("scale", torch.tensor(s0, dtype=torch.float32))  # multiplier s
        self.register_buffer("did_calibrate", torch.tensor(False, dtype=torch.bool))

        
    @torch.no_grad()
    def round_at(self, x: torch.Tensor, tau: Optional[float] = None) -> torch.Tensor:
        # stochastic-free threshold rounding: floor + (frac>=tau)
        if tau is None:
            tau = self.tau
        sign = torch.sign(x)
        ax = x.abs()
        flo = torch.floor(ax)
        frac = ax - flo
        ri = flo + (frac >= tau).to(ax.dtype)
        return sign * ri

    @torch.no_grad()
    def _compute_alpha(self, x: torch.Tensor) -> torch.Tensor:
        # alpha = percentile(|x|) (robust symmetric range)
        ax = x.detach().abs().flatten()
        ax = ax[torch.isfinite(ax)]
        if ax.numel() == 0:
            # fallback: infer alpha from current scale if possible
            # alpha = qmax / s
            s = self.scale.clamp_min(self.eps)
            return (torch.tensor(float(self.qmax), device=s.device, dtype=s.dtype) / s)

        k = max(1, int(self.percentile / 100.0 * ax.numel()))
        k = min(k, ax.numel())
        p = ax.sort().values[k - 1]
        return p.clamp_min(self.eps)

    @dynamo.disable
    @torch.no_grad()
    def calibrate_scale_(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sets scale s = qmax / alpha, where alpha = percentile(|x|).
        Returns s (tensor).
        """
        alpha = self._compute_alpha(x)
        s = (float(self.qmax) / alpha).to(torch.float32)
        self.scale.copy_(s)
        self.did_calibrate.fill_(True)
        return s

    @torch.no_grad()
    def hard_quantize(self, x: torch.Tensor) -> torch.Tensor:
        s = self.scale.to(dtype=torch.float32, device=x.device).clamp_min(self.eps)
        xs = x.to(torch.float32) * s
        xi = self.round_at(xs, self.tau).clamp(self.qmin, self.qmax)
        return (xi / s).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrate_once and not bool(self.did_calibrate):
            self.calibrate_scale_(x)

        s = self.scale.to(dtype=torch.float32, device=x.device).clamp_min(self.eps)
        xs = x.to(torch.float32) * s
        xi = self.round_at(xs, self.tau).clamp(self.qmin, self.qmax)
        xq = (xi / s).to(dtype=x.dtype)
        return x + (xq - x).detach()


QUANTIZER_CLASSES = {
    "NoQuantizer": NoQuantizer,
    "AbsMaxQuantizer": AbsMaxQuantizer,
    "MSEQuantizer": MSEQuantizer,
    "TrustQuantizer": TrustQuantizer,
    "GaussianSTEQuantizer": GaussianSTEQuantizer,
    "GaussianClipQuantizer": GaussianClipQuantizer,
    "GaussianTrustQuantizer": GaussianTrustQuantizer,
    "Q99IntQuantizer": Q99IntQuantizer,
    "Q99FP4Quantizer": Q99FP4Quantizer,
    "SymmetricIntQuantizer": SymmetricIntQuantizer,
}

# Add Hadamard quantizers only if fast_hadamard_transform is available
if HAS_HADAMARD:
    QUANTIZER_CLASSES.update({
        "HadamardMSEQuantizer": HadamardMSEQuantizer,
        "HalfHadamardTrustQuantizer": HalfHadamardTrustQuantizer,
        "HadamardTrustQuantizer": HadamardTrustQuantizer,
        "HadamardGaussianClipQuantizer": HadamardGaussianClipQuantizer,
        "HalfHadamardGaussianTrustQuantizer": HalfHadamardGaussianTrustQuantizer,
        "HadamardGaussianTrustQuantizer": HadamardGaussianTrustQuantizer,
    })


class QuantizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, weight_quantizer=None, activation_quantizer=None, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        if weight_quantizer is None:
            weight_quantizer = NoQuantizer()
        if activation_quantizer is None:
            activation_quantizer = NoQuantizer()
        
        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer
        setattr(self.weight, "quantizer", weight_quantizer)
        
        with torch.no_grad():
            if hasattr(weight_quantizer, "calibrate_scale_"):
                weight_quantizer.calibrate_scale_(self.weight)
            elif hasattr(weight_quantizer, "calibrate_scale"):
                weight_quantizer.calibrate_scale(self.weight)
            print("WEIGHT QUANTIZER", weight_quantizer)
            print("ACT QUANTIZER", activation_quantizer)
            self.weight.copy_(weight_quantizer.hard_quantize(self.weight))
            if hasattr(weight_quantizer, "did_calibrate"):
                weight_quantizer.did_calibrate.fill_(True)

    def forward(self, x):
        x = self.activation_quantizer(x)
        w = self.weight_quantizer(self.weight)
        return F.linear(x, w, self.bias)
