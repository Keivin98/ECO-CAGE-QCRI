import torch
from models.quantization import AbsMaxQuantizer


class LotionRegularizer:
    def __init__(
        self,
        model,
        bitwidth=4,
        beta=0.999,
        device=None,
        coeff=1.0,
    ):
        self.beta = beta
        self.bitwidth = bitwidth
        self.coeff = coeff
        self.state = {}
        self.device = device
        for p in model.parameters():
            if hasattr(p, "quantizer"):
                assert isinstance(p.quantizer, AbsMaxQuantizer), (
                    f"Weight quantizer {p.quantizer} must be AbsMaxQuantizer"
                )
                assert p.quantizer.bits == bitwidth, f"Weight quantizer {p.quantizer} must have bits {bitwidth}"
            if p.requires_grad:
                self.state[p] = torch.zeros_like(p, device=p.device if device is None else device)

    @torch.no_grad()
    def update_fisher(self, params):
        for p in params:
            if p.requires_grad and p.grad is not None:
                v = self.state[p]
                self.state[p].copy_(self.beta * v + (1 - self.beta) * (p.grad.detach() ** 2))

    def rr_variance(self, x) -> torch.Tensor:
        # only for AbsMaxQuantizer
        n_levels = 2**self.bitwidth
        mxval = torch.max(torch.abs(x)).item()
        scale = mxval * 2 / (n_levels)
        z = x / scale
        delta = z - torch.floor(z)
        var = (scale**2) * delta * (1 - delta)
        return var

    # Add the value to the loss before doing the backward pass
    def reg(self, params) -> torch.Tensor:
        # find the index of largest parameter
        reg_terms = []
        for i, p in enumerate(params):
            if not p.requires_grad or not hasattr(p, "quantizer"):
                continue
            var_rr = self.rr_variance(p)
            Gdiag = self.state[p].detach()
            reg_terms.append(0.5 * self.coeff * (Gdiag * var_rr).sum())
        return torch.stack(reg_terms).sum()
