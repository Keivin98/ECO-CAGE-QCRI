'''
    This module contains the final version of the vanilla "Stateless Adam" optimizer. 
    While the performance of this optimizer matches that of AdamW, the best learning rate doesn't match to that of AdamW.
    This compatibility will be addressed in another variant of this optimizer.
'''

import copy
import torch
from optim.qcri.grouping import group_parameters_llm_2d1d_only

class _Adam0(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, eps=1e-8, weight_decay=0, n_attention_heads=None, is_vector=False, enter_once=True):
        self.enter_once = enter_once
        defaults = dict(
            lr=lr, beta1=beta1, eps=eps, weight_decay=weight_decay,
            n_attention_heads=n_attention_heads, is_vector=is_vector, enter_once=enter_once
        )
        assert n_attention_heads != None 
        super().__init__(params, defaults)

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
                    denom = self.update_weights(p, lr, beta1, eps, step, n_attention_heads, is_vector=is_vector)
                else:                       # if grad is associated with the linear layer's bias
                    denom = self.update_biases(p, lr, beta1, eps, step, n_attention_heads, is_vector=is_vector)
                
                theta_tilde = p.clone()
                
                if hasattr(p, "quantizer") and p.quantizer is not None:
                    if self.enter_once:
                        print("entered quantization")
                        self.enter_once = False
                    theta_hat = p.quantizer.hard_quantize(theta_tilde)
                else:
                    theta_hat = theta_tilde
                    
                e_next = theta_tilde - theta_hat

                # ECO momentum correction
                bias_c1 = 1.0 - beta1 ** step
                coeff = (bias_c1 / lr) * (1.0 - 1.0 / beta1)
                
                
                # IT WAS: / (1 - beta1). BUT BE CAREFUL HERE, THIS OPTIMIZER DOESNT MULTIPLY M BY (1 - beta1) 
                correction = coeff * denom * e_next 
                grad.add_(correction.squeeze(0) if correction.ndim > grad.ndim else correction)

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
        if is_vector:
            p.data.addcdiv_(m[0], denom[0], value=-lr)
        else:
            p.data.addcdiv_(m, denom, value=-lr)  # no need to divide by beta1_sq
        
        return denom

    @torch.no_grad()
    def update_biases(self, p, lr, beta1, eps, step, n_attention_heads, is_vector):
        if is_vector:
            raise NotImplementedError("Layer norm biases are not implemented")
        m = p.grad.data[:, None] / (1 - beta1 ** step)
        variance_map = self.compute_attention_variance_vectorized_efficient(m, beta1, n_attention_heads)
        denom = variance_map.sqrt() + eps        
        p.data.addcdiv_(m[:, 0], denom[:, 0], value=-lr)
        return denom


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
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none=False):
        for opt in self.optimizers:
            opt.zero_grad()

# Just for the sake of consistency with torch interface, otherwise we don't really implement Adam0 with L2 weight decay
class Adam0(AdamW0):
    pass