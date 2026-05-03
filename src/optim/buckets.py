"""Per-bucket grouping of model parameters for layer-wise dynamics logging.

Five buckets motivated by ECO-0's per-head attention factorization:
  - embedding   : token embeddings, positional embeddings, lm_head (tied)
  - attn.qkv    : q/k/v projections (head-structured)
  - attn.out    : attention output projection (head-mixed)
  - mlp.up      : SwiGLU gate (w1) and up (w2) projections
  - mlp.down    : MLP down projection
  - other       : everything else (layer norms, etc.) — usually tiny
"""

BUCKETS = ['embedding', 'attn.qkv', 'attn.out', 'mlp.up', 'mlp.down', 'other']


def get_bucket(param_name: str) -> str:
    n = param_name
    if 'wte' in n or 'wpe' in n or 'tok_embeddings' in n or 'lm_head' in n:
        return 'embedding'
    if 'q_proj' in n or 'k_proj' in n or 'v_proj' in n:
        return 'attn.qkv'
    if 'attn.c_proj' in n or 'attn.o_proj' in n:
        return 'attn.out'
    if 'mlp.w1' in n or 'mlp.w2' in n or 'mlp.gate_proj' in n or 'mlp.up_proj' in n:
        return 'mlp.up'
    if 'mlp.c_proj' in n or 'mlp.down_proj' in n:
        return 'mlp.down'
    return 'other'


def build_param_to_bucket(model) -> dict:
    """Map id(param) -> bucket name, given a model. Use after model is built."""
    return {id(p): get_bucket(name) for name, p in model.named_parameters()}


def build_param_to_name(model) -> dict:
    """Map id(param) -> name, useful for debug/quantizer scale lookup."""
    return {id(p): name for name, p in model.named_parameters()}
