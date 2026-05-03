import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# -----------------------------
# Helpers
# -----------------------------

_FUSED_QKV_RE = re.compile(
    r"(^|\.)(query_key_value|c_attn|w_?qkv|wqkvlinear|in_proj|linear_qkv|qkv_proj|qkv)(_proj)?(\.|$)"
)
_MOE_RE = re.compile(r"(^|\.)(block_sparse_moe|moe|experts?)(\.|$)")
_HEAD_RE = re.compile(r"(^|\.)(lm_head|embed_out)(\.|$)")
_EMBED_RE = re.compile(r"(^|\.)(embed_tokens|word_embeddings|embeddings|wte)(\.|$)")

# For "separate Q/K/V" attention modules (Llama/Mistral/Qwen2/T5-style, etc.)
_Q_RE = re.compile(r"(^|\.)(q_proj|w_q|wq|to_q|query|q)(\.|$)")
_K_RE = re.compile(r"(^|\.)(k_proj|w_k|wk|to_k|key|k)(\.|$)")
_V_RE = re.compile(r"(^|\.)(v_proj|w_v|wv|to_v|value|v)(\.|$)")


def _unwrap_wrappers(m: nn.Module) -> nn.Module:
    """
    Unwrap only actual wrappers (DDP/FSDP/torch.compile), NOT Hugging Face base_model_prefix (.model).
    """
    changed = True
    while changed:
        changed = False
        for attr in ("module", "_fsdp_wrapped_module", "_orig_mod"):
            if hasattr(m, attr):
                m = getattr(m, attr)
                changed = True
    return m


@dataclass(frozen=True)
class AttnMeta:
    n_q_heads: int
    n_kv_heads: Optional[int]


def _infer_attention_heads(m: nn.Module) -> AttnMeta:
    cfg = getattr(m, "config", None)

    # Query heads
    n_q = None
    if cfg is not None:
        for k in ("num_attention_heads", "n_head", "n_heads", "num_heads"):
            v = getattr(cfg, k, None)
            if isinstance(v, int) and v > 0:
                n_q = int(v)
                break
    if n_q is None:
        for _, mod in m.named_modules():
            for k in ("num_attention_heads", "num_heads", "n_heads", "n_head"):
                v = getattr(mod, k, None)
                if isinstance(v, int) and v > 0:
                    n_q = int(v)
                    break
            if n_q is not None:
                break
    if n_q is None:
        raise ValueError("Could not infer num_attention_heads / n_head from config/modules.")

    # KV heads (optional, for GQA/MQA models)
    n_kv = None
    if cfg is not None:
        for k in ("num_key_value_heads", "n_kv_head", "n_kv_heads", "num_kv_heads"):
            v = getattr(cfg, k, None)
            if isinstance(v, int) and v > 0:
                n_kv = int(v)
                break
    if n_kv is None:
        for _, mod in m.named_modules():
            for k in ("num_key_value_heads", "n_kv_head", "n_kv_heads", "num_kv_heads"):
                v = getattr(mod, k, None)
                if isinstance(v, int) and v > 0:
                    n_kv = int(v)
                    break
            if n_kv is not None:
                break

    return AttnMeta(n_q_heads=n_q, n_kv_heads=n_kv)


def _format_numel(n: int) -> str:
    # readable sizes
    if n >= 10**9: return f"{n/1e9:.3f}B"
    if n >= 10**6: return f"{n/1e6:.3f}M"
    if n >= 10**3: return f"{n/1e3:.3f}K"
    return str(n)


# -----------------------------
# Main grouping function
# -----------------------------

def group_parameters_llm_2d1d_only(
    model: nn.Module,
    *,
    verbose: bool = False,
    allow_fallback_for_unsupported: bool = False,
) -> List[Dict[str, Any]]:
    """
    Properties:
      - Rejects ANY trainable parameter with ndim not in {1,2}.
      - Rejects fused-QKV attention (query_key_value / c_attn / Wqkv / etc).
      - If it returns: every unique trainable param is assigned to exactly one group.
      - Unknown/unsupported trainable params cause hard failure.
      - If allow_fallback_for_unsupported=True, unsupported params are put into a fallback group
        (rule='fallback') instead of raising.
      - Biases on nn.Linear are grouped with their owning linear weights (weight then bias order),
        so optimizers that rely on immediate weight->bias updates can do so safely.
      - verbose=True: prints audit log including inferred attention heads (if attention params exist).
    """
    m = _unwrap_wrappers(model)

    # Collect all trainable params (unique by id) for coverage check
    trainable: List[Tuple[str, torch.nn.Parameter]] = []
    seen = set()
    for name, p in m.named_parameters(recurse=True):
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        trainable.append((name, p))

    # Collect parameter aliases (including tied-weight aliases when available).
    aliases_by_id: Dict[int, List[str]] = {}
    try:
        named_for_aliases = m.named_parameters(recurse=True, remove_duplicate=False)
    except TypeError:
        named_for_aliases = m.named_parameters(recurse=True)
    for alias_name, p in named_for_aliases:
        if not p.requires_grad:
            continue
        aliases_by_id.setdefault(id(p), []).append(alias_name)

    tied_head_embedding: List[Tuple[List[str], List[str]]] = []
    for pid, aliases in aliases_by_id.items():
        head_aliases = sorted({a for a in aliases if _HEAD_RE.search(a.lower())})
        embed_aliases = sorted({a for a in aliases if _EMBED_RE.search(a.lower())})
        if head_aliases and embed_aliases:
            tied_head_embedding.append((embed_aliases, head_aliases))

    # Unsupported parameter candidates for optional fallback.
    unsupported = [(n, p) for (n, p) in trainable if p.ndim not in (1, 2)]
    unsupported_ids = {id(p) for _, p in unsupported}
    if unsupported and not allow_fallback_for_unsupported:
        preview = "\n".join([f"  - {n} | shape={tuple(p.shape)} | ndim={p.ndim}" for n, p in unsupported[:50]])
        more = "" if len(unsupported) <= 50 else f"\n  ... (+{len(unsupported)-50} more)"
        raise ValueError(
            "Unsupported parameter dimensionality detected. This optimizer supports only 1D/2D tensors.\n"
            "This commonly happens for MoE models where expert weights are stored as 3D tensors.\n"
            f"First few unsupported params:\n{preview}{more}"
        )

    # Reject MoE/expert models (unsupported structural assumptions for this optimizer)
    moe_hits = [n for (n, _) in trainable if _MOE_RE.search(n.lower())]
    moe_ids = {id(p) for (n, p) in trainable if _MOE_RE.search(n.lower())}
    if moe_hits and not allow_fallback_for_unsupported:
        preview = "\n".join([f"  - {n}" for n in moe_hits[:50]])
        more = "" if len(moe_hits) <= 50 else f"\n  ... (+{len(moe_hits)-50} more)"
        raise ValueError(
            "Detected MoE/expert parameters. This optimizer currently supports dense transformer blocks only.\n"
            "MoE expert tensors (including 3D expert stacks) are unsupported.\n"
            f"First few MoE params:\n{preview}{more}"
        )

    # Buckets: store (name, param)
    g_embed: List[Tuple[str, torch.nn.Parameter]] = []
    g_vec: List[Tuple[str, torch.nn.Parameter]] = []
    g_attn_q: List[Tuple[str, torch.nn.Parameter]] = []
    g_attn_kv: List[Tuple[str, torch.nn.Parameter]] = []
    g_2d: List[Tuple[str, torch.nn.Parameter]] = []
    g_head: List[Tuple[str, torch.nn.Parameter]] = []
    g_fallback: List[Tuple[str, torch.nn.Parameter]] = []

    handled = set()
    fused_hits: List[str] = []
    unknown_2d_hits: List[Tuple[str, str, Tuple[int, ...]]] = []

    # Walk modules to classify by module type & module name
    for mod_name, mod in m.named_modules():
        lname = mod_name.lower()
        module_is_fused = bool(_FUSED_QKV_RE.search(lname))

        # Hard reject fused QKV if the module name indicates it
        if module_is_fused:
            # record direct params (recurse=False) so we can show meaningful names
            for pn, p in mod.named_parameters(recurse=False):
                if p.requires_grad:
                    full = f"{mod_name}.{pn}" if mod_name else pn
                    fused_hits.append(full)

        for pn, p in mod.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in handled:
                continue
            handled.add(pid)

            full_name = f"{mod_name}.{pn}" if mod_name else pn
            full_lc = full_name.lower()

            if pid in unsupported_ids or pid in moe_ids or module_is_fused:
                g_fallback.append((full_name, p))
                continue

            # Linear params are grouped by owning linear rule.
            # Keep module-local param order (weight then bias) for deterministic weight->bias updates.
            if isinstance(mod, nn.Linear):
                is_head_linear = bool(_HEAD_RE.search(lname) or _HEAD_RE.search(full_lc))
                is_q_linear = bool(_Q_RE.search(lname))
                is_k_linear = bool(_K_RE.search(lname))
                is_v_linear = bool(_V_RE.search(lname))
                is_attn_linear = bool(is_q_linear or is_k_linear or is_v_linear)

                if is_head_linear:
                    g_head.append((full_name, p))
                elif is_attn_linear:
                    if is_q_linear:
                        g_attn_q.append((full_name, p))
                    else:
                        g_attn_kv.append((full_name, p))
                else:
                    g_2d.append((full_name, p))
                continue

            # 1D always vector group
            if p.ndim == 1:
                g_vec.append((full_name, p))
                continue

            # 2D classification
            if isinstance(mod, nn.Embedding):
                g_embed.append((full_name, p))
                continue

            # Head: common HF naming
            if _HEAD_RE.search(full_lc):
                g_head.append((full_name, p))
                continue

            # Non-standard 2D owner -> fail loudly after collection.
            if allow_fallback_for_unsupported:
                g_fallback.append((full_name, p))
            else:
                unknown_2d_hits.append((full_name, mod.__class__.__name__, tuple(p.shape)))

    # Reject fused-QKV models
    if fused_hits and not allow_fallback_for_unsupported:
        preview = "\n".join([f"  - {n}" for n in fused_hits[:50]])
        more = "" if len(fused_hits) <= 50 else f"\n  ... (+{len(fused_hits)-50} more)"
        raise ValueError(
            "Detected fused QKV attention weights (single projection producing Q,K,V together).\n"
            "This optimizer requires *separate* Q/K/V projection layers.\n"
            f"First few fused-QKV params:\n{preview}{more}"
        )

    if unknown_2d_hits:
        preview = "\n".join(
            [
                f"  - {n} | owner={owner} | shape={shape}"
                for (n, owner, shape) in unknown_2d_hits[:50]
            ]
        )
        more = "" if len(unknown_2d_hits) <= 50 else f"\n  ... (+{len(unknown_2d_hits)-50} more)"
        raise ValueError(
            "Strict grouping failed: found trainable 2D parameters with unsupported owning module types.\n"
            "Expected known owners such as nn.Linear / nn.Embedding.\n"
            f"First few unsupported 2D params:\n{preview}{more}"
        )

    # Final safety: ensure perfect coverage (no misses)
    train_ids = {id(p) for _, p in trainable}
    missing = train_ids - handled
    if missing:
        # This should never happen; if it does, print names to debug
        miss_names = [n for (n, p) in trainable if id(p) in missing]
        preview = "\n".join([f"  - {n}" for n in miss_names[:50]])
        more = "" if len(miss_names) <= 50 else f"\n  ... (+{len(miss_names)-50} more)"
        msg = f"BUG: grouping missed trainable params:\n{preview}{more}"
        raise RuntimeError(msg)

    # Infer attention heads only if we actually have attention params
    attn_meta: Optional[AttnMeta] = None
    if g_attn_q or g_attn_kv:
        attn_meta = _infer_attention_heads(m)

    # Build optimizer param_groups (strip names)
    groups: List[Dict[str, Any]] = []
    if g_embed:
        groups.append({"params": [p for _, p in g_embed], "rule": "adam0"})
    if g_vec:
        groups.append({"params": [p for _, p in g_vec], "rule": "adam0", "is_vector": True})
    if g_attn_q:
        groups.append(
            {"params": [p for _, p in g_attn_q], "rule": "adam0_att", "n_attention_heads": attn_meta.n_q_heads}
        )
    if g_attn_kv:
        kv_heads = attn_meta.n_kv_heads if attn_meta.n_kv_heads is not None else attn_meta.n_q_heads
        groups.append(
            {"params": [p for _, p in g_attn_kv], "rule": "adam0_att", "n_attention_heads": kv_heads}
        )
    if g_2d:
        groups.append({"params": [p for _, p in g_2d], "rule": "adam0"})
    if g_head:
        groups.append({"params": [p for _, p in g_head], "rule": "adam0"})
    if g_fallback:
        groups.append({"params": [p for _, p in g_fallback], "rule": "fallback"})

    # Verbose audit log
    if verbose:
        print("[Adam0 grouping]")
        if attn_meta is None:
            print("  inferred_attention_heads: <none> (no attention Q/K/V params detected)")
        else:
            print(
                f"  inferred_attention_heads: n_q={attn_meta.n_q_heads}"
                + (f", n_kv={attn_meta.n_kv_heads}" if attn_meta.n_kv_heads else "")
            )
        if tied_head_embedding:
            print("  tied_lm_head_embeddings: yes")
            for embed_aliases, head_aliases in tied_head_embedding[:20]:
                print(f"    - embedding={embed_aliases[0]} <-> head={head_aliases[0]}")
            if len(tied_head_embedding) > 20:
                print(f"    ... (+{len(tied_head_embedding)-20} more)")
        else:
            print("  tied_lm_head_embeddings: no")

        def _dump(title: str, bucket: List[Tuple[str, torch.nn.Parameter]]):
            if not bucket:
                return
            total = sum(p.numel() for _, p in bucket)
            print(f"  {title}: {len(bucket)} params | total_numel={_format_numel(total)}")
            # Preserve insertion order so verbose output matches actual optimizer update order.
            for n, p in bucket:
                print(f"    - {n} | shape={tuple(p.shape)} | numel={_format_numel(p.numel())}")

        _dump("EMBED", g_embed)
        _dump("VECTOR(1D)", g_vec)
        _dump("ATTN_Q", g_attn_q)
        _dump("ATTN_KV", g_attn_kv)
        _dump("LINEAR_2D", g_2d)
        _dump("HEAD", g_head)
        _dump("FALLBACK", g_fallback)

    return groups