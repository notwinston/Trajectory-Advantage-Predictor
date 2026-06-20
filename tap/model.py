"""SmallTAP — the small Trajectory Advantage Predictor model.

Architecture (spec section "SMALL TAP MODEL"):

* candidate embedding (256) -> Linear -> 64
* gradient sketch (64)      -> Linear -> 32
* candidate numeric MLP     -> 64
* state numeric + fingerprint MLP -> 32
* history records (256+64+7 = 327 each) projected -> 64, plus a relative-step
  embedding added to the history keys
* ONE 4-head cross-attention from the candidate to its <=4 history records
* final 2-layer MLP (hidden 128) -> 1 scalar predicted_utility_points

Target: < 250,000 trainable parameters.

torch is imported lazily *inside* the class factory so this module imports (and
``py_compile``/``unittest`` run) on CPU-only ARM64 without torch — required for
the ``TAP_NO_TORCH`` sklearn fallback path. ``SmallTAP(...)`` is a thin factory
returning an instance of the lazily-built ``nn.Module`` subclass, so
``SmallTAP().num_params()`` works whenever torch is present.
"""

from __future__ import annotations

import os

from tap.featurize import HISTORY_RECORD_DIM
from tap.schema import (
    CANDIDATE_EMBEDDING_DIM,
    GRADIENT_SKETCH_DIM,
    POLICY_FINGERPRINT_DIM,
)
from tap.dataset import CANDIDATE_NUMERIC_COLS, STATE_NUMERIC_COLS

# Default block dims so a bare ``SmallTAP()`` builds the real model.
DEFAULT_DIMS = dict(
    cand_emb_dim=CANDIDATE_EMBEDDING_DIM,  # 256
    grad_dim=GRADIENT_SKETCH_DIM,  # 64
    numeric_dim=len(CANDIDATE_NUMERIC_COLS),  # 16
    state_dim=len(STATE_NUMERIC_COLS) + POLICY_FINGERPRINT_DIM,  # 26
    hist_dim=HISTORY_RECORD_DIM,  # 327
    hist_window=4,
    attn_dim=64,
    n_heads=4,
)

_SMALLTAP_CLS = None


def torch_available() -> bool:
    """True when the torch path is enabled (torch installed and TAP_NO_TORCH unset)."""
    if os.environ.get("TAP_NO_TORCH") == "1":
        return False
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _smalltap_class():
    """Build (once) and return the ``nn.Module`` SmallTAP subclass."""
    global _SMALLTAP_CLS
    if _SMALLTAP_CLS is not None:
        return _SMALLTAP_CLS

    import torch
    import torch.nn as nn

    class _SmallTAP(nn.Module):
        def __init__(
            self,
            cand_emb_dim: int = DEFAULT_DIMS["cand_emb_dim"],
            grad_dim: int = DEFAULT_DIMS["grad_dim"],
            numeric_dim: int = DEFAULT_DIMS["numeric_dim"],
            state_dim: int = DEFAULT_DIMS["state_dim"],
            hist_dim: int = DEFAULT_DIMS["hist_dim"],
            hist_window: int = DEFAULT_DIMS["hist_window"],
            attn_dim: int = DEFAULT_DIMS["attn_dim"],
            n_heads: int = DEFAULT_DIMS["n_heads"],
        ):
            super().__init__()
            self.hist_window = hist_window
            self.attn_dim = attn_dim

            # Block projections.
            self.cand_emb_proj = nn.Sequential(nn.Linear(cand_emb_dim, 64), nn.LayerNorm(64))
            self.grad_proj = nn.Sequential(nn.Linear(grad_dim, 32), nn.LayerNorm(32))
            self.numeric_mlp = nn.Sequential(
                nn.Linear(numeric_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.LayerNorm(64)
            )
            self.state_mlp = nn.Sequential(
                nn.Linear(state_dim, 32), nn.ReLU(), nn.Linear(32, 32), nn.LayerNorm(32)
            )
            cand_rep_dim = 64 + 32 + 64 + 32  # 192

            # History projection + relative-step embedding (index 0 = padding).
            self.hist_proj = nn.Linear(hist_dim, attn_dim)
            self.hist_norm = nn.LayerNorm(attn_dim)
            self.rel_age_emb = nn.Embedding(hist_window + 1, attn_dim, padding_idx=0)

            # Candidate -> history cross-attention (one 4-head layer).
            self.query_proj = nn.Linear(cand_rep_dim, attn_dim)
            self.attn = nn.MultiheadAttention(attn_dim, n_heads, batch_first=True)

            # Final 2-layer MLP (hidden 128) -> scalar.
            self.head = nn.Sequential(
                nn.Linear(cand_rep_dim + attn_dim, 128), nn.ReLU(), nn.Linear(128, 1)
            )

        def num_params(self) -> int:
            return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

        def forward(self, batch: dict):
            cand = self.cand_emb_proj(batch["cand_emb"])
            grad = self.grad_proj(batch["grad_sketch"])
            num = self.numeric_mlp(batch["cand_numeric"])
            state_in = torch.cat([batch["state_numeric"], batch["fingerprint"]], dim=1)
            state = self.state_mlp(state_in)
            cand_rep = torch.cat([cand, grad, num, state], dim=1)  # [B, 192]

            hist = batch["history"]  # [B, W, hist_dim]
            hist_keys = self.hist_norm(self.hist_proj(hist))
            hist_keys = hist_keys + self.rel_age_emb(batch["history_rel_age"])  # [B, W, attn]

            query = self.query_proj(cand_rep).unsqueeze(1)  # [B, 1, attn]
            mask = batch["history_mask"]  # [B, W] bool, True = real record
            key_padding = ~mask  # True = ignore (padding)
            fully_masked = key_padding.all(dim=1)
            # Avoid all-(-inf) softmax NaNs for states with no history: temporarily
            # un-mask position 0, then zero those rows' attended output.
            safe_padding = key_padding.clone()
            safe_padding[fully_masked, 0] = False
            attended, _ = self.attn(query, hist_keys, hist_keys, key_padding_mask=safe_padding)
            attended = attended.squeeze(1)  # [B, attn]
            attended = attended.masked_fill(fully_masked.unsqueeze(1), 0.0)

            out = self.head(torch.cat([cand_rep, attended], dim=1))  # [B, 1]
            return out.squeeze(1)

    _SMALLTAP_CLS = _SmallTAP
    return _SMALLTAP_CLS


def SmallTAP(*args, **kwargs):
    """Construct a SmallTAP instance (lazily importing torch)."""
    return _smalltap_class()(*args, **kwargs)
