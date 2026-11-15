"""One place that answers: 'given source and target characters, which joints drive which?'
Modes: oracle (label-derived M_gt), hard (Hungarian w/ dustbin), g (trained G, mutual softmax),
g_sinkhorn (unit-capacity Sinkhorn), g_hungarian (one-to-one Hungarian on mutual scores).
NOTE: the label targets are capacity-aware (many-to-one allowed), so 'g' is the mode that
matches the supervision; the other two G modes are one-to-one experiments."""
import json
import os

import numpy as np
import torch

from .labels import build_M_gt
from .soft import (row_col_probs, mutual, log_sinkhorn, weights_from_M, gt_probs)
from .hard import hungarian_probs, assign_from_scores


def load_threshold(ckpt_path, default=0.2):
    """Reads the threshold calibrated by train_g (checkpoints/g.pt.thr.json)."""
    p = ckpt_path + ".thr.json"
    if os.path.exists(p):
        with open(p) as f:
            return float(json.load(f)["thr"])
    return default


class GTCache:
    """Caches (p_row_t, p_col_t, binary_match) per (source, target) pair.
    binary_match[i,j] = M[i,j] / min(r_i, c_j) >= 0.5  (capacity-normalised GT pair).
    Size-capped (oldest entries evicted) so long runs don't eat RAM."""

    def __init__(self, device="cpu", max_items=30000):
        self.device, self.store, self.max_items = device, {}, max_items

    def _build(self, a, b):
        key = (a.name, b.name)
        if key not in self.store:
            M, M_aug, r, c = build_M_gt(a.names, b.names)
            pr, pc = gt_probs(M_aug, self.device)
            cap = np.minimum(r[:, None], c[None, :])
            binm = torch.as_tensor((M / cap) >= 0.5, device=self.device)
            if len(self.store) >= self.max_items:
                self.store.pop(next(iter(self.store)))
            self.store[key] = (pr, pc, binm)
        return self.store[key]

    def get(self, a, b):
        pr, pc, _ = self._build(a, b)
        return pr, pc

    def get_bin(self, a, b):
        return self._build(a, b)[2]


class CorrProvider:
    MODES = ("oracle", "hard", "g", "g_sinkhorn", "g_hungarian")

    def __init__(self, mode, G=None, thr=0.2, gt_cache=None, device="cpu"):
        assert mode in self.MODES
        self.mode, self.G, self.thr = mode, G, thr
        self.gt = gt_cache if gt_cache is not None else GTCache(device)
        if G is not None:
            G.eval()

    @torch.no_grad()
    def probs(self, a, b):
        """-> (p_row (Ja,Jb+1), p_col (Ja+1,Jb), M (Ja,Jb))"""
        if self.mode == "oracle":
            pr, pc = self.gt.get(a, b)
            return pr, pc, mutual(pr, pc)
        if self.mode == "hard":
            pr, pc = hungarian_probs(a.feats.cpu().numpy(), b.feats.cpu().numpy())
            pr, pc = pr.to(a.feats.device), pc.to(a.feats.device)
            return pr, pc, mutual(pr, pc)
        S = self.G(a.feats, a.pts, b.feats, b.pts)
        pr, pc = row_col_probs(S)
        if self.mode == "g":
            M = mutual(pr, pc)
        elif self.mode == "g_sinkhorn":
            M = log_sinkhorn(S)
        else:  # g_hungarian
            M = assign_from_scores(mutual(pr, pc), self.thr)
        return pr, pc, M

    def __call__(self, a, b):
        _, _, M = self.probs(a, b)
        W, matched = weights_from_M(M, self.thr)
        return W, matched, M