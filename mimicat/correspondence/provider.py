"""One place that answers: 'given source and target characters, which joints drive which?'
Modes: oracle (label-derived M_gt), hard (Hungarian), g (trained G, mutual softmax),
g_sinkhorn (trained G, Sinkhorn)."""
import torch

from .labels import build_M_gt
from .soft import (row_col_probs, mutual, log_sinkhorn, weights_from_M, gt_probs)
from .hard import hungarian_probs


class GTCache:
    def __init__(self, device="cpu"):
        self.device, self.store = device, {}

    def get(self, a, b):
        key = (a.name, b.name)
        if key not in self.store:
            _, M_aug, _, _ = build_M_gt(a.names, b.names)
            self.store[key] = gt_probs(M_aug, self.device)
        return self.store[key]


class CorrProvider:
    MODES = ("oracle", "hard", "g", "g_sinkhorn")

    def __init__(self, mode, G=None, thr=0.2, gt_cache=None, device="cpu"):
        assert mode in self.MODES
        self.mode, self.G, self.thr = mode, G, thr
        self.gt = gt_cache or GTCache(device)
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
        M = mutual(pr, pc) if self.mode == "g" else log_sinkhorn(S)
        return pr, pc, M

    def __call__(self, a, b):
        _, _, M = self.probs(a, b)
        W, matched = weights_from_M(M, self.thr)
        return W, matched, M
