"""From logits to soft correspondences, the G training loss, and threshold calibration."""
import math
import numpy as np
import torch
import torch.nn.functional as F


def row_col_probs(S_aug):
    """Row softmax over (targets + dustbin), column softmax over (sources + dustbin)."""
    p_row = F.softmax(S_aug[:-1, :], dim=1)     # (Js, Jt+1)
    p_col = F.softmax(S_aug[:, :-1], dim=0)     # (Js+1, Jt)
    return p_row, p_col


def mutual(p_row, p_col):
    """Real-block soft matching: geometric mean of both directions. (Js, Jt)"""
    return (p_row[:, :-1] * p_col[:-1, :]).clamp_min(0).sqrt()


def log_sinkhorn_full(S_aug, iters=30):
    """SuperGlue-style Sinkhorn with dustbin (unit capacities). Returns the full
    log-assignment matrix (Js+1, Jt+1); differentiable."""
    Js, Jt = S_aug.shape[0] - 1, S_aug.shape[1] - 1
    dt, dev = S_aug.dtype, S_aug.device
    log_mu = torch.cat([torch.zeros(Js, dtype=dt, device=dev),
                        torch.tensor([math.log(Jt)], dtype=dt, device=dev)])
    log_nu = torch.cat([torch.zeros(Jt, dtype=dt, device=dev),
                        torch.tensor([math.log(Js)], dtype=dt, device=dev)])
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(S_aug + v[None, :], dim=1)
        v = log_nu - torch.logsumexp(S_aug + u[:, None], dim=0)
    return S_aug + u[:, None] + v[None, :]


def log_sinkhorn(S_aug, iters=30):
    """Real block (Js, Jt) of the Sinkhorn assignment."""
    Js, Jt = S_aug.shape[0] - 1, S_aug.shape[1] - 1
    return log_sinkhorn_full(S_aug, iters).exp()[:Js, :Jt]


def weights_from_M(M, thr=0.2):
    """W[j,i]: how much source joint i drives target joint j. matched[j]: target is driven."""
    matched = M.max(0).values >= thr
    W = M.T / M.sum(0)[:, None].clamp_min(1e-8)
    return W * matched[:, None], matched


def gt_probs(M_aug, device="cpu"):
    """Training targets from build_M_gt's M_aug (numpy)."""
    M = torch.as_tensor(M_aug, dtype=torch.float32, device=device)
    p_row_t = M[:-1, :] / M[:-1, :].sum(1, keepdim=True).clamp_min(1e-8)
    p_col_t = M[:, :-1] / M[:, :-1].sum(0, keepdim=True).clamp_min(1e-8)
    return p_row_t, p_col_t


def corr_loss(S_aug, p_row_t, p_col_t, match_w=2.0, ot_w=0.0):
    """Row/col cross-entropy with matched rows/cols up-weighted by match_w
    (1.0 = old behaviour). Optional Sinkhorn NLL term (ot_w > 0)."""
    lr = F.log_softmax(S_aug[:-1, :], dim=1)
    lc = F.log_softmax(S_aug[:, :-1], dim=0)

    w_r = 1.0 + (match_w - 1.0) * (1.0 - p_row_t[:, -1])
    w_c = 1.0 + (match_w - 1.0) * (1.0 - p_col_t[-1, :])
    l_row = -((p_row_t * lr).sum(1) * w_r).sum() / w_r.sum()
    l_col = -((p_col_t * lc).sum(0) * w_c).sum() / w_c.sum()
    loss = l_row + l_col

    if ot_w > 0:
        logP = log_sinkhorn_full(S_aug)
        T = torch.zeros_like(logP)
        T[:-1, :] = p_row_t
        T[-1, :-1] = p_col_t[-1, :]
        loss = loss + ot_w * (-(T * logP).sum() / T.sum().clamp_min(1e-8))
    return loss


@torch.no_grad()
def best_threshold(Ms, gts, grid=None):
    """Pick the threshold on mutual scores that maximises micro-F1 of matched pairs.
    Ms: list of (Js,Jt) float tensors, gts: list of (Js,Jt) bool tensors.
    Returns (thr, f1, precision, recall)."""
    grid = np.arange(0.05, 0.80, 0.05) if grid is None else grid
    n_gt = sum(int(g.sum()) for g in gts)
    best = (0.2, -1.0, 0.0, 0.0)
    for t in grid:
        tp = sum(int(((m >= t) & g).sum()) for m, g in zip(Ms, gts))
        npred = sum(int((m >= t).sum()) for m in Ms)
        p, r = tp / max(npred, 1), tp / max(n_gt, 1)
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best[1]:
            best = (float(t), f1, p, r)
    return best