"""From logits to soft correspondences, plus the G training loss."""
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


def log_sinkhorn(S_aug, iters=30):
    """SuperGlue-style Sinkhorn with a dustbin (unit capacities). Returns real block (Js, Jt)."""
    Js, Jt = S_aug.shape[0] - 1, S_aug.shape[1] - 1
    dt, dev = S_aug.dtype, S_aug.device
    log_mu = torch.cat([torch.zeros(Js, dtype=dt, device=dev),
                        torch.tensor([float(np.log(Jt))], dtype=dt, device=dev)])
    log_nu = torch.cat([torch.zeros(Jt, dtype=dt, device=dev),
                        torch.tensor([float(np.log(Js))], dtype=dt, device=dev)])
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(S_aug + v[None, :], dim=1)
        v = log_nu - torch.logsumexp(S_aug + u[:, None], dim=0)
    return (S_aug + u[:, None] + v[None, :]).exp()[:Js, :Jt]


def weights_from_M(M, thr=0.2):
    """W[j,i]: how much source joint i drives target joint j. matched[j]: target joint is driven."""
    matched = M.max(0).values >= thr
    W = M.T / M.sum(0)[:, None].clamp_min(1e-8)
    return W * matched[:, None], matched


def gt_probs(M_aug, device="cpu"):
    """Training targets from build_M_gt's M_aug (numpy)."""
    M = torch.as_tensor(M_aug, dtype=torch.float32, device=device)
    p_row_t = M[:-1, :] / M[:-1, :].sum(1, keepdim=True)
    p_col_t = M[:, :-1] / M[:, :-1].sum(0, keepdim=True)
    return p_row_t, p_col_t


def corr_loss(S_aug, p_row_t, p_col_t):
    lr = F.log_softmax(S_aug[:-1, :], dim=1)
    lc = F.log_softmax(S_aug[:, :-1], dim=0)
    return -(p_row_t * lr).sum(1).mean() - (p_col_t * lc).sum(0).mean()
