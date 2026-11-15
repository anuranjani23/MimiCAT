"""Geometric hard-matching baselines.

hungarian_probs now supports a dustbin (reject matches whose cost is above tau),
which makes the baseline fair: v1 forced min(Ja, Jb) matches, which tanks accuracy
whenever some joints have no counterpart.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

_BIG = 1e6


def hungarian_probs(fa, fb, tau=None, tau_scale=1.5, reject=True):
    """fa (Ja,F), fb (Jb,F) -> (p_row (Ja,Jb+1), p_col (Ja+1,Jb)) one-hot.

    reject=False reproduces the old forced assignment.
    tau=None -> auto: tau_scale * median of per-row/col minimum costs.
    """
    fa = np.asarray(fa, np.float64)
    fb = np.asarray(fb, np.float64)
    allf = np.concatenate([fa, fb], 0)
    mu, sd = allf.mean(0), allf.std(0) + 1e-6
    za, zb = (fa - mu) / sd, (fb - mu) / sd
    cost = np.linalg.norm(za[:, None, :] - zb[None, :, :], axis=-1)
    Ja, Jb = len(fa), len(fb)

    if reject:
        if tau is None:
            mins = np.concatenate([cost.min(1), cost.min(0)])
            tau = tau_scale * float(np.median(mins))
        n = Ja + Jb
        C = np.zeros((n, n), np.float64)
        C[:Ja, :Jb] = cost
        C[:Ja, Jb:] = _BIG
        C[Ja:, :Jb] = _BIG
        C[np.arange(Ja), Jb + np.arange(Ja)] = tau
        C[Ja + np.arange(Jb), np.arange(Jb)] = tau
        ri, ci = linear_sum_assignment(C)
        pairs = [(i, j) for i, j in zip(ri, ci) if i < Ja and j < Jb]
    else:
        ri, ci = linear_sum_assignment(cost)
        pairs = list(zip(ri, ci))

    p_row = np.zeros((Ja, Jb + 1), np.float32); p_row[:, -1] = 1
    p_col = np.zeros((Ja + 1, Jb), np.float32); p_col[-1, :] = 1
    for i, j in pairs:
        p_row[i, :] = 0; p_row[i, j] = 1
        p_col[:, j] = 0; p_col[i, j] = 1
    return torch.from_numpy(p_row), torch.from_numpy(p_col)


def assign_from_scores(M, thr=0.2):
    """One-to-one Hungarian on a learned score matrix M (Js,Jt); drops pairs < thr.
    Returns a (Js,Jt) tensor holding the kept scores (0 elsewhere)."""
    Mn = M.detach().cpu().numpy().astype(np.float64)
    ri, ci = linear_sum_assignment(-Mn)
    out = np.zeros(Mn.shape, np.float32)
    for i, j in zip(ri, ci):
        if Mn[i, j] >= thr:
            out[i, j] = Mn[i, j]
    return torch.from_numpy(out).to(M.device)