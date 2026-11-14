"""Geometric hard-matching baseline: Hungarian assignment on joint-feature distances."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def hungarian_probs(fa, fb):
    """fa (Ja,F), fb (Jb,F) numpy/tensors -> (p_row (Ja,Jb+1), p_col (Ja+1,Jb)) one-hot."""
    fa = np.asarray(fa, np.float64); fb = np.asarray(fb, np.float64)
    allf = np.concatenate([fa, fb], 0)
    mu, sd = allf.mean(0), allf.std(0) + 1e-6
    za, zb = (fa - mu) / sd, (fb - mu) / sd
    cost = np.linalg.norm(za[:, None, :] - zb[None, :, :], axis=-1)
    ri, ci = linear_sum_assignment(cost)
    Ja, Jb = len(fa), len(fb)
    p_row = np.zeros((Ja, Jb + 1), np.float32); p_row[:, -1] = 1
    p_col = np.zeros((Ja + 1, Jb), np.float32); p_col[-1, :] = 1
    for i, j in zip(ri, ci):
        p_row[i, :] = 0; p_row[i, j] = 1
        p_col[:, j] = 0; p_col[i, j] = 1
    return torch.from_numpy(p_row), torch.from_numpy(p_col)
