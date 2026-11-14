"""Correspondence and pose-transfer metrics."""
import torch

from ..correspondence.soft import mutual
from ..correspondence.transfer import geodesic_deg


def _dir_metrics(p, pt):
    """p, pt: (N, K+1) predicted / target distributions, last column = dustbin."""
    pred = p.argmax(1)
    tmax = pt.max(1).values
    hit = pt.gather(1, pred[:, None])[:, 0] >= 0.5 * tmax          # any acceptable answer counts
    t_dust = pt[:, -1] >= 0.5
    p_dust = pred == p.shape[1] - 1
    tp = ((~t_dust) & (~p_dust) & hit).sum().item()
    prec = tp / max((~p_dust).sum().item(), 1)
    rec = tp / max((~t_dust).sum().item(), 1)
    return dict(
        acc=hit.float().mean().item(),
        acc_match=hit[~t_dust].float().mean().item() if (~t_dust).any() else float("nan"),
        acc_dust=hit[t_dust].float().mean().item() if t_dust.any() else float("nan"),
        f1=2 * prec * rec / max(prec + rec, 1e-8))


def corr_metrics(p_row, p_col, p_row_t, p_col_t):
    """Average of the source->target and target->source directions."""
    a = _dir_metrics(p_row, p_row_t)
    b = _dir_metrics(p_col.T.contiguous(), p_col_t.T.contiguous())
    out = {}
    for k in a:
        vals = [x for x in (a[k], b[k]) if x == x]
        out[k] = sum(vals) / len(vals) if vals else float("nan")
    return out


def label_agreement_deg(R_src, R_tgt, p_row_t, p_col_t, thr=0.4):
    """Mean geodesic angle (deg) between each target joint's rotation and the rotation of its
    label-matched source joint (only confident label matches). Independent of the cycle loss."""
    M = mutual(p_row_t, p_col_t)
    best, src_idx = M.max(0)
    sel = best >= thr
    if not sel.any():
        return float("nan")
    return geodesic_deg(R_src[:, src_idx[sel]], R_tgt[:, sel]).mean().item()
