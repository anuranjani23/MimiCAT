"""Stage II: train the pose transfer transformer H with cycle consistency.

    python -m mimicat.train.train_h --corr oracle --steps 2000 --out checkpoints/h_oracle.pt
    python -m mimicat.train.train_h --corr g --g-ckpt checkpoints/g.pt --steps 2000 --out checkpoints/h.pt

Uses G's val-calibrated threshold (checkpoints/g.pt.thr.json) for the g* modes.
"""
import argparse
import math
import time
from collections import deque

import numpy as np
import torch
from tqdm import tqdm

from ..paths import DATA_ROOT
from ..data.tensors import CharStore
from ..models.pose_transformer import PoseTransformer
from ..correspondence.provider import GTCache, CorrProvider, load_threshold
from ..correspondence.soft import mutual
from ..eval.cycle import cycle_error, transfer_pose, vertex_error
from ..eval.metrics import label_agreement_deg
from .common import (get_device, seed_all, load_split, names_in, sample_pair, all_pairs,
                     save_ckpt, load_G)


def agree_loss(R_src, R_tgt, pr_t, pc_t, thr=0.4):
    """Optional weak supervision: confident label-matched joints should rotate alike."""
    best, si = mutual(pr_t, pc_t).max(0)
    sel = best >= thr
    if not sel.any():
        return R_tgt.sum() * 0
    return ((R_src[:, si[sel]] - R_tgt[:, sel]) ** 2).mean()


@torch.no_grad()
def validate(store, gtc, prov, H, pairs, T, seed=123):
    rng = np.random.default_rng(seed)
    out = {"cyc_H": [], "cyc_0": [], "agr_H": [], "agr_0": []}
    H.eval()
    for an, bn in pairs:
        A, B = store.get(an), store.get(bn)
        R_a = store.sample_motion(A, T, rng)
        pr_t, pc_t = gtc.get(A, B)
        e0, Rb0, _ = cycle_error(A, B, R_a, prov, H=None)
        e1, Rb1, _ = cycle_error(A, B, R_a, prov, H=H)
        out["cyc_0"].append(e0.item()); out["cyc_H"].append(e1.item())
        out["agr_0"].append(label_agreement_deg(R_a, Rb0, pr_t, pc_t))
        out["agr_H"].append(label_agreement_deg(R_a, Rb1, pr_t, pc_t))
    return {k: float(np.nanmean(v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="configs/subset.json")
    ap.add_argument("--corr", default="oracle", choices=list(CorrProvider.MODES))
    ap.add_argument("--g-ckpt", default="checkpoints/g.pt")
    ap.add_argument("--thr", type=float, default=None,
                    help="override match threshold (default: G's calibrated one for g* modes)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--w-id", type=float, default=0.5, help="A->A identity transfer loss")
    ap.add_argument("--w-prior", type=float, default=0.1, help="keep H close to the initialisation")
    ap.add_argument("--w-agree", type=float, default=0.0, help="optional label-agreement loss (ablation)")
    ap.add_argument("--val-pairs", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=0, help="evals without gain before stop; 0=off")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/h.pt")
    a = ap.parse_args()

    seed_all(a.seed); dev = get_device(a.device)
    rng = np.random.default_rng(a.seed)
    split = load_split(a.subset)
    tr, va = names_in(split, "train"), names_in(split, "val")
    store, gtc = CharStore(DATA_ROOT, dev), GTCache(dev)

    uses_G = a.corr.startswith("g")
    G = load_G(a.g_ckpt, dev) if uses_G else None
    if a.thr is not None:
        thr = a.thr
    else:
        thr = load_threshold(a.g_ckpt) if a.corr in ("g", "g_hungarian") else 0.2
    print(f"Correspondence: {a.corr} | match threshold: {thr:.2f}")
    prov = CorrProvider(a.corr, G=G, thr=thr, gt_cache=gtc, device=dev)

    H = PoseTransformer(d=a.d, n_layers=a.n_layers).to(dev)
    print(f"H params: {sum(p.numel() for p in H.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(H.parameters(), lr=a.lr, weight_decay=1e-4)

    def lr_lambda(s):
        if s < a.warmup:
            return (s + 1) / a.warmup
        prog = (s - a.warmup) / max(1, a.steps - a.warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    vp = all_pairs(va)
    if len(vp) > a.val_pairs:
        vp = [vp[i] for i in np.random.default_rng(1).choice(len(vp), a.val_pairs, replace=False)]
    print(f"Validation pairs: {len(vp)}")

    best, best_step, bad, t0 = 1e9, 0, 0, time.time()
    hist = {k: deque(maxlen=50) for k in ("loss", "cyc", "id", "prior", "agree")}

    pbar = tqdm(range(1, a.steps + 1), desc="Training H", dynamic_ncols=True,
                unit="step", mininterval=2.0)
    for step in pbar:
        H.train(); tot = 0.0
        parts = dict(cyc=0.0, id=0.0, prior=0.0, agree=0.0)
        for _ in range(a.batch):
            an, bn = sample_pair(rng, tr)
            A, B = store.get(an), store.get(bn)
            R_a = store.sample_motion(A, a.frames, rng)
            cyc, R_b, deltas = cycle_error(A, B, R_a, prov, H=H)
            R_aa, _, d_id, _ = transfer_pose(A, A, R_a, prov, H)
            l_id = vertex_error(A, R_a, R_aa)
            prior = sum((d ** 2).mean() for d in deltas + [d_id]) / (len(deltas) + 1)
            l_ag = torch.zeros((), device=dev)
            if a.w_agree > 0:
                pr_t, pc_t = gtc.get(A, B)
                l_ag = agree_loss(R_a, R_b, pr_t, pc_t)
            loss = cyc + a.w_id * l_id + a.w_prior * prior + a.w_agree * l_ag
            tot = tot + loss / a.batch
            for k, v in zip(parts, (cyc, l_id, prior, l_ag)):
                parts[k] += float(v.detach()) / a.batch

        opt.zero_grad()
        if not torch.isfinite(tot):
            continue
        tot.backward()
        torch.nn.utils.clip_grad_norm_(H.parameters(), 1.0)
        opt.step(); sched.step()

        hist["loss"].append(float(tot))
        for k, v in parts.items():
            hist[k].append(v)
        pbar.set_postfix(loss=f"{np.mean(hist['loss']):.4f}", cyc=f"{np.mean(hist['cyc']):.4f}",
                         id=f"{np.mean(hist['id']):.4f}", best=f"{best:.4f}")

        if step % a.eval_every == 0 or step == a.steps:
            r = validate(store, gtc, prov, H, vp, a.frames)
            tqdm.write(
                f"\nStep {step}/{a.steps} | {time.time() - t0:.0f}s\n"
                f"  [val] cycle: no-H {r['cyc_0']:.4f} -> H {r['cyc_H']:.4f} | "
                f"label-agreement deg: no-H {r['agr_0']:.1f} -> H {r['agr_H']:.1f}")
            if r["cyc_H"] < best:
                best, best_step, bad = r["cyc_H"], step, 0
                save_ckpt(a.out, H, step=step, val_cycle=best)
                tqdm.write(f"  [CHECKPOINT] {a.out} (val cycle {best:.4f})")
            else:
                bad += 1
                if a.patience and bad >= a.patience:
                    tqdm.write("Early stopping.")
                    break
    pbar.close()
    print("\nTraining completed.")
    print(f"Best val cycle error: {best:.4f} at step {best_step}")
    print(f"Total training time: {time.time() - t0:.1f}s")
    print(f"Best checkpoint: {a.out}")


if __name__ == "__main__":
    main()