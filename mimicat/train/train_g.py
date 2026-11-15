"""Stage I: train the correspondence transformer G with label-derived targets.

Example (quick check, ~20 min):
    python -m mimicat.train.train_g --steps 3000 --batch 16 --out checkpoints/g.pt
Full run:
    python -m mimicat.train.train_g --steps 8000 --batch 16 --d 128 --out checkpoints/g.pt
"""

import argparse
import copy
import json
import math
import time

import numpy as np
import torch
from tqdm import tqdm

from ..paths import DATA_ROOT
from ..data.dataset import build_index
from ..data.tensors import CharStore
from ..models.corr_transformer import CorrTransformer
from ..correspondence.provider import GTCache
from ..correspondence.soft import corr_loss, row_col_probs, mutual, best_threshold
from ..eval.eval_g import evaluate_pairs
from .common import (
    get_device,
    seed_all,
    load_split,
    names_in,
    sample_pair,
    all_pairs,
    save_ckpt,
    load_G,
)


@torch.no_grad()
def collect(G, store, gtc, pairs):
    """Mutual scores and BINARY ground-truth match matrices for (name_a, name_b) pairs."""
    Ms, gts = [], []
    for an, bn in pairs:
        A, B = store.get(an), store.get(bn)
        S = G(A.feats, A.pts, B.feats, B.pts)
        pr, pc = row_col_probs(S)
        Ms.append(mutual(pr, pc).cpu())
        gts.append(gtc.get_bin(A, B).cpu())
    return Ms, gts


def f1_at(Ms, gts, thr):
    n_gt = sum(int(g.sum()) for g in gts)
    tp = sum(int(((m >= thr) & g).sum()) for m, g in zip(Ms, gts))
    npred = sum(int((m >= thr).sum()) for m in Ms)
    p, r = tp / max(npred, 1), tp / max(n_gt, 1)
    return 2 * p * r / (p + r + 1e-9), p, r


@torch.no_grad()
def gt_stats(store, gtc, names, rng, n=200):
    """Fraction of source rows that are 'matched' (real mass / capacity >= 0.5)."""
    fr = []
    for _ in range(n):
        an, bn = sample_pair(rng, names, p_self=0.0)
        A, B = store.get(an), store.get(bn)
        pr_t, _ = gtc.get(A, B)
        fr.append(float(((1.0 - pr_t[:, -1]) >= 0.5).float().mean()))
    return float(np.mean(fr))


@torch.no_grad()
def ema_update(ema, G, decay):
    for pe, p in zip(ema.parameters(), G.parameters()):
        pe.mul_(decay).add_(p.detach(), alpha=1 - decay)
    for be, b in zip(ema.buffers(), G.buffers()):
        be.copy_(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="configs/subset.json")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--n-self", type=int, default=3)
    ap.add_argument("--n-cross", type=int, default=3)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--match-w", type=float, default=2.0,
                    help="weight of matched rows/cols vs dustbin rows/cols in the loss")
    ap.add_argument("--ot-w", type=float, default=0.0,
                    help="optional unit-capacity Sinkhorn NLL; conflicts with many-to-one "
                         "targets, leave at 0 unless experimenting")
    ap.add_argument("--feat-noise", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--val-pairs", type=int, default=150)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=0, help="evals without gain before stop; 0=off")
    ap.add_argument("--gt-cache", type=int, default=30000)
    ap.add_argument("--test-pairs", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/g.pt")
    a = ap.parse_args()

    seed_all(a.seed)
    dev = get_device(a.device)
    rng = np.random.default_rng(a.seed)

    split = load_split(a.subset)
    tr = names_in(split, "train")
    va = names_in(split, "val")
    te = names_in(split, "test")
    print(f"Characters: train {len(tr)} | val {len(va)} | test {len(te)}")

    store = CharStore(DATA_ROOT, dev)
    gtc = GTCache(dev, max_items=a.gt_cache)
    idx = build_index(DATA_ROOT)

    G = CorrTransformer(d=a.d, n_self=a.n_self, n_cross=a.n_cross, drop=a.drop).to(dev)
    ema = copy.deepcopy(G).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    print(f"Params: {sum(p.numel() for p in G.parameters()) / 1e6:.2f}M")

    opt = torch.optim.AdamW(G.parameters(), lr=a.lr, weight_decay=a.wd)

    def lr_lambda(s):
        if s < a.warmup:
            return (s + 1) / a.warmup
        prog = (s - a.warmup) / max(1, a.steps - a.warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    val_pairs = all_pairs(va)
    if len(val_pairs) > a.val_pairs:
        sel = np.random.default_rng(1).choice(len(val_pairs), a.val_pairs, replace=False)
        val_pairs = [val_pairs[i] for i in sel]
    print(f"Validation pairs: {len(val_pairs)}")
    print(f"GT matched-row fraction (train): {gt_stats(store, gtc, tr, rng):.3f}")

    best, best_step, bad = -1.0, 0, 0
    t0 = time.time()
    run_loss = []

    pbar = tqdm(range(1, a.steps + 1), desc="Training G", dynamic_ncols=True,
                unit="step", mininterval=2.0)

    for step in pbar:
        G.train()
        loss = 0.0
        for _ in range(a.batch):
            an, bn = sample_pair(rng, tr, p_self=0.1)
            A, B = store.get(an), store.get(bn)
            pr_t, pc_t = gtc.get(A, B)

            fa, fb = A.feats, B.feats
            if a.feat_noise > 0:
                fa = fa + a.feat_noise * torch.randn_like(fa)
                fb = fb + a.feat_noise * torch.randn_like(fb)

            S = G(fa, A.pts, fb, B.pts)
            loss = loss + corr_loss(S, pr_t, pc_t, match_w=a.match_w, ot_w=a.ot_w)
        loss = loss / a.batch

        opt.zero_grad()
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt.step()
        sched.step()
        ema_update(ema, G, min(a.ema, (1 + step) / (10 + step)))

        run_loss.append(loss.item())
        run_loss = run_loss[-100:]
        pbar.set_postfix(loss=f"{np.mean(run_loss):.3f}",
                         lr=f"{sched.get_last_lr()[0]:.1e}",
                         bestF1=f"{best:.3f}")

        if step % a.eval_every == 0 or step == a.steps:
            ema.eval()
            Ms, gts = collect(ema, store, gtc, val_pairs)
            thr, f1, prec, rec = best_threshold(Ms, gts)

            with torch.no_grad():
                res = evaluate_pairs(store, gtc, val_pairs, ["g", "hard"], ema, idx)
            g, h = res["g"]["all"], res["hard"]["all"]

            tqdm.write(
                f"\nStep {step}/{a.steps} | {time.time() - t0:.0f}s\n"
                f"[VAL] G(ema): acc={g['acc']:.3f} (match={g['acc_match']:.3f}, "
                f"dust={g['acc_dust']:.3f}) F1={g['f1']:.3f} | "
                f"F1@thr={f1:.3f} (thr={thr:.2f}, P={prec:.3f}, R={rec:.3f})\n"
                f"      Hungarian: acc={h['acc']:.3f} (match={h['acc_match']:.3f}, "
                f"dust={h['acc_dust']:.3f}) F1={h['f1']:.3f}"
            )

            if f1 > best:
                best, best_step, bad = f1, step, 0
                save_ckpt(a.out, ema, step=step, val_acc=g["acc"])
                with open(a.out + ".thr.json", "w") as f:
                    json.dump({"thr": thr, "val_f1": f1, "step": step}, f)
                tqdm.write(f"[CHECKPOINT] {a.out} (F1@thr={best:.3f}, thr={thr:.2f})")
            else:
                bad += 1
                if a.patience and bad >= a.patience:
                    tqdm.write("Early stopping.")
                    break

    pbar.close()
    print("\nTraining completed.")
    print(f"Best val F1@thr: {best:.3f} at step {best_step}")
    print(f"Total training time: {time.time() - t0:.1f}s")

    # ---- final held-out test with the best checkpoint and its val-calibrated threshold ----
    try:
        Gb = load_G(a.out, dev)
        with open(a.out + ".thr.json") as f:
            thr = json.load(f)["thr"]
        tp_ = all_pairs(te)
        if len(tp_) > a.test_pairs:
            sel = np.random.default_rng(0).choice(len(tp_), a.test_pairs, replace=False)
            tp_ = [tp_[i] for i in sel]
        Ms, gts = collect(Gb, store, gtc, tp_)
        f1, p, r = f1_at(Ms, gts, thr)
        with torch.no_grad():
            res = evaluate_pairs(store, gtc, tp_, ["g", "hard"], Gb, idx)
        g, h = res["g"]["all"], res["hard"]["all"]
        print(f"[TEST] G: F1@thr={f1:.3f} (P={p:.3f}, R={r:.3f}, thr={thr:.2f}) | "
              f"acc={g['acc']:.3f} match={g['acc_match']:.3f} dust={g['acc_dust']:.3f} "
              f"F1={g['f1']:.3f}")
        print(f"[TEST] Hungarian: acc={h['acc']:.3f} match={h['acc_match']:.3f} "
              f"dust={h['acc_dust']:.3f} F1={h['f1']:.3f}")
    except Exception as e:  # keep training result even if test eval fails
        print(f"(test evaluation skipped: {e})")

    print(f"Best checkpoint: {a.out} (threshold in {a.out}.thr.json)")


if __name__ == "__main__":
    main()