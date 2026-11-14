"""Stage I: train the correspondence transformer G with label-derived targets.

    python -m mimicat.train.train_g --steps 3000 --out checkpoints/g.pt
"""
import argparse, time
import numpy as np
import torch

from ..paths import DATA_ROOT
from ..data.dataset import build_index
from ..data.tensors import CharStore
from ..models.corr_transformer import CorrTransformer
from ..correspondence.provider import GTCache
from ..correspondence.soft import corr_loss
from ..eval.eval_g import evaluate_pairs
from .common import (get_device, seed_all, load_split, names_in, sample_pair, all_pairs, save_ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="configs/subset.json")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--n-self", type=int, default=2)
    ap.add_argument("--n-cross", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/g.pt")
    a = ap.parse_args()

    seed_all(a.seed); dev = get_device(a.device)
    rng = np.random.default_rng(a.seed)
    split = load_split(a.subset)
    tr, va = names_in(split, "train"), names_in(split, "val")
    print(f"characters: train {len(tr)}  val {len(va)}  test {len(names_in(split, 'test'))}")
    store, gtc = CharStore(DATA_ROOT, dev), GTCache(dev)
    idx = build_index(DATA_ROOT)
    G = CorrTransformer(d=a.d, n_self=a.n_self, n_cross=a.n_cross).to(dev)
    opt = torch.optim.AdamW(G.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    val_pairs = all_pairs(va)
    if len(val_pairs) > 60:
        val_pairs = [val_pairs[i] for i in np.random.default_rng(1).choice(len(val_pairs), 60, replace=False)]
    best, t0 = -1.0, time.time()

    for step in range(1, a.steps + 1):
        G.train(); loss = 0.0
        for _ in range(a.batch):
            an, bn = sample_pair(rng, tr, p_self=0.1)
            A, B = store.get(an), store.get(bn)
            pr_t, pc_t = gtc.get(A, B)
            S = G(A.feats, A.pts, B.feats, B.pts)
            loss = loss + corr_loss(S, pr_t, pc_t)
        loss = loss / a.batch
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 50 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s")
        if step % a.eval_every == 0 or step == a.steps:
            G.eval()
            with torch.no_grad():
                res = evaluate_pairs(store, gtc, val_pairs, ["g", "hard"], G, idx)
            g, h = res["g"]["all"], res["hard"]["all"]
            print(f"  [val] G acc {g['acc']:.3f} f1 {g['f1']:.3f} | Hungarian acc {h['acc']:.3f} f1 {h['f1']:.3f}")
            if g["acc"] > best:
                best = g["acc"]; save_ckpt(a.out, G, step=step, val_acc=best)
                print(f"  saved {a.out} (val acc {best:.3f})")


if __name__ == "__main__":
    main()
