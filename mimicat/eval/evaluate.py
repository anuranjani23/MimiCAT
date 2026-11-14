"""Pose-transfer ablation table: hard vs oracle vs G (+H), same-plan vs cross-plan pairs.

    python -m mimicat.eval.evaluate --g-ckpt checkpoints/g.pt --h-ckpt checkpoints/h.pt
"""
import argparse, csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from ..paths import DATA_ROOT
from ..data.dataset import build_index
from ..data.tensors import CharStore
from ..correspondence.provider import GTCache, CorrProvider
from ..train.common import load_split, names_in, all_pairs, load_G, load_H, get_device
from .cycle import cycle_error
from .metrics import label_agreement_deg
from .eval_g import pair_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="configs/subset.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--g-ckpt", default="checkpoints/g.pt")
    ap.add_argument("--h-ckpt", default="checkpoints/h.pt")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--max-pairs", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/results.csv")
    a = ap.parse_args()
    dev = get_device(a.device)
    names = names_in(load_split(a.subset), a.split)
    pairs = all_pairs(names)
    rng = np.random.default_rng(0)
    if len(pairs) > a.max_pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), a.max_pairs, replace=False)]
    store, gtc, idx = CharStore(DATA_ROOT, dev), GTCache(dev), build_index(DATA_ROOT)
    G = load_G(a.g_ckpt, dev) if Path(a.g_ckpt).exists() else None
    H = load_H(a.h_ckpt, dev) if Path(a.h_ckpt).exists() else None

    variants = [("hard", "hard", False), ("oracle", "oracle", False)]
    if G is not None:
        variants += [("G", "g", False), ("G+sinkhorn", "g_sinkhorn", False)]
    if H is not None:
        variants += [("oracle+H", "oracle", True)]
        if G is not None:
            variants += [("G+H", "g", True)]
    provs = {m: CorrProvider(m, G=G, gt_cache=gtc, device=dev) for _, m, _ in variants}

    acc = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for an, bn in pairs:
            A, B = store.get(an), store.get(bn)
            R_a = store.sample_motion(A, a.frames, rng)
            pr_t, pc_t = gtc.get(A, B)
            for name, mode, use_h in variants:
                e, R_b, _ = cycle_error(A, B, R_a, provs[mode], H=H if use_h else None)
                ag = label_agreement_deg(R_a, R_b, pr_t, pc_t)
                for k in ("all", pair_type(idx, an, bn)):
                    acc[(name, k)]["cycle"].append(e.item())
                    acc[(name, k)]["agree"].append(ag)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["variant", "pair_type", "n", "cycle_err", "label_agree_deg"])
        print(f"{'variant':12s} {'pairs':10s} {'n':>4s} {'cycle err':>10s} {'label agree (deg)':>18s}")
        for (name, k), d in sorted(acc.items(), key=lambda kv: (kv[0][1], [v[0] for v in variants].index(kv[0][0]))):
            row = [name, k, len(d["cycle"]), np.mean(d["cycle"]), np.nanmean(d["agree"])]
            w.writerow([row[0], row[1], row[2], f"{row[3]:.5f}", f"{row[4]:.2f}"])
            print(f"{row[0]:12s} {row[1]:10s} {row[2]:4d} {row[3]:10.5f} {row[4]:18.2f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
