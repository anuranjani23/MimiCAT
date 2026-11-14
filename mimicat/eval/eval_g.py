"""Evaluate the correspondence transformer G against the Hungarian baseline.

    python -m mimicat.eval.eval_g --ckpt checkpoints/g.pt --split test
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
from ..train.common import load_split, names_in, all_pairs, load_G, get_device
from .metrics import corr_metrics


def pair_type(idx, a, b):
    return "same_plan" if set(idx[a]["tags"]) == set(idx[b]["tags"]) else "cross_plan"


def evaluate_pairs(store, gtc, pairs, modes, G=None, idx=None):
    """-> {mode: {pair_type|'all': {metric: mean}}}"""
    provs = {m: CorrProvider(m, G=G, gt_cache=gtc, device=store.device) for m in modes}
    acc = {m: defaultdict(lambda: defaultdict(list)) for m in modes}
    for an, bn in pairs:
        a, b = store.get(an), store.get(bn)
        pr_t, pc_t = gtc.get(a, b)
        for m, prov in provs.items():
            pr, pc, _ = prov.probs(a, b)
            r = corr_metrics(pr, pc, pr_t, pc_t)
            keys = ["all"] + ([pair_type(idx, an, bn)] if idx else [])
            for k in keys:
                for mk, mv in r.items():
                    acc[m][k][mk].append(mv)
    return {m: {k: {mk: float(np.nanmean(v)) for mk, v in d.items()} for k, d in acc[m].items()}
            for m in modes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/g.pt")
    ap.add_argument("--subset", default="configs/subset.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-pairs", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/g_results.csv")
    a = ap.parse_args()
    dev = get_device(a.device)
    split = load_split(a.subset)
    names = names_in(split, a.split)
    pairs = all_pairs(names)
    rng = np.random.default_rng(0)
    if len(pairs) > a.max_pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), a.max_pairs, replace=False)]
    store, gtc = CharStore(DATA_ROOT, dev), GTCache(dev)
    idx = build_index(DATA_ROOT)
    G = load_G(a.ckpt, dev)
    res = evaluate_pairs(store, gtc, pairs, ["hard", "g"], G, idx)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "pair_type", "acc", "acc_match", "acc_dust", "f1"])
        for m, d in res.items():
            for k, r in d.items():
                w.writerow([m, k, *(f"{r[x]:.4f}" for x in ("acc", "acc_match", "acc_dust", "f1"))])
                print(f"{m:11s} {k:10s} acc {r['acc']:.3f}  match {r['acc_match']:.3f}  "
                      f"dust {r['acc_dust']:.3f}  f1 {r['f1']:.3f}")
    print("wrote", a.out, f"({len(pairs)} pairs, {len(names)} {a.split} characters)")


if __name__ == "__main__":
    main()
