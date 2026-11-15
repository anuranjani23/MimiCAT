"""Pose-transfer ablation table: hard vs oracle vs G (+H), same-plan vs cross-plan pairs.

    python -m mimicat.eval.evaluate --g-ckpt checkpoints/g.pt \
        --h-ckpt checkpoints/h.pt --h-oracle-ckpt checkpoints/h_oracle.pt

h.pt is the H trained on G's correspondences; h_oracle.pt is the H trained on oracle ones.
Match thresholds: G's calibrated threshold for g / g_hungarian, 0.2 otherwise.
"""
import argparse, csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from ..paths import DATA_ROOT
from ..data.dataset import build_index
from ..data.tensors import CharStore
from ..correspondence.provider import GTCache, CorrProvider, load_threshold
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
    ap.add_argument("--h-oracle-ckpt", default="checkpoints/h_oracle.pt")
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
    H_g = load_H(a.h_ckpt, dev) if Path(a.h_ckpt).exists() else None
    H_o = load_H(a.h_oracle_ckpt, dev) if Path(a.h_oracle_ckpt).exists() else None
    H_main = H_g if H_g is not None else H_o
    thr_g = load_threshold(a.g_ckpt) if G is not None else 0.2
    print(f"G threshold: {thr_g:.2f} | H(G-trained): {H_g is not None} | H(oracle-trained): {H_o is not None}")

    # (name, correspondence mode, H or None)
    variants = [("hard", "hard", None), ("oracle", "oracle", None)]
    if H_main is not None:
        variants.append(("hard+H", "hard", H_main))
    if H_o is not None:
        variants.append(("oracle+H", "oracle", H_o))
    if G is not None:
        variants.append(("G", "g", None))
        if H_g is not None:
            variants.append(("G+H", "g", H_g))
        if H_o is not None:
            variants.append(("G+H(orc)", "g", H_o))
        variants += [("G+sinkhorn", "g_sinkhorn", None), ("G+hungarian", "g_hungarian", None)]

    def thr_for(mode):
        return thr_g if mode in ("g", "g_hungarian") else 0.2

    provs = {m: CorrProvider(m, G=G, thr=thr_for(m), gt_cache=gtc, device=dev)
             for m in {m for _, m, _ in variants}}

    acc = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for an, bn in tqdm(pairs, desc=f"Evaluating ({a.split})", unit="pair", dynamic_ncols=True):
            A, B = store.get(an), store.get(bn)
            R_a = store.sample_motion(A, a.frames, rng)
            pr_t, pc_t = gtc.get(A, B)
            for name, mode, Hm in variants:
                e, R_b, _ = cycle_error(A, B, R_a, provs[mode], H=Hm)
                ag = label_agreement_deg(R_a, R_b, pr_t, pc_t)
                for k in ("all", pair_type(idx, an, bn)):
                    acc[(name, k)]["cycle"].append(e.item())
                    acc[(name, k)]["agree"].append(ag)

    order = [v[0] for v in variants]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "pair_type", "n", "cycle_err", "cycle_sem", "label_agree_deg"])
        print(f"\n{'variant':12s} {'pairs':10s} {'n':>4s} {'cycle err':>10s} {'± sem':>8s} {'label agree (deg)':>18s}")
        for (name, k), d in sorted(acc.items(), key=lambda kv: (kv[0][1] != "all", kv[0][1], order.index(kv[0][0]))):
            cy = np.asarray(d["cycle"])
            n, m, sem = len(cy), float(cy.mean()), float(cy.std() / np.sqrt(max(len(cy), 1)))
            ag = float(np.nanmean(d["agree"]))
            w.writerow([name, k, n, f"{m:.5f}", f"{sem:.5f}", f"{ag:.2f}"])
            print(f"{name:12s} {k:10s} {n:4d} {m:10.5f} {sem:8.5f} {ag:18.2f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()